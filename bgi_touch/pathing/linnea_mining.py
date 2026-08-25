"""莉奈娅矿物扫描与瞄准挖矿。

BetterGI 的 ``LinneaMiningTask`` 使用 ``bgi_mine.onnx`` 找出画面中的矿物，
再把相邻检测框聚成矿堆，闭环修正瞄准点后攻击。这里保留同一套几何和
重试语义，但将鼠标输入映射到 iOS 的瞄准键、元素视野和攻击按钮。

该任务只在一次路点动作期间读取截图，不创建后台截图线程；因此可以和
PathingExecutor/TriggerLoop 共用 DeviceHub 的截图通道。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from ..vision.yolo import Detection, YoloPredictor


@dataclass
class MineralCluster:
    """A group of nearby mineral detections and its current aim target."""

    first: Detection
    area_ratio_threshold: float = 4.0
    prefer_right: bool = True
    detections: list[Detection] = field(default_factory=list)
    center_x: float = 0.0
    center_y: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    target_width: float = 0.0
    target_height: float = 0.0

    def __post_init__(self) -> None:
        self.detections.append(self.first)
        self._recalculate()

    @staticmethod
    def _center(detection: Detection) -> tuple[float, float]:
        return detection.center

    @staticmethod
    def _area(detection: Detection) -> float:
        return max(0.0, detection.width) * max(0.0, detection.height)

    def try_add(self, detection: Detection) -> bool:
        average_area = sum(self._area(item) for item in self.detections) / len(
            self.detections
        )
        area = self._area(detection)
        if area > average_area * self.area_ratio_threshold:
            return False
        if area < average_area / self.area_ratio_threshold:
            return False
        self.detections.append(detection)
        self._recalculate()
        return True

    def _recalculate(self) -> None:
        centers = [self._center(item) for item in self.detections]
        self.center_x = sum(item[0] for item in centers) / len(centers)
        self.center_y = sum(item[1] for item in centers) / len(centers)
        ordered = sorted(
            self.detections,
            key=lambda item: (
                (item.center[0] - self.center_x) ** 2
                + (item.center[1] - self.center_y) ** 2
            ),
        )
        candidates = ordered[:2] if self.prefer_right and len(ordered) >= 2 else ordered[:1]
        target = max(candidates, key=lambda item: item.center[0])
        self.target_x, self.target_y = target.center
        self.target_width = target.width
        self.target_height = target.height


def cluster_minerals(
    detections: Iterable[Detection],
    *,
    width_scale: float = 1.0,
    area_ratio_threshold: float = 4.0,
    prefer_right: bool = True,
) -> list[MineralCluster]:
    """Greedily cluster ore boxes using BetterGI's scale-aware threshold."""

    width_scale = max(0.01, float(width_scale))
    distance_threshold = 400.0 * width_scale
    reference_area = 1800.0 * width_scale * width_scale
    clusters: list[MineralCluster] = []
    for detection in detections:
        nearest: MineralCluster | None = None
        nearest_distance = math.inf
        for cluster in clusters:
            distance = math.hypot(
                detection.center[0] - cluster.center_x,
                detection.center[1] - cluster.center_y,
            )
            if distance < nearest_distance:
                nearest_distance = distance
                nearest = cluster
        if nearest is not None:
            average_area = sum(
                max(0.0, item.width) * max(0.0, item.height)
                for item in nearest.detections
            ) / len(nearest.detections)
            detection_area = max(0.0, detection.width) * max(0.0, detection.height)
            effective_threshold = distance_threshold * math.sqrt(
                (average_area * len(nearest.detections) + detection_area)
                / (len(nearest.detections) + 1)
                / max(1.0, reference_area)
            )
            if nearest_distance < effective_threshold and nearest.try_add(detection):
                continue
        clusters.append(
            MineralCluster(
                detection,
                area_ratio_threshold=area_ratio_threshold,
                prefer_right=prefer_right,
            )
        )
    return clusters


def parse_linnea_mining_params(value: object) -> tuple[int, int]:
    """Parse ``mines,rounds`` and ``mines=3,rounds=10`` route parameters."""

    mine_count: int | None = None
    scan_rounds: int | None = None

    def clamp(number: int) -> int:
        return max(1, min(999, number))

    for part in str(value or "").replace("，", ",").split(","):
        item = part.strip()
        lowered = item.casefold()
        try:
            if lowered.startswith("mines="):
                mine_count = clamp(int(item.split("=", 1)[1]))
            elif lowered.startswith("rounds="):
                scan_rounds = clamp(int(item.split("=", 1)[1]))
            elif mine_count is None:
                mine_count = clamp(int(item))
            elif scan_rounds is None:
                scan_rounds = clamp(int(item))
        except (TypeError, ValueError):
            continue

    mine_count = mine_count or 1
    scan_rounds = scan_rounds or 1
    return mine_count, max(mine_count, scan_rounds)


class LinneaMiningTask:
    """Run a bounded, single-threaded mineral scan at one pathing waypoint."""

    DEFAULT_MINE_COUNT = 1
    DEFAULT_SCAN_ROUNDS = 1
    CONFIDENCE_THRESHOLD = 0.70
    MAX_INNER_RETRY = 7
    ELEMENT_SIGHT_REFRESH_S = 3.0
    BASE_CLUSTER_DISTANCE = 400.0
    BASE_EDGE_IGNORE = 200.0
    ALIGNMENT_EXPANSION = 3.0
    AIM_SENSITIVITY_X = 0.45
    AIM_SENSITIVITY_Y = 0.80
    LEFT_TURN_STEP = -250.0

    def __init__(
        self,
        ctx,
        *,
        scan_rounds: int = DEFAULT_SCAN_ROUNDS,
        mine_count: int = DEFAULT_MINE_COUNT,
        predictor: YoloPredictor | None = None,
        log: Callable[[str], None] = print,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.ctx = ctx
        self.scan_rounds = max(1, min(999, int(scan_rounds)))
        self.mine_count = max(1, min(999, int(mine_count)))
        self.scan_rounds = max(self.scan_rounds, self.mine_count)
        self.predictor = predictor
        self.log = log
        self.clock = clock
        self._last_refresh_at = 0.0

    def _ensure_predictor(self) -> YoloPredictor:
        if self.predictor is None:
            self.predictor = YoloPredictor("bgi_mine")
        return self.predictor

    def _cancelled(self, cancelled: Callable[[], bool] | None) -> bool:
        return bool(cancelled and cancelled())

    def _scale(self, frame) -> tuple[float, float]:
        height, width = frame.shape[:2]
        return max(0.01, width / 1920.0), max(0.01, height / 1080.0)

    def _find_cluster(self, frame) -> tuple[MineralCluster | None, float, float]:
        width_scale, _height_scale = self._scale(frame)
        detections = self._ensure_predictor().predict(
            frame, conf_threshold=self.CONFIDENCE_THRESHOLD
        )
        # bgi_mine contains the single ``ore`` class. Keep the explicit class
        # filter so a future multi-class model cannot make aim corrections on
        # unrelated objects.
        ore = [
            detection
            for detection in detections
            if detection.confidence >= self.CONFIDENCE_THRESHOLD
            and int(getattr(detection, "class_id", 0)) == 0
        ]
        clusters = cluster_minerals(
            ore,
            width_scale=width_scale,
            prefer_right=self.scan_rounds > 1,
        )
        height, width = frame.shape[:2]
        center_x, center_y = width / 2.0, height / 2.0
        if not clusters:
            return None, center_x, center_y
        edge_ignore = self.BASE_EDGE_IGNORE * width_scale
        center_clusters = [
            cluster
            for cluster in clusters
            if edge_ignore <= cluster.center_x <= width - edge_ignore
            and edge_ignore <= cluster.center_y <= height - edge_ignore
        ]
        candidates = center_clusters or clusters
        nearest = min(
            candidates,
            key=lambda cluster: (cluster.center_x - center_x) ** 2
            + (cluster.center_y - center_y) ** 2,
        )
        return nearest, center_x, center_y

    def _sight_down(self) -> None:
        self.ctx.input.button_down("elementalSight")

    def _sight_up(self) -> None:
        self.ctx.input.button_up("elementalSight")

    def _align_and_mine(
        self,
        cluster: MineralCluster,
        center_x: float,
        center_y: float,
        *,
        width_scale: float,
        height_scale: float,
        cancelled: Callable[[], bool] | None,
    ) -> tuple[bool, bool, int, int]:
        total_dx = 0
        total_dy = 0
        had_result = True
        for retry in range(self.MAX_INNER_RETRY):
            if self._cancelled(cancelled):
                return False, False, total_dx, total_dy
            if self.clock() - self._last_refresh_at >= self.ELEMENT_SIGHT_REFRESH_S:
                self._sight_up()
                self.ctx.sleep(100)
                self._sight_down()
                self.ctx.sleep(1500)
                self._last_refresh_at = self.clock()

            offset_x = cluster.target_x - center_x
            offset_y = cluster.target_y - center_y
            aligned = (
                abs(offset_x) <= (cluster.target_width + self.ALIGNMENT_EXPANSION * 2) / 2
                and abs(offset_y) <= (cluster.target_height + self.ALIGNMENT_EXPANSION * 2) / 2
            )
            if aligned or (retry == self.MAX_INNER_RETRY - 1 and had_result):
                self._sight_up()
                self.ctx.sleep(300)
                if total_dy < 0:
                    self.ctx.input.move_camera_by(0, -25)
                    self.ctx.sleep(10)
                self.log("[pathing] 莉奈娅瞄准矿物，执行攻击")
                self.ctx.input.attack()
                self.ctx.sleep(2000)
                return True, aligned, 0, 0

            mouse_dx = int(offset_x * self.AIM_SENSITIVITY_X / max(0.01, width_scale))
            mouse_dy = int(offset_y * self.AIM_SENSITIVITY_Y / max(0.01, height_scale))
            self.ctx.input.move_camera_by(mouse_dx, mouse_dy)
            total_dx += mouse_dx
            total_dy += mouse_dy
            self.ctx.sleep(150)
            frame = self.ctx.capture_bgr()
            refreshed, center_x, center_y = self._find_cluster(frame)
            if refreshed is None:
                had_result = False
                return False, False, total_dx, total_dy
            cluster = refreshed
            width_scale, height_scale = self._scale(frame)
        return False, False, total_dx, total_dy

    def run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        """Run the scan and always restore aiming/elemental-sight state."""

        aiming_mode = False
        mined_count = 0
        try:
            self._ensure_predictor()
            self.ctx.input.key_press("R")
            aiming_mode = True
            self.ctx.sleep(400)
            for round_index in range(self.scan_rounds):
                if self._cancelled(cancelled):
                    return False
                self._sight_down()
                self.ctx.sleep(1500)
                self._last_refresh_at = self.clock()
                frame = self.ctx.capture_bgr()
                cluster, center_x, center_y = self._find_cluster(frame)
                if cluster is not None:
                    width_scale, height_scale = self._scale(frame)
                    aligned, counted, compensate_dx, compensate_dy = self._align_and_mine(
                        cluster,
                        center_x,
                        center_y,
                        width_scale=width_scale,
                        height_scale=height_scale,
                        cancelled=cancelled,
                    )
                    if aligned:
                        if counted:
                            mined_count += 1
                        if mined_count >= self.mine_count:
                            return True
                        continue
                    self._sight_up()
                    self.ctx.sleep(300)
                    if compensate_dx or compensate_dy:
                        self._sight_down()
                        self.ctx.sleep(1500)
                        self._last_refresh_at = self.clock()
                        self.ctx.input.move_camera_by(-compensate_dx, -compensate_dy)
                        self.ctx.sleep(800)
                        self._sight_up()
                        self.ctx.sleep(300)
                else:
                    self._sight_up()
                    self.ctx.sleep(300)
                if round_index + 1 < self.scan_rounds:
                    width_scale, _height_scale = self._scale(frame)
                    self.ctx.input.move_camera_by(
                        self.LEFT_TURN_STEP * width_scale, 0
                    )
                    self.ctx.sleep(800)
            return True
        except FileNotFoundError as error:
            self.log(f"[pathing] 莉奈娅矿物模型不可用：{error}")
            return False
        finally:
            try:
                self._sight_up()
            finally:
                if aiming_mode:
                    self.ctx.input.key_press("R")
