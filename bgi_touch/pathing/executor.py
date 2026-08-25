"""pathing 执行框架。

原版 PathExecutor 的核心依赖是小地图定位（SIFT/模板匹配到大地图，地图特征
数据以 NuGet 包 BetterGI.Assets.Map 分发，不在脚本仓库内）。本移植版把定位
抽象为 Positioner 协议：
- 提供 Positioner 时，执行器可完整走点（转向 + 前进 + 到点触发 action）；
- 未提供时，teleport/寻路会抛出明确的 NotImplementedError，但文件解析、
  校验与 action 执行（战斗 DSL 等）可用。

相机朝向检测使用 BetterGI 的极坐标边缘算法，并在低置信度时回退到
兼容的亮度扇区估计，供转向闭环使用。
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable, Optional, Protocol

import numpy as np

from ..engine.context import GameContext
from .actions import PathingActionRunner
from .farming import (
    FarmingRouteInfo,
    FarmingSession,
    FarmingStatsRecorder,
)
from .model import PathingTask, Waypoint
from .camera import crop_minimap_for_orientation, orientation_with_confidence
from .trap_escaper import StuckDetector, TrapEscaper


class Positioner(Protocol):
    def get_position(self, bgr: np.ndarray) -> Optional[tuple[float, float]]:
        """当前帧 → 世界地图坐标；无法定位返回 None。"""
        ...


def camera_orientation_deg(ctx: GameContext, bgr: np.ndarray) -> Optional[float]:
    """从小地图视野扇形估计相机朝向（度，地图北为 0，顺时针）。

    使用 BetterGI 的极坐标边缘峰值算法；弱纹理帧自动回退到兼容的
    圆环亮度估计。小地图位置来自布局 profile 的 minimap 配置。
    """
    crop = crop_minimap_for_orientation(ctx, bgr)
    if crop is None:
        return None
    angle, _confidence = orientation_with_confidence(crop)
    return angle


class PathingExecutor:
    def __init__(self, ctx: GameContext, positioner: Positioner | None = None,
                 party_slots: dict[str, int] | None = None,
                 log: Callable[[str], None] = print, map_name: str = "Teyvat",
                 farming_config_path: str | Path | None = None,
                 farming_route_info: dict | FarmingRouteInfo | None = None,
                 farming_recorder: FarmingStatsRecorder | None = None):
        self.ctx = ctx
        self.positioner = positioner
        self._map_name = map_name
        self.log = log
        self._tp_task = None
        self._cam_gain = 5.5
        self._cam_sign = 1.0
        self._last_mode_action_at = 0.0
        # BetterGI keeps this counter on PathExecutor, so repeated traps in
        # different waypoints still cause a route retry after the third one.
        self._stuck_detector = StuckDetector()
        self.actions = PathingActionRunner(ctx, party_slots=party_slots, log=log)
        self.farming = farming_recorder or FarmingStatsRecorder(
            farming_config_path, log=log
        )
        self.farming_route_info = (
            farming_route_info
            if isinstance(farming_route_info, FarmingRouteInfo)
            else FarmingRouteInfo.from_mapping(farming_route_info)
        )

    def _route_farming_info(self, task: PathingTask) -> FarmingRouteInfo:
        configured = self.farming_route_info
        source = Path(task.source_path) if task.source_path else None
        group_name = configured.group_name or str(
            task.info.get("group_name", task.info.get("groupName", "")) or ""
        )
        project_name = configured.project_name or (
            source.name if source is not None else task.name
        )
        folder_name = configured.folder_name or (
            source.parent.name if source is not None else str(
                task.info.get("folder_name", task.info.get("folderName", "")) or ""
            )
        )
        return FarmingRouteInfo(group_name, project_name, folder_name)

    def _ensure_positioner(self, map_name: str) -> None:
        from .map_locator import resolve_map_name

        resolved_name = resolve_map_name(map_name)
        previous_name = resolve_map_name(self._map_name)
        self._map_name = resolved_name
        if previous_name != resolved_name:
            self._tp_task = None
        if self.positioner is not None:
            current_name = getattr(self.positioner, "map_name", None)
            # A caller-supplied Positioner without map metadata remains
            # authoritative. Auto positioners advertise map_name and can be
            # replaced safely when one executor runs routes across maps.
            if not isinstance(current_name, str) or resolve_map_name(current_name) == resolved_name:
                return
        try:
            from .positioner import MinimapPositioner
            self.positioner = MinimapPositioner(self.ctx, resolved_name)
            self.log(f"[pathing] 已加载 {resolved_name} 地图定位（SIFT）")
        except FileNotFoundError as e:
            self.log(f"[pathing] 无地图定位：{e}")

    @staticmethod
    def _realtime_trigger_name(name: object) -> str:
        """Normalize route trigger aliases to GameContext names."""

        value = str(name or "").strip()
        aliases = {
            "AutoFishing": "AutoFish",
            "自动钓鱼": "AutoFish",
            "自动吃药": "AutoEat",
            "地图遮罩": "MapMask",
            "技能冷却": "SkillCd",
            "自动开门": "GameLoading",
            "快速传送": "QuickTeleport",
        }
        return aliases.get(value, value)

    def _enable_realtime_triggers(self, task: PathingTask) -> tuple[list[str], list[object] | None]:
        enabled: list[str] = []
        if not hasattr(self.ctx, "enable_trigger"):
            return enabled, None
        loop = getattr(self.ctx, "_trigger_loop", None)
        previous = list(loop.triggers) if loop is not None else None
        supported = {
            "AutoPick", "AutoSkip", "AutoEat", "MapMask", "SkillCd",
            "GameLoading", "QuickTeleport", "AutoFish",
        }
        for raw_name, active in task.realtime_triggers.items():
            if not active:
                continue
            name = self._realtime_trigger_name(raw_name)
            if name not in supported:
                self.log(f"[pathing] 实时触发器 {raw_name} 暂不支持")
                continue
            kwargs = {}
            if name == "MapMask":
                kwargs = {"map_name": task.map_name, "mini_map_enabled": True}
            elif name == "SkillCd":
                kwargs = {"party_slots": self.actions.party_slots or None}
            try:
                self.ctx.enable_trigger(name, **kwargs)
                enabled.append(name)
            except Exception as error:
                # Realtime triggers are optional route helpers.  Missing map
                # assets or an older host must not prevent the route itself
                # from running; keep the failure visible in the task log.
                self.log(f"[pathing] 启用实时触发器 {raw_name} 失败：{error}")
        return enabled, previous

    def _clear_realtime_triggers(
        self,
        state: tuple[list[str], list[object] | None],
    ) -> None:
        enabled, previous = state
        if not enabled:
            return
        loop = getattr(self.ctx, "_trigger_loop", None)
        if loop is None:
            return
        if previous is None:
            loop.clear()
        else:
            loop.replace(previous)
            if previous:
                loop.start()

    def run(self, task: PathingTask) -> bool:
        task.validate()
        self.log(f"[pathing] {task.name}: {len(task.positions)} 个路点 @ {task.map_name}")
        farming_session = FarmingSession.from_mapping(task.farming_info)
        decision = self.farming.check_limit(farming_session)
        if decision.skip:
            route_name = task.name or self._route_farming_info(task).project_name
            self.log(f"[pathing] {route_name}:{decision.message}，跳过此任务")
            return True
        if task.map_match_method.lower() not in {"sift"}:
            self.log(f"[pathing] {task.map_match_method} 暂未移植，回退 SIFT")
        self._ensure_positioner(task.map_name)
        trigger_state = self._enable_realtime_triggers(task)
        retry_count = self._retry_count(task)
        success = False
        try:
            for index, wp in enumerate(task.positions, start=1):
                self.log(f"[pathing] 路点 {index}/{len(task.positions)} id={wp.id}")

                # BetterGI handles four-leaf seals before normal movement. The
                # waypoint itself is a camera target, not a walking target.
                if wp.action == "up_down_grab_leaf":
                    self._run_with_retry(lambda: self._face_to(wp), wp, retry_count)
                    self._do_action(wp)
                    continue

                if wp.action == "log_output":
                    self._do_action(wp)

                if wp.type == "teleport" or wp.action == "force_tp":
                    self._run_with_retry(lambda: self._teleport(wp), wp, retry_count)
                elif wp.type == "orientation":
                    self._run_with_retry(lambda: self._face_to(wp), wp, retry_count)
                elif wp.type in ("path", "target"):
                    self._run_with_retry(
                        lambda: self._move_to(wp, arrive_dist=2.0 if wp.type == "target" else 4.0),
                        wp,
                        retry_count,
                    )
                else:
                    self.log(f"[pathing] 未知路点类型 {wp.type}，跳过移动")

                if wp.action and wp.action not in {"force_tp", "log_output"}:
                    self._do_action(wp)
            success = True
            return True
        finally:
            self.ctx.input.release_all()
            self._clear_realtime_triggers(trigger_state)
            if success and farming_session.allow_farming_count:
                try:
                    self.farming.record(
                        farming_session, self._route_farming_info(task)
                    )
                except Exception as error:
                    self.log(f"[farming] 锄地进度记录失败：{error}")

    @staticmethod
    def _retry_count(task: PathingTask) -> int:
        raw = task.config.get("retry_times", task.config.get("retryTimes", 1))
        try:
            return max(0, min(5, int(raw)))
        except (TypeError, ValueError):
            return 1

    def _run_with_retry(self, operation, waypoint: Waypoint, retries: int) -> None:
        for attempt in range(retries + 1):
            try:
                operation()
                return
            except TimeoutError as e:
                if attempt >= retries:
                    raise
                self.log(
                    f"[pathing] 路点 {waypoint.id} 失败，重试 {attempt + 1}/{retries}: {e}"
                )
                if self.positioner is not None:
                    reset = getattr(self.positioner, "reset", None)
                    if callable(reset):
                        reset()
                self.ctx.input.release_all()
                self.ctx.sleep(800)

    # ---- movement ----

    def _teleport(self, wp: Waypoint) -> None:
        if self._tp_task is None:
            from .tp import TpTask
            self._tp_task = TpTask(self.ctx, log=self.log, map_name=self._map_name)
        self._tp_task.tp(wp.x, wp.y, force=wp.action == "force_tp")
        if self.positioner is not None:
            # 传送落点≈目标锚点：直接设为局部搜索先验（白天/城内全局匹配不稳）
            if hasattr(self.positioner, "set_prior"):
                self.positioner.set_prior(wp.x, wp.y)
            else:
                reset = getattr(self.positioner, "reset", None)
                if callable(reset):
                    reset()

    # 相机反馈增益（px/°）与符号：位移反馈自校准（见 _move_to）
    @staticmethod
    def _bearing(dx: float, dy: float) -> float:
        """世界坐标位移 → 罗盘方位角（北=0，顺时针）。

        原神世界坐标轴向：+x=地图西，+y=地图北（img = origin − 2·world）。
        """
        return math.degrees(math.atan2(-dx, dy)) % 360

    def _get_position(self, frame: np.ndarray) -> Optional[tuple[float, float]]:
        if self.positioner is None:
            return None
        stable = getattr(self.positioner, "get_position_stable", None)
        if callable(stable):
            return stable(frame)
        return self.positioner.get_position(frame)

    def _face_to(self, wp: Waypoint) -> None:
        if self.positioner is None:
            self.log("[pathing] 方位点缺少地图定位，跳过朝向")
            return
        frame = self.ctx.capture_bgr()
        position = self._get_position(frame)
        if position is None:
            self.log("[pathing] 方位点定位失败，跳过朝向")
            return
        desired = self._bearing(wp.x - position[0], wp.y - position[1])
        current = camera_orientation_deg(self.ctx, frame)
        if current is None:
            self.log("[pathing] 方位点无法识别当前视角")
            return
        delta = (desired - current + 540) % 360 - 180
        if abs(delta) > 10:
            self.ctx.input.move_camera_by(self._cam_sign * delta * self._cam_gain, 0)
            self.ctx.sleep(500)

    @staticmethod
    def _misidentification_type_names(wp: Waypoint) -> set[str]:
        return {
            str(value).strip().replace("_", "").casefold()
            for value in wp.misidentification.types
            if str(value).strip()
        }

    def _position_from_big_map(self) -> tuple[float, float] | None:
        """Read the current player position from the full-screen map.

        BetterGI opens the map for ``handlingMode=mapRecognition`` and closes
        it again before returning to movement.  Reuse ``TpTask``'s map
        locator and trigger exclusion so this fallback does not click through
        the map or start a competing AutoPick/AutoSkip input path.
        """

        from .tp import TpTask

        task = self._tp_task
        opened = False
        try:
            if task is None:
                task = TpTask(self.ctx, log=self.log, map_name=self._map_name)
                self._tp_task = task
            with task.exclusive_triggers():
                opened = task.open_map()
                try:
                    if not opened:
                        return None
                    view = task.big.locate_view(self.ctx.capture_bgr())
                    if view is None:
                        return None
                    return task.big.feature_to_world(view[0], view[1])
                finally:
                    # Keep the map close inside exclusive_triggers so an
                    # AutoPick/AutoSkip tick cannot consume the ESC transition
                    # or press an interaction key on the closing frame.
                    if opened:
                        try:
                            self.ctx.input.key_press("ESCAPE")
                            self.ctx.sleep(500)
                        except Exception as error:
                            self.log(f"[pathing] 关闭异常识别地图失败：{error}")
        except Exception as error:
            self.log(f"[pathing] 大地图异常识别失败：{error}")
            return None

    def _resolve_misidentified_position(
        self,
        wp: Waypoint,
        raw_position: tuple[float, float] | None,
        last_good_position: tuple[float, float] | None,
        raw_distance: float | None,
        *,
        last_map_recognition_at: float,
        now: float,
    ) -> tuple[tuple[float, float] | None, float]:
        """Apply BetterGI's configured fallback for one bad position fix."""

        types = self._misidentification_type_names(wp)
        trigger = (
            "unrecognized"
            if raw_position is None
            else "pathtoofar"
            if raw_distance is not None and raw_distance > 500
            else None
        )
        if trigger is None or trigger not in types:
            return raw_position, last_map_recognition_at

        mode = str(wp.misidentification.handling_mode or "").strip().casefold()
        if mode == "previousdetectedpoint":
            if last_good_position is not None:
                self.log("[pathing] 未识别到具体路径，取上次点位")
                return last_good_position, last_map_recognition_at
            # No previous fix exists yet.  Treating a missing minimap as
            # (0,0) would steer the character toward the map origin, so let
            # the normal lost-fix retry path handle it.
            return None, last_map_recognition_at

        if mode == "maprecognition":
            # Opening/closing the full map is expensive on the iPhone.  A
            # short cooldown is enough to avoid repeating it for every stale
            # screenshot while still matching the upstream fallback intent.
            if now - last_map_recognition_at < 2.0:
                return last_good_position, last_map_recognition_at
            position = self._position_from_big_map()
            if position is not None:
                self.log(
                    f"[pathing] 未识别到具体路径，使用大地图中心 ({position[0]:.0f},{position[1]:.0f})"
                )
                return position, now
            return last_good_position, now

        # ScheduledArrival is present in upstream route JSON but currently has
        # no executable behavior there either.  Preserve the last known point
        # for all unknown modes instead of steering from a zero coordinate.
        if last_good_position is not None:
            self.log(f"[pathing] 未识别处理模式 {wp.misidentification.handling_mode}，取上次点位")
            return last_good_position, last_map_recognition_at
        return None, last_map_recognition_at

    def _move_to(self, wp: Waypoint, timeout_s: float = 120, arrive_dist: float = 3.0) -> None:
        if self.positioner is None:
            raise NotImplementedError(
                "寻路需要小地图定位（Positioner）。当前未配置地图特征资产，"
                "参见 docs/ROADMAP.md。可先使用战斗/宏/JS 脚本能力。"
            )
        deadline = time.monotonic() + timeout_s
        moving = False
        last_fix: tuple[float, float, float] | None = None  # (x, y, t)
        prev_err: float | None = None
        last_stuck_sample_at = 0.0
        last_map_recognition_at = float("-inf")
        last_good_position: tuple[float, float] | None = None
        lost_fixes = 0
        reached = False
        try:
            while time.monotonic() < deadline:
                try:
                    frame = self.ctx.capture_bgr()
                except Exception as e:  # 设备偶发超时，重试
                    self.log(f"[pathing] 截图失败重试: {e}")
                    self.ctx.sleep(1000)
                    continue
                now = time.monotonic()
                raw_pos = self._get_position(frame)
                raw_distance = (
                    math.hypot(wp.x - raw_pos[0], wp.y - raw_pos[1])
                    if raw_pos is not None else None
                )
                pos, last_map_recognition_at = self._resolve_misidentified_position(
                    wp,
                    raw_pos,
                    last_good_position,
                    raw_distance,
                    last_map_recognition_at=last_map_recognition_at,
                    now=now,
                )
                if pos is None:
                    lost_fixes += 1
                    if moving:
                        self.ctx.input.key_up("W")
                        moving = False
                    if lost_fixes >= 8:
                        self.log("[pathing] 连续无法定位，重置地图匹配缓存")
                        reset = getattr(self.positioner, "reset", None)
                        if callable(reset):
                            reset()
                        lost_fixes = 0
                    self.ctx.sleep(600)
                    continue
                lost_fixes = 0
                dx, dy = wp.x - pos[0], wp.y - pos[1]
                dist = math.hypot(dx, dy)
                if raw_pos is not None and (raw_distance is None or raw_distance <= 500):
                    last_good_position = raw_pos
                if dist <= arrive_dist:
                    self.log(f"[pathing] 到达路点 ({wp.x:.0f},{wp.y:.0f})")
                    reached = True
                    break

                # BetterGI samples once per second, compares an eight-point
                # window, and allows only two recoveries before retrying the
                # whole route.  The old mobile shortcut checked every frame
                # and could repeatedly jump/turn without ever leaving a trap.
                if (
                    moving
                    and wp.move_mode != "climb"
                    and now - last_stuck_sample_at >= 1.0
                ):
                    last_stuck_sample_at = now
                    if self._stuck_detector.add(pos):
                        in_trap = self._stuck_detector.trap_count
                        self.ctx.input.key_up("W")
                        moving = False
                        if in_trap > 2:
                            raise TimeoutError("此路线出现3次卡死，重试一次路线或放弃此路线")
                        self.log(f"[pathing] 疑似卡死，尝试脱离（第 {in_trap} 次）")
                        escaper = TrapEscaper(
                            self.ctx,
                            self.positioner,
                            log=self.log,
                            cam_sign=self._cam_sign,
                            cam_gain=self._cam_gain,
                        )
                        escaper.rotate_and_move()
                        escaper.move_to((wp.x, wp.y), wp.move_mode)
                        self.ctx.input.key_down("W")
                        moving = True
                        last_fix = None
                        prev_err = None
                        last_stuck_sample_at = now
                        self.log("[pathing] 卡死脱离结束")
                        continue

                desired = self._bearing(dx, dy)

                # 位移反馈：用实际移动向量估计当前航向并修正相机
                if last_fix is not None and moving:
                    mdx, mdy = pos[0] - last_fix[0], pos[1] - last_fix[1]
                    if math.hypot(mdx, mdy) >= 1.5:  # 位移足够大才可信
                        heading = self._bearing(mdx, mdy)
                        err = (desired - heading + 540) % 360 - 180
                        self.log(f"[pathing] 距离{dist:.0f} 期望{desired:.0f}° 实际{heading:.0f}° "
                                 f"误差{err:+.0f}° 符号{self._cam_sign:+.0f}")
                        if prev_err is not None and abs(err) > abs(prev_err) + 25:
                            self._cam_sign = -self._cam_sign  # 越修越偏 → 反号
                            self.log(f"[pathing] 相机增益反号 → {self._cam_sign:+.0f}")
                        prev_err = err
                        if abs(err) > 10:
                            # 摇杆按住时合成手势内的相机拖动无效（实测），
                            # 必须停步后独立滑动转相机，再继续前进
                            self.ctx.input.key_up("W")
                            moving = False
                            # 等手势泵当前原子手势(≤1.4s)走完，否则滑动会被
                            # 当成第二根手指而失效
                            self.ctx.sleep(1600)
                            turn = max(-90.0, min(90.0, err))  # 单次限幅防过冲
                            self.ctx.input.move_camera_by(
                                self._cam_sign * turn * self._cam_gain, 0)
                            self.ctx.sleep(500)
                        last_fix = (pos[0], pos[1], now)
                elif last_fix is None:
                    # 起步引导：用小地图扇形粗对准一次（失败就直接开走靠反馈纠偏）
                    cam = camera_orientation_deg(self.ctx, frame)
                    if cam is not None:
                        delta = (desired - cam + 540) % 360 - 180
                        if abs(delta) > 12:
                            self.ctx.input.move_camera_by(
                                self._cam_sign * max(-90.0, min(90.0, delta)) * self._cam_gain, 0)
                            self.ctx.sleep(500)
                    last_fix = (pos[0], pos[1], now)

                if not moving:
                    self.ctx.input.key_down("W")
                    moving = True
                if now - self._last_mode_action_at >= 1.0:
                    if wp.move_mode in ("run", "dash"):
                        self.ctx.input.key_press("LSHIFT")
                    elif wp.move_mode in ("fly", "jump"):
                        self.ctx.input.key_press("SPACE")
                    self._last_mode_action_at = now
                if last_fix is not None and now - last_fix[2] < 0.9:
                    self.ctx.sleep(400)  # 攒足位移再做下一次航向估计
        finally:
            if moving:
                self.ctx.input.key_up("W")
        if not reached:
            raise TimeoutError(f"[pathing] 路点 ({wp.x:.0f},{wp.y:.0f}) 执行超时")

    # ---- actions ----

    def _do_action(self, wp: Waypoint) -> None:
        self.actions.run(wp)
