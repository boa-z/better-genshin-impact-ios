"""Current BetterGI AutoLeyLineOutcrop workflow for the iOS touch runtime."""

from __future__ import annotations

import json
import math
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from ..engine.context import GameContext
from ..engine.genshin_api import GenshinApi
from ..engine.recognition import Mat, RecognitionObject
from ..pathing.executor import PathingExecutor
from ..pathing.model import PathingTask
from ..pathing.tp import TpTask
from .auto_fight import AutoFightTask


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "assets" / "data"
ROUTE_ROOT = PROJECT_ROOT / "assets" / "pathing" / "leyline"
TEMPLATE_ROOT = PROJECT_ROOT / "assets" / "templates" / "leyline"
GRAPH_PATH = DATA_ROOT / "leyline_graph.json"
CONFIG_PATH = DATA_ROOT / "leyline_config.json"

VALID_TYPES = frozenset(("启示之花", "藏金之花"))


@dataclass(frozen=True)
class LeyLineNode:
    id: int
    region: str
    x: float
    y: float
    kind: str


@dataclass(frozen=True)
class LeyLineEdge:
    source: int
    target: int
    route: str


@dataclass(frozen=True)
class LeyLineRoutePlan:
    start: LeyLineNode
    target: LeyLineNode
    routes: tuple[str, ...]


class LeyLineRouteGraph:
    def __init__(self, nodes: Iterable[LeyLineNode], edges: Iterable[LeyLineEdge]):
        self.nodes = {node.id: node for node in nodes}
        self.edges = tuple(edges)
        self.next: dict[int, list[LeyLineEdge]] = {}
        self.previous: dict[int, list[LeyLineEdge]] = {}
        for edge in self.edges:
            if edge.source not in self.nodes or edge.target not in self.nodes:
                continue
            self.next.setdefault(edge.source, []).append(edge)
            self.previous.setdefault(edge.target, []).append(edge)

    @classmethod
    def load(cls, path: str | Path = GRAPH_PATH) -> "LeyLineRouteGraph":
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        nodes = [
            LeyLineNode(
                int(item["id"]), str(item.get("region", "")),
                float(item["position"]["x"]), float(item["position"]["y"]), kind,
            )
            for kind, key in (("teleport", "teleports"), ("blossom", "blossoms"))
            for item in raw.get(key, [])
        ]
        edges = [
            LeyLineEdge(int(item["source"]), int(item["target"]), str(item["route"]))
            for item in raw.get("edges", [])
        ]
        return cls(nodes, edges)

    def nearest_blossom(
        self, x: float, y: float, *, country: str = "", threshold: float = 50,
    ) -> LeyLineNode | None:
        candidates = [
            node for node in self.nodes.values()
            if node.kind == "blossom"
            and (not country or node.region.startswith(country))
            and abs(node.x - x) <= threshold
            and abs(node.y - y) <= threshold
        ]
        return min(candidates, key=lambda node: math.hypot(node.x - x, node.y - y), default=None)

    def shortest_plan(self, target: LeyLineNode) -> LeyLineRoutePlan | None:
        """Multi-source BFS, equivalent to upstream's shortest teleport path search."""

        def forward_plan(goal: LeyLineNode) -> LeyLineRoutePlan | None:
            queue: deque[tuple[int, LeyLineNode, tuple[str, ...]]] = deque()
            visited: set[int] = set()
            for node in self.nodes.values():
                if node.kind == "teleport":
                    queue.append((node.id, node, ()))
                    visited.add(node.id)
            while queue:
                node_id, start, routes = queue.popleft()
                if node_id == goal.id:
                    return LeyLineRoutePlan(start, goal, routes)
                for edge in self.next.get(node_id, ()):
                    if edge.target in visited:
                        continue
                    visited.add(edge.target)
                    queue.append((edge.target, start, (*routes, edge.route)))
            return None

        direct = forward_plan(target)
        if direct is not None:
            return direct
        # Upstream permits a reverse-assisted final hop when the graph has no
        # forward plan. Search each incoming predecessor from teleport starts.
        for incoming in self.previous.get(target.id, ()):
            predecessor = self.nodes[incoming.source]
            partial = forward_plan(predecessor)
            if partial is not None:
                return LeyLineRoutePlan(
                    partial.start, target, (*partial.routes, incoming.route)
                )
        return None

    def resolve_route(self, route: str) -> Path:
        normalized = str(route).replace("\\", "/")
        for prefix in ("assets/pathing/", "Assets/pathing/"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        return ROUTE_ROOT / normalized

    @staticmethod
    def target_route(route: str) -> str:
        normalized = str(route).replace("\\", "/")
        if "assets/pathing/rerun/" in normalized:
            normalized = normalized.replace(
                "assets/pathing/rerun/", "assets/pathing/target/"
            )
        elif "assets/pathing/target/" not in normalized:
            normalized = normalized.replace(
                "assets/pathing/", "assets/pathing/target/"
            )
        return normalized.replace("-rerun", "")


@dataclass(frozen=True)
class LeyLineResinCounts:
    original: int = 0
    condensed: int = 0
    transient: int = 0
    fragile: int = 0


@dataclass(frozen=True)
class LeyLineRunCount:
    total: int
    original: int
    condensed: int
    transient: int
    fragile: int


def calculate_leyline_run_count(
    counts: LeyLineResinCounts, *, use_transient: bool, use_fragile: bool,
) -> LeyLineRunCount:
    original = max(0, counts.original) // 40
    if max(0, counts.original) % 40 >= 20:
        original += 1
    condensed = max(0, counts.condensed)
    transient = max(0, counts.transient) if use_transient else 0
    fragile = max(0, counts.fragile) if use_fragile else 0
    return LeyLineRunCount(
        original + condensed + transient + fragile,
        original, condensed, transient, fragile,
    )


def choose_leyline_reward_resins(
    lines: Iterable[str], *, use_transient: bool, use_fragile: bool,
) -> list[str]:
    normalized = [re.sub(r"\s+", "", str(line)) for line in lines]
    original_empty = any("补充" in line for line in normalized)
    double_reward = any(
        "双倍" in line or "2倍产出" in line or "2倍" in line
        for line in normalized
    )
    has_original = not original_empty and any("原粹" in line for line in normalized)
    has_condensed = any("浓缩" in line for line in normalized)
    has_transient = use_transient and any("须臾" in line for line in normalized)
    has_fragile = use_fragile and any("脆弱" in line for line in normalized)
    if double_reward and has_original:
        return ["原粹树脂"]
    if original_empty:
        return [
            name for name, present in (
                ("浓缩树脂", has_condensed),
                ("须臾树脂", has_transient),
                ("脆弱树脂", has_fragile),
            ) if present
        ]
    return [
        name for name, present in (
            ("浓缩树脂", has_condensed),
            ("须臾树脂", has_transient),
            ("原粹树脂", has_original),
            ("脆弱树脂", has_fragile),
        ) if present
    ]


class AutoLeyLineOutcropTask:
    def __init__(
        self,
        ctx: GameContext,
        *,
        count: int = 1,
        country: str = "蒙德",
        ley_line_type: str = "启示之花",
        route_path: str | Path | None = None,
        open_mode_count_min: bool = False,
        resin_exhaustion_mode: bool = False,
        use_adventurer_handbook: bool = False,
        friendship_team: str = "",
        team: str = "",
        use_fragile_resin: bool = False,
        use_transient_resin: bool = False,
        scan_drops_after_reward_enabled: bool = False,
        scan_drops_after_reward_seconds: int = 12,
        combat_strategy_path: str | None = None,
        timeout_s: float = 120,
        party_slots: dict[str, int] | None = None,
        one_dragon_mode: bool = False,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.count = max(1, int(count))
        self.country = str(country or "")
        self.ley_line_type = str(ley_line_type or "")
        self.route_path = Path(route_path).expanduser() if route_path else None
        self.open_mode_count_min = bool(open_mode_count_min)
        self.resin_exhaustion_mode = bool(resin_exhaustion_mode)
        self.use_adventurer_handbook = bool(use_adventurer_handbook)
        self.friendship_team = str(friendship_team or "")
        self.team = str(team or "")
        self.use_fragile_resin = bool(use_fragile_resin)
        self.use_transient_resin = bool(use_transient_resin)
        self.scan_drops_after_reward_enabled = bool(scan_drops_after_reward_enabled)
        self.scan_drops_after_reward_seconds = max(0, min(60, int(scan_drops_after_reward_seconds)))
        self.party_slots = party_slots or {}
        self.one_dragon_mode = bool(one_dragon_mode)
        self.log = log
        self.api = GenshinApi(ctx, log=log)
        self.tp = TpTask(ctx, log=log)
        self.graph = LeyLineRouteGraph.load()
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
        self.fight = AutoFightTask(
            ctx, combat_strategy_path=combat_strategy_path,
            timeout_s=max(30, float(timeout_s)), party_slots=self.party_slots, log=log,
        )
        self._templates: dict[str, Mat] = {}

    def _validate(self) -> None:
        if self.ley_line_type not in VALID_TYPES:
            raise ValueError("地脉花类型必须为启示之花或藏金之花")
        if self.country not in self.config.get("mapPositions", {}):
            raise ValueError(f"未找到国家 {self.country} 的地脉配置")
        if self.friendship_team and not self.team:
            raise ValueError("配置好感队时必须配置战斗队伍")
        if self.route_path is not None and not self.route_path.is_file():
            raise FileNotFoundError(f"地脉路线不存在：{self.route_path}")

    def _ro(self, name: str, roi=None, threshold: float = 0.8) -> RecognitionObject:
        if name not in self._templates:
            self._templates[name] = Mat.from_file(str(TEMPLATE_ROOT / name))
        ro = RecognitionObject.template_match(self._templates[name])
        ro.threshold = threshold
        ro.roi = roi
        return ro

    @staticmethod
    def _ocr_lines(region, roi=(0, 0, 1920, 1080)) -> tuple[list, list[str]]:
        hits = region.find_multi(RecognitionObject.ocr(*roi), limit=60)
        hits.sort(key=lambda hit: (hit.y, hit.x))
        lines: list[list] = []
        for hit in hits:
            target = next((line for line in lines if abs(line[0].y - hit.y) <= max(18, hit.height)), None)
            if target is None:
                lines.append([hit])
            else:
                target.append(hit)
        texts = ["".join(item.text.strip() for item in sorted(line, key=lambda h: h.x)) for line in lines]
        return hits, texts

    @staticmethod
    def _number(text: str) -> int:
        match = re.search(r"\d+", str(text).translate(str.maketrans("０１２３４５６７８９", "0123456789")))
        return int(match.group()) if match else 0

    def _count_resin(self) -> LeyLineResinCounts:
        try:
            return self._count_resin_from_ui()
        finally:
            self.api.returnMainUi()

    def _count_resin_from_ui(self) -> LeyLineResinCounts:
        if not self.tp.open_map():
            raise RuntimeError("无法打开大地图统计树脂")
        region = self.ctx.capture_region()
        original_icon = region.find(self._ro("original_resin.png"))
        condensed_icon = region.find(self._ro("condensed_resin.png"))
        original = condensed = transient = fragile = 0
        if original_icon.is_exist():
            text = "".join(self._ocr_lines(region, (
                original_icon.x, original_icon.y, 220, 50,
            ))[1])
            match = re.search(r"(\d{1,3})\s*/", text)
            original = int(match.group(1)) if match else self._number(text)
        if condensed_icon.is_exist():
            text = "".join(self._ocr_lines(region, (
                condensed_icon.x + condensed_icon.width, condensed_icon.y, 100, 50,
            ))[1])
            condensed = self._number(text)

        if self.use_transient_resin or self.use_fragile_resin:
            replenish = region.find(self._ro("replenish_resin_button.png"))
            if replenish.is_exist():
                replenish.click()
                self.ctx.sleep(800)
                panel = self.ctx.capture_region()
                for enabled, template, attr in (
                    (self.use_transient_resin, "transient_resin.png", "transient"),
                    (self.use_fragile_resin, "fragile_resin.png", "fragile"),
                ):
                    if not enabled:
                        continue
                    icon = panel.find(self._ro(template))
                    if not icon.is_exist():
                        continue
                    text = "".join(self._ocr_lines(panel, (
                        icon.x, icon.y + icon.height, max(80, icon.width), 70,
                    ))[1])
                    if attr == "transient":
                        transient = self._number(text)
                    else:
                        fragile = self._number(text)
        return LeyLineResinCounts(original, condensed, transient, fragile)

    def _run_limit(self) -> int:
        if not self.resin_exhaustion_mode:
            return self.count
        counts = calculate_leyline_run_count(
            self._count_resin(), use_transient=self.use_transient_resin,
            use_fragile=self.use_fragile_resin,
        )
        self.log(
            f"[AutoLeyLine] 树脂可执行 {counts.total} 次：原粹 {counts.original}，"
            f"浓缩 {counts.condensed}，须臾 {counts.transient}，脆弱 {counts.fragile}"
        )
        return min(self.count, counts.total) if self.open_mode_count_min else counts.total

    def _flower_template(self) -> RecognitionObject:
        name = "Blossom_of_Revelation.png" if self.ley_line_type == "启示之花" else "Blossom_of_Wealth.png"
        return self._ro(name, threshold=0.8)

    def _locate_flower(self) -> tuple[float, float]:
        try:
            return self._locate_flower_on_map()
        finally:
            self.api.returnMainUi()

    def _locate_flower_on_map(self) -> tuple[float, float]:
        # Both upstream modes ultimately identify the same icon on the big map.
        # Direct probes avoid a second screenshot consumer in the handbook UI.
        if not self.tp.open_map():
            raise RuntimeError("无法打开大地图寻找地脉花")
        self.tp.set_big_map_zoom_level(3.0)
        for probe in self.config["mapPositions"][self.country]:
            self.tp.move_map_to(float(probe["x"]), float(probe["y"]))
            region = self.ctx.capture_region()
            icons = region.find_multi(self._flower_template(), limit=12)
            if not icons:
                continue
            center_x = self.ctx.transform.device_width / 2
            center_y = self.ctx.transform.device_height / 2
            icon = min(
                icons,
                key=lambda hit: math.hypot(
                    hit.dx + hit.dw / 2 - center_x, hit.dy + hit.dh / 2 - center_y,
                ),
            )
            view = self.tp.big.locate_view(region.bgr)
            if view is None:
                continue
            map_x = view[0] + (icon.dx + icon.dw / 2 - center_x) / view[2]
            map_y = view[1] + (icon.dy + icon.dh / 2 - center_y) / view[2]
            world = self.tp.config.image_to_world(map_x * 8, map_y * 8)
            self.log(f"[AutoLeyLine] 定位 {self.ley_line_type} ({world[0]:.0f}, {world[1]:.0f})")
            return world
        mode = "冒险之证" if not self.use_adventurer_handbook else "大地图"
        raise RuntimeError(f"通过{mode}探测未找到 {self.country}{self.ley_line_type}")

    def _run_route(self, path: Path) -> bool:
        self.api.returnMainUi()
        return PathingExecutor(
            self.ctx, party_slots=self.party_slots, log=self.log,
        ).run(PathingTask.load(path))

    def _navigate_to_flower(self, position: tuple[float, float]) -> bool:
        if self.route_path is not None:
            return self._run_route(self.route_path)
        target = self.graph.nearest_blossom(
            *position, country=self.country,
            threshold=float(self.config.get("errorThreshold", 40)) + 10,
        )
        if target is None:
            raise RuntimeError(f"未找到地脉点位策略：({position[0]:.0f},{position[1]:.0f})")
        plan = self.graph.shortest_plan(target)
        if plan is None or not plan.routes:
            raise RuntimeError(f"未找到前往地脉点位 {target.region} 的路线")
        self.log(f"[AutoLeyLine] 匹配 {target.region}，执行 {len(plan.routes)} 段路线")
        for route in plan.routes:
            path = self.graph.resolve_route(route)
            if not path.is_file() or not self._run_route(path):
                return False
        correction = self.graph.resolve_route(self.graph.target_route(plan.routes[-1]))
        if correction.is_file() and not self._run_route(correction):
            return False
        return True

    def _reward_state(self, region) -> str | None:
        _hits, lines = self._ocr_lines(region, (420, 140, 1080, 760))
        text = "".join(lines).replace(" ", "")
        if "激活地脉之花" in text or "选择激活方式" in text:
            return "panel"
        if "接触" in text or "地脉之花" in text:
            return "interact"
        return None

    def _navigate_to_reward(self, cancelled=None, timeout_s: float = 60) -> bool:
        deadline = time.monotonic() + timeout_s
        moving = False
        try:
            while time.monotonic() < deadline:
                if cancelled and cancelled():
                    return False
                region = self.ctx.capture_region()
                state = self._reward_state(region)
                if state == "panel":
                    return True
                if state == "interact":
                    if moving:
                        self.ctx.input.key_up("W")
                        moving = False
                    self.ctx.input.key_press("F")
                    self.ctx.sleep(700)
                    continue
                icon = region.find(self._ro("box.png", (250, 80, 1420, 850)))
                if not icon.is_exist():
                    if moving:
                        self.ctx.input.key_up("W")
                        moving = False
                    self.ctx.input.move_camera_by(180, 0)
                    self.ctx.sleep(300)
                    continue
                x_offset = icon.x + icon.width / 2 - 960
                if abs(x_offset) > 100:
                    if moving:
                        self.ctx.input.key_up("W")
                        moving = False
                    self.ctx.input.move_camera_by(max(-300, min(300, x_offset)), 0)
                    self.ctx.sleep(300)
                    continue
                if not moving:
                    self.ctx.input.key_down("W")
                    moving = True
                self.ctx.sleep(500)
        finally:
            if moving:
                self.ctx.input.key_up("W")
        return False

    def _press_resin(self, hits: list, resin_name: str) -> bool:
        resin = next((hit for hit in hits if resin_name in hit.text.replace(" ", "")), None)
        if resin is None:
            return False
        uses = [hit for hit in hits if "使用" in hit.text and hit.x > 960]
        use = min(
            uses,
            key=lambda hit: abs((hit.y + hit.height / 2) - (resin.y + resin.height / 2)),
            default=None,
        )
        if use is None or abs((use.y + use.height / 2) - (resin.y + resin.height / 2)) > 45:
            return False
        use.click()
        self.ctx.sleep(60)
        use.click()
        return True

    def _claim_reward(self) -> bool:
        self.ctx.input.key_press("F")
        self.ctx.sleep(800)
        for _ in range(3):
            region = self.ctx.capture_region()
            hits, lines = self._ocr_lines(region, (420, 140, 1080, 760))
            candidates = choose_leyline_reward_resins(
                lines, use_transient=self.use_transient_resin,
                use_fragile=self.use_fragile_resin,
            )
            if candidates == ["原粹树脂"] and any("双倍" in line or "2倍" in line for line in lines):
                switch = region.find(self._ro("switch_button.png", threshold=0.7))
                if switch.is_exist() and not any("40" in line and "原粹" in line for line in lines):
                    switch.click()
                    self.ctx.sleep(500)
                    continue
            if any(self._press_resin(hits, candidate) for candidate in candidates):
                self.ctx.sleep(1200)
                self.api.returnMainUi()
                if self.team:
                    self.api.switchParty(self.team)
                return True
            self.ctx.sleep(350)
        self.log("[AutoLeyLine] 无可用树脂或领奖 OCR 失败")
        self.api.returnMainUi()
        if self.team:
            self.api.switchParty(self.team)
        return False

    def _scan_drops(self, cancelled=None) -> None:
        if not self.scan_drops_after_reward_enabled or self.scan_drops_after_reward_seconds <= 0:
            return
        loop = self.ctx.triggers
        previous = list(loop.triggers)
        self.ctx.enable_trigger("AutoPick")
        deadline = time.monotonic() + self.scan_drops_after_reward_seconds
        try:
            while time.monotonic() < deadline and not (cancelled and cancelled()):
                self.ctx.input.move_camera_by(180, 0)
                self.ctx.sleep(700)
        finally:
            loop.replace(previous)

    def run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        self._validate()
        limit = self._run_limit()
        if limit <= 0:
            self.log("[AutoLeyLine] 树脂耗尽，任务结束")
            return False
        self.api.returnMainUi()
        if not self.one_dragon_mode:
            self.api.tpToStatueOfTheSeven()
        if self.team and not self.api.switchParty(self.team):
            self.log(f"[AutoLeyLine] 未能切换战斗队 {self.team}，保持当前队伍")
        completed = 0
        try:
            while completed < limit:
                if cancelled and cancelled():
                    return False
                self.log(f"[AutoLeyLine] 第 {completed + 1}/{limit} 次 {self.country}{self.ley_line_type}")
                position = (0.0, 0.0) if self.route_path is not None else self._locate_flower()
                if not self._navigate_to_flower(position):
                    return False
                self.ctx.input.key_press("F")
                self.ctx.sleep(2500)
                if not self.fight.run(cancelled=cancelled):
                    return False
                if self.friendship_team and not self.api.switchParty(self.friendship_team):
                    self.log(f"[AutoLeyLine] 未能切换好感队 {self.friendship_team}")
                if not self._navigate_to_reward(cancelled):
                    self.log("[AutoLeyLine] 未能导航到地脉花奖励")
                    return False
                if not self._claim_reward():
                    return False
                completed += 1
                self._scan_drops(cancelled)
            self.log(f"[AutoLeyLine] 完成 {completed} 次")
            return True
        finally:
            self.ctx.input.release_all()
            try:
                self.api.returnMainUi()
            except Exception as error:
                self.log(f"[AutoLeyLine] 结束时返回主界面失败：{error}")
