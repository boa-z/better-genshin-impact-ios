"""地图追踪路点动作。

BetterGI 的 Windows 动作处理器依赖桌面窗口、键盘扫描和角色识别。这里
把能稳定映射到 DeviceHub 触控输入的动作集中起来，避免 PathingExecutor
里堆积大量与路线控制无关的分支。
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Callable

import numpy as np

from ..combat.dsl import CombatCommand, CombatExecutor
from ..engine.context import GameContext
from ..tasks.dispatcher import TaskDispatcher
from ..vision.ocr import get_ocr
from .model import Waypoint


_PICK_UP_ACTION_BODIES = {
    "枫原万叶-长E": "attack(0.08),keydown(E),wait(0.8),keyup(E),attack(0.5)",
    "枫原万叶-短E": "attack(0.08),keydown(E),wait(0.47),keyup(E),attack(0.5)",
    "琴-短E": (
        "wait(0.1),keydown(E),wait(0.4),moveby(1000,0),wait(0.2),"
        "moveby(1000,0),wait(0.2),moveby(1000,0),wait(0.2),"
        "moveby(1000,-3500),wait(1.8),keyup(E),wait(0.3),click(middle)"
    ),
    # The long-press Jean route is intentionally generated instead of copied
    # as a wall of repeated tokens.  It is the same 40 short camera advances
    # as BetterGI's PickUpCollectHandler, followed by the return sweep.
    "琴-长E": (
        "wait(0.1),click(middle),keydown(E),click(middle),wait(0.4),"
        + "".join("moveby(500,0),wait(0.1)," for _ in range(40))
        + "moveby(1000,3500),wait(1.8),keyup(E),wait(0.3),"
        "click(middle),wait(0.3)"
    ),
}
_PICK_UP_ALIASES = {
    "万叶": "枫原万叶",
    "kazuha": "枫原万叶",
    "jean": "琴",
}
_MINING_ACTIONS = (
    "莉奈娅 moveby(0,2000),wait(0.5),charge(0.6),wait(0.5),click(middle),wait(0.1)",
    "爱诺 attack(0.8)",
    "诺艾尔 attack(1.25)",
    "玛薇卡 attack(0.20),jump,wait(0.5),attack(0.6)",
    "迪希雅 attack(0.6),mousedown,wait(2.1),mouseup,jump",
    "娜维娅 attack(1.25)",
    "辛焱 attack(0.28),jump,wait(0.8),attack(0.28),jump,wait(0.8),attack(0.28),jump,wait(0.8)",
    "重云 attack(0.28),jump,wait(0.8),attack(0.28),jump,wait(0.8),attack(0.28),jump,wait(0.8)",
    "荒泷一斗 attack(0.1),charge(1.9),jump,wait(0.5),attack(0.2)",
    "基尼奇 attack(0.28),jump,wait(0.8),attack(0.28),jump,wait(0.8),attack(0.28),jump,wait(0.8),attack(0.28),jump,wait(0.8)",
    "菲米尼 attack(0.28),jump,wait(0.8),attack(0.28),jump,wait(0.8),attack(0.28),jump,wait(0.8),attack(0.28),jump,wait(0.8)",
    "卡维 attack(0.28),jump,wait(0.8),attack(0.28),jump,wait(0.8),attack(0.28),jump,wait(0.8),attack(0.28),jump,wait(0.8)",
    "优菈 attack(0.28),jump,wait(0.8),attack(0.28),jump,wait(0.8),attack(0.28),jump,wait(0.8),attack(0.28),jump,wait(0.8)",
    "嘉明 attack(0.28),jump,wait(0.8),attack(0.28),jump,wait(0.8),attack(0.28),jump,wait(0.8),attack(0.28),jump,wait(0.8)",
    "多莉 attack(2.0)",
    "北斗 attack(0.28),jump,wait(0.8),attack(0.28),jump,wait(0.8),attack(0.28),jump,wait(0.8)",
    "早柚 attack(0.23),jump,wait(0.6),attack(0.23),jump,wait(0.6),attack(0.23),jump,wait(0.6),attack(0.23),jump,wait(0.6)",
    "迪卢克 charge(3.15),jump",
    "坎蒂丝 e(hold,wait)",
    "雷泽 e(hold,wait)",
    "凝光 attack(4.0)",
    "钟离 e(hold,wait)",
)

# These are the same 1080p reference rectangles used by BetterGI's
# AutoFightAssets and Common/Element recognition assets.  The right/bottom
# anchors matter on the iPhone 13 Pro Max: its 19.5:9 frame has extra width
# which must not be treated as a centred 16:9 canvas.
_GADGET_COOLDOWN_RECT = (1790.0, 814.0, 60.0, 24.0)
_MOTION_KEY_TIP_RECT = (1570.0, 1010.0, 200.0, 70.0)
_MOTION_TEMPLATE_DIR = (
    Path(__file__).resolve().parents[2] / "assets" / "templates" / "pathing"
)
_MOTION_TEMPLATE_NAMES = {"space": "key_space.png", "x": "key_x.png"}
_MOTION_TEMPLATE_CACHE: dict[str, tuple[np.ndarray, np.ndarray | None] | None] = {}


def _safe_scale(transform) -> float:
    try:
        value = float(transform.scale)
    except (AttributeError, TypeError, ValueError):
        return 1.0
    return value if math.isfinite(value) and value > 0 else 1.0


def _crop_reference_rect(
    bgr: np.ndarray, transform, rect: tuple[float, float, float, float],
    *, anchor: str = "right",
) -> np.ndarray | None:
    """Crop a 1080p reference rectangle using the mobile edge anchor rules."""
    if not isinstance(bgr, np.ndarray) or bgr.ndim < 2 or bgr.size == 0:
        return None
    x, y, width, height = (float(value) for value in rect)
    scale = _safe_scale(transform)
    try:
        dx, dy = transform.to_device(x, y, anchor=anchor)
        dx, dy = float(dx), float(dy)
    except (AttributeError, TypeError, ValueError):
        dx, dy = x * scale, y * scale
    left = max(0, int(round(dx)))
    top = max(0, int(round(dy)))
    right = min(bgr.shape[1], int(round(dx + width * scale)))
    bottom = min(bgr.shape[0], int(round(dy + height * scale)))
    if right <= left or bottom <= top:
        return None
    return bgr[top:bottom, left:right]


def _parse_ocr_number(value) -> float:
    """Parse the first cooldown number from OCR text without raising."""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = "".join(str(getattr(item, "text", item)) for item in value)
        except TypeError:
            text = str(value or "")
    # Paddle occasionally returns a comma as the decimal separator and can
    # confuse the digit 0 with O.  Only normalize characters in numeric runs.
    text = text.replace(",", ".").replace("O", "0").replace("o", "0")
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)", text)
    if match is None:
        return 0.0
    try:
        result = float(match.group(1))
    except ValueError:
        return 0.0
    return result if math.isfinite(result) and result >= 0 else 0.0


def read_gadget_cooldown(frame: np.ndarray, transform) -> float:
    """Read the quick-use gadget cooldown from one already captured frame.

    BetterGI thresholds the gadget cooldown crop to near-white before OCR.
    Returning ``0`` for an unavailable/undecidable crop preserves the
    original handler's fail-open behaviour: the action is attempted instead
    of blocking a route indefinitely.
    """
    import cv2

    crop = _crop_reference_rect(frame, transform, _GADGET_COOLDOWN_RECT)
    if crop is None:
        return 0.0
    if crop.ndim == 2:
        gray = crop
    else:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray = cv2.inRange(
            hsv,
            np.array((0, 0, 235), dtype=np.uint8),
            np.array((0, 25, 255), dtype=np.uint8),
        )
    try:
        return _parse_ocr_number(get_ocr().recognize(gray))
    except Exception:
        # OCR is optional on the portable runtime and can also be unavailable
        # while the game is transitioning between frames.
        return 0.0


def _load_motion_template(kind: str) -> tuple[np.ndarray, np.ndarray | None] | None:
    cached = _MOTION_TEMPLATE_CACHE.get(kind, ...)
    if cached is not ...:
        return cached
    import cv2

    name = _MOTION_TEMPLATE_NAMES.get(kind)
    if not name:
        _MOTION_TEMPLATE_CACHE[kind] = None
        return None
    image = cv2.imread(str(_MOTION_TEMPLATE_DIR / name), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        _MOTION_TEMPLATE_CACHE[kind] = None
        return None
    mask = None
    if image.ndim == 3 and image.shape[2] == 4:
        mask = image[:, :, 3]
        image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _MOTION_TEMPLATE_CACHE[kind] = (image, mask)
    return image, mask


def _template_visible(
    crop: np.ndarray, transform, template_data: tuple[np.ndarray, np.ndarray | None],
) -> bool:
    import cv2

    template, mask = template_data
    scale = _safe_scale(transform)
    target_size = (
        max(1, int(round(template.shape[1] * scale))),
        max(1, int(round(template.shape[0] * scale))),
    )
    if target_size != (template.shape[1], template.shape[0]):
        template = cv2.resize(template, target_size, interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] < template.shape[0] or gray.shape[1] < template.shape[1]:
        return False
    if mask is not None and cv2.countNonZero(mask) > 0:
        result = cv2.matchTemplate(
            gray, template, cv2.TM_SQDIFF_NORMED, mask=mask,
        )
        return float(cv2.minMaxLoc(result)[0]) <= 0.35
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    return float(cv2.minMaxLoc(result)[1]) >= 0.62


def detect_motion_status(frame: np.ndarray, transform) -> str | None:
    """Return ``normal``, ``fly`` or ``climb`` from BetterGI key prompts.

    ``None`` means the detector could not run (for example, a stripped asset
    bundle).  A loaded detector with no matching prompt returns ``normal``;
    this distinction lets the action fail safely without confusing a missing
    detector with a landed character.
    """
    import cv2

    crop = _crop_reference_rect(frame, transform, _MOTION_KEY_TIP_RECT)
    if crop is None:
        return None
    found: dict[str, bool] = {}
    for kind in _MOTION_TEMPLATE_NAMES:
        template = _load_motion_template(kind)
        if template is None:
            continue
        try:
            found[kind] = _template_visible(crop, transform, template)
        except (cv2.error, TypeError, ValueError):
            found[kind] = False
    if not found:
        return None
    if found.get("space"):
        return "climb" if found.get("x") else "fly"
    return "normal"


class CircularMotionCalculator:
    """Calculate the same expanding pickup arc used by BetterGI.

    ``pick_around`` is deliberately a movement pattern rather than a series
    of blind ``F`` presses.  The original handler turns the camera in small
    increments while walking a growing arc; AutoPick can then recognize the
    interaction prompt in each direction.  Keeping the calculator separate
    makes the timing contract deterministic and easy to test without a
    device.
    """

    START_MS = 600
    INTERVAL_MS = 400
    CIRCLE_MS = 33000.0
    RADIUS_MS = CIRCLE_MS / (2 * math.pi)

    def __init__(self, speed: float = 1.1):
        self._speed = 1.0
        self.speed = speed

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("拾取轨迹速度必须为正数")
        self._speed = value
        self._view_reset_ms = 350 * value
        self._mix_angle = (
            self._view_reset_ms / self.CIRCLE_MS + 1.0 / 4
        ) * 2 * math.pi
        self._mix_x, self._mix_y = self._arc_point(
            self._view_reset_ms / self._mix_angle, self._mix_angle
        )

    @staticmethod
    def _arc_point(radius: float, angle: float) -> tuple[float, float]:
        return radius * (1 - math.cos(angle)), radius * math.sin(angle)

    def get_circle_info(self, index: int) -> tuple[float, float, float]:
        edge_ms = self.START_MS + int(index) * self.INTERVAL_MS
        angle = (edge_ms / self.CIRCLE_MS + 1.0 / 4) * math.pi
        rest_x, rest_y = self._arc_point(
            self.RADIUS_MS, 2 * angle - self._mix_angle
        )
        x = self._mix_x - rest_x
        y = self._mix_y + rest_y
        small_radius_ms = math.hypot(x, y) / (2 * math.sin(angle))
        end_angle = (
            angle - self._mix_angle + math.atan2(x, y) + math.pi / 2
        )
        return edge_ms / self.speed, small_radius_ms / self.speed, end_angle

    # Keep the upstream C#-style spelling available to converted helpers.
    GetCircleInfo = get_circle_info


class PathingActionRunner:
    """Execute the portable subset of BetterGI's waypoint action handlers."""

    SUPPORTED = frozenset({
        "combat_script", "fight", "normal_attack", "elemental_skill", "stop_flying", "mining",
        "linnea_mining", "nahida_collect", "hydro_collect", "electro_collect",
        "anemo_collect", "pyro_collect", "pick_around", "pick_up_collect",
        "fishing", "use_gadget", "up_down_grab_leaf", "log_output", "force_tp",
        "exit_and_relogin", "set_time", "wonderland_cycle",
    })

    def __init__(
        self,
        ctx: GameContext,
        *,
        party_slots: dict[str, int] | None = None,
        log: Callable[[str], None] = print,
        motion_detector: Callable[[np.ndarray], str | None] | None = None,
    ):
        self.ctx = ctx
        self.log = log
        self.motion_detector = motion_detector
        configured_slots = party_slots
        if configured_slots is None:
            configured_slots = getattr(ctx, "party_slots", None)
        self.party_slots = {
            str(name): int(slot)
            for name, slot in (configured_slots or {}).items()
            if str(name).strip()
            and isinstance(slot, (int, float))
            and 1 <= int(slot) <= 4
        }
        self.combat = CombatExecutor.for_context(
            ctx, party_slots=self.party_slots, log=log
        )

    @staticmethod
    def _number(value: str, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def run(self, waypoint: Waypoint) -> bool:
        action = waypoint.action.strip().lower()
        if not action:
            return True
        handler = getattr(self, f"_action_{action}", None)
        if handler is None:
            self.log(f"[pathing] 动作 {action} 暂不支持，已停止在该路点前")
            return False
        handler(waypoint)
        return True

    def _action_combat_script(self, waypoint: Waypoint) -> None:
        if not waypoint.action_params.strip():
            self.log("[pathing] combat_script 缺少 action_params")
            return
        self.combat.run(waypoint.action_params)

    def _action_fight(self, waypoint: Waypoint) -> None:
        self.log("[pathing] 执行通用战斗")
        raw = waypoint.action_params.strip()
        config: dict = {}
        if raw.startswith("{"):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError("fight action_params 不是有效 JSON") from error
            if isinstance(value, dict):
                config = value
        elif raw:
            # Route authors commonly put a strategy filename directly in
            # action_params; keep accepting a full task mapping as well.
            config["combatStrategyPath"] = raw
        result = TaskDispatcher(
            self.ctx, party_slots=self.party_slots, log=self.log
        ).run_auto_fight_task(config)
        if result is False:
            raise RuntimeError("fight 动作失败")

    def _action_normal_attack(self, waypoint: Waypoint) -> None:
        seconds = max(0.2, self._number(waypoint.action_params, 1.2))
        self.combat.run(f"attack({seconds})")

    def _action_elemental_skill(self, waypoint: Waypoint) -> None:
        self.ctx.input.key_press("E")
        self.ctx.sleep(1000)

    def _action_stop_flying(self, waypoint: Waypoint) -> None:
        # BetterGI's optional parameter is a free-fall interval in milliseconds
        # (not seconds).  Keep the two jump presses even for ``0``: routes use
        # that form to explicitly close the glider before the plunge attack.
        raw_wait = waypoint.action_params.strip()
        try:
            wait_ms = int(raw_wait) if raw_wait else None
        except ValueError:
            wait_ms = None
        if wait_ms is not None:
            self.ctx.input.key_press("SPACE")
            self.ctx.sleep(max(0, wait_ms))
            self.ctx.input.key_press("SPACE")
            self.ctx.sleep(300)

        self.log("[pathing] 执行动作：下落攻击")
        self.ctx.input.attack()
        capture = getattr(self.ctx, "capture_bgr", None)
        if not callable(capture):
            # Minimal script hosts may not expose screenshots.  The attack has
            # still been sent, which is the same useful fallback as before.
            self.log("[pathing] 当前运行环境没有飞行状态截图，结束下落攻击")
            return

        for _ in range(50):
            try:
                frame = capture()
            except Exception as error:
                self.log(f"[pathing] 飞行状态截图失败，结束下落攻击：{error}")
                return
            if not isinstance(frame, np.ndarray):
                self.log("[pathing] 飞行状态帧无效，结束下落攻击")
                return
            status = self._motion_status(frame)
            if status == "fly":
                self.ctx.sleep(300)
                continue
            if status is None:
                self.log("[pathing] 无法识别飞行提示，结束下落攻击")
            else:
                self.log("[pathing] 下落攻击结束")
            return
        self.log("[pathing] 下落攻击超时结束")

    def _motion_status(self, frame: np.ndarray) -> str | None:
        detector = self.motion_detector
        if detector is None:
            return detect_motion_status(frame, getattr(self.ctx, "transform", None))
        try:
            value = detector(frame)
        except Exception as error:
            self.log(f"[pathing] 自定义飞行状态识别失败：{error}")
            return None
        name = getattr(value, "name", value)
        normalized = str(name or "").strip().casefold()
        return {
            "normal": "normal", "0": "normal",
            "fly": "fly", "flying": "fly", "1": "fly",
            "climb": "climb", "climbing": "climb", "2": "climb",
        }.get(normalized)

    def _action_mining(self, waypoint: Waypoint) -> None:
        # BetterGI chooses a character-specific mining macro from the current
        # party.  Reuse the same combat DSL so the mobile profile receives
        # proper charged attacks, jumps and holds instead of six blind clicks.
        for script in _MINING_ACTIONS:
            character, body = script.split(" ", 1)
            selected = self._party_character({character})
            if selected is None:
                continue
            self.log(f"[pathing] 使用 {selected} 挖矿动作")
            self.combat.run(f"{selected} {body}")
            return
        self.log("[pathing] 队伍中没有专用挖矿角色，使用通用攻击")
        self.combat.run("attack(1.6)")

    def _action_linnea_mining(self, waypoint: Waypoint) -> None:
        from .linnea_mining import LinneaMiningTask, parse_linnea_mining_params

        selected = self._party_character({"莉奈娅", "Linnea"})
        if selected is None:
            self.log("[pathing] 队伍中未找到莉奈娅，跳过 Yolo 挖矿")
            return
        mine_count, scan_rounds = parse_linnea_mining_params(waypoint.action_params)
        self.log(
            f"[pathing] 莉奈娅 Yolo 挖矿：挖矿 {mine_count} 次，扫描 {scan_rounds} 轮"
        )
        self.combat.switch_to(selected)
        if not LinneaMiningTask(
            self.ctx,
            scan_rounds=scan_rounds,
            mine_count=mine_count,
            log=self.log,
        ).run():
            raise RuntimeError("linnea_mining 执行失败")

    def _action_nahida_collect(self, waypoint: Waypoint) -> None:
        self.log("[pathing] 纳西妲长按 E 旋转采集")
        self.combat.switch_to("纳西妲")
        self.combat.exec(CombatCommand("ready"))
        self.ctx.input.move_camera_by(0, 10000)
        self.ctx.sleep(200)
        self.ctx.input.key_down("E")
        try:
            for _ in range(15):
                self.ctx.input.move_camera_by(400, 500)
                self.ctx.sleep(30)
            for index in range(60, 0, -1):
                vertical = -50 if index <= 40 else -30
                self.ctx.input.move_camera_by(400, vertical)
                self.ctx.sleep(30)
        finally:
            self.ctx.input.key_up("E")
        self.ctx.sleep(800)
        self._middle_click()
        self.ctx.sleep(1000)

    def _action_hydro_collect(self, waypoint: Waypoint) -> None:
        self._action_elemental_collect(waypoint, "hydro")

    def _action_electro_collect(self, waypoint: Waypoint) -> None:
        self._action_elemental_collect(waypoint, "electro")

    def _action_anemo_collect(self, waypoint: Waypoint) -> None:
        self._action_elemental_collect(waypoint, "anemo")

    def _action_pyro_collect(self, waypoint: Waypoint) -> None:
        self._action_elemental_collect(waypoint, "pyro")

    def _action_pick_around(self, waypoint: Waypoint) -> None:
        try:
            turns = int(float(waypoint.action_params.strip() or "1"))
        except (TypeError, ValueError):
            turns = 1
        turns = max(1, turns)
        calculator = CircularMotionCalculator(1.1)
        old_radius = 0.0
        angle = 0.0
        for index in range(turns):
            edge_ms, radius_ms, end_angle = calculator.get_circle_info(index)
            self._move_to_next_pickup_start(old_radius, radius_ms, angle)
            self._move_pickup_circle(edge_ms, 6)
            old_radius = radius_ms
            angle = end_angle

    def _middle_click(self) -> None:
        """Use the profile's middle-button semantic (elemental sight)."""
        tap_button = getattr(self.ctx.input, "tap_button", None)
        if callable(tap_button):
            tap_button("elementalSight")
            return
        # Small test doubles and older input adapters may expose the original
        # method name directly.
        self.ctx.input.middle_button_click()

    def _move_pickup_circle(self, edge_ms: float, count: int) -> None:
        self.ctx.input.key_down("A")
        try:
            self.ctx.sleep(30)
            for _ in range(max(0, int(count))):
                self._middle_click()
                self.ctx.sleep(int(round(edge_ms)))
        finally:
            self.ctx.input.key_up("A")
            self.ctx.sleep(200)

    def _move_after_pickup_turn(self, direction: str, milliseconds: int = 0) -> None:
        self.ctx.input.key_press(direction)
        self.ctx.sleep(200)
        self._middle_click()
        self.ctx.sleep(500)
        if milliseconds <= 0:
            return
        self.ctx.input.key_down("W")
        try:
            self.ctx.sleep(int(milliseconds))
        finally:
            self.ctx.input.key_up("W")
        self.ctx.sleep(200)

    def _move_to_next_pickup_start(
        self, old_radius: float, new_radius: float, angle: float
    ) -> None:
        x = new_radius - old_radius * math.cos(angle)
        y = old_radius * math.sin(angle)
        self._middle_click()
        self.ctx.sleep(500)
        self._move_after_pickup_turn("S", int(round(y)) + 200)
        self._move_after_pickup_turn("A", int(round(x)))

    def _party_character(self, candidates: set[str]) -> str | None:
        """Return the first configured party member matching a canonical name."""
        try:
            from ..engine.party_hud import canonical_avatar_name
        except Exception:  # pragma: no cover - only for minimal host doubles
            canonical_avatar_name = lambda value: None
        for name, _slot in sorted(
            self.party_slots.items(), key=lambda item: int(item[1])
        ):
            canonical = canonical_avatar_name(name) or str(name).strip()
            if canonical.casefold() in {value.casefold() for value in candidates}:
                return name
        return None

    def _action_elemental_collect(self, waypoint: Waypoint, element: str) -> None:
        # These lists mirror ElementalCollectAvatarConfigs in the upstream
        # handler.  ``attack`` is preferred where the original character can
        # apply an element with a normal attack; otherwise the skill is used.
        normal_attack = {
            "hydro": {"芭芭拉", "莫娜", "珊瑚宫心海", "玛拉妮", "那维莱特", "芙宁娜"},
            "electro": {"丽莎", "八重神子", "瓦雷莎"},
            "anemo": {"砂糖", "鹿野院平藏", "流浪者", "闲云", "蓝砚"},
            "pyro": {"烟绯", "可莉"},
        }
        elemental = {
            "hydro": {"妮露", "坎蒂丝", "行秋", "神里绫人"},
            "electro": {"雷电将军", "久岐忍", "北斗", "菲谢尔", "雷泽"},
            "anemo": {"枫原万叶", "珐露珊", "琳妮特", "温迪", "琴", "早柚"},
            "pyro": {
                "迪卢克", "班尼特", "香菱", "托马", "胡桃", "迪希雅",
                "夏沃蕾", "辛焱", "林尼", "宵宫",
            },
        }
        normal_names = normal_attack.get(element, set())
        skill_names = elemental.get(element, set())
        selected = self._party_character(normal_names | skill_names)
        if selected is None:
            self.log(f"[pathing] 队伍中没有可用的{element}元素采集角色")
            return
        try:
            from ..engine.party_hud import canonical_avatar_name
            canonical = canonical_avatar_name(selected) or selected
        except Exception:
            canonical = selected
        self.log(f"[pathing] {element} 元素采集：使用 {selected}")
        self.combat.switch_to(selected)
        if canonical in normal_names:
            # The upstream Avatar.Attack(100) is a short, single normal hit.
            self.combat.exec(CombatCommand("attack", ["0.1"]))
        elif canonical in skill_names:
            self.combat.exec(CombatCommand("ready"))
            self.combat.exec(CombatCommand("e"))

    def _action_pick_up_collect(self, waypoint: Waypoint) -> None:
        raw = waypoint.action_params.strip()
        requests = [part.strip() for part in raw.split(",") if part.strip()]
        if not requests:
            requests = [
                base for base in ("枫原万叶", "琴")
                if self._party_character({base}) is not None
            ]
        if not requests:
            self.log("[pathing] 队伍中未找到万叶/琴，跳过聚集材料")
            return

        for request in requests:
            action_key, base_name = self._resolve_pickup_action(request)
            if action_key is None or base_name is None:
                self.log(f"[pathing] 未找到聚集动作：{request}")
                continue
            selected = self._party_character({base_name})
            if selected is None:
                # Explicit names are still passed through to the combat host;
                # its OCR TeamSwitcher can resolve a party not yet cached by
                # the caller.
                selected = base_name
            body = _PICK_UP_ACTION_BODIES[action_key]
            self.combat.run(f"{selected} {body}")

    @staticmethod
    def _pickup_name(value: str) -> str:
        base = str(value).strip().split("-", 1)[0].casefold()
        return _PICK_UP_ALIASES.get(base, base)

    def _resolve_pickup_action(self, request: str) -> tuple[str | None, str | None]:
        value = str(request).strip()
        requested_name = self._pickup_name(value)
        requested_key = value.casefold()
        for key in _PICK_UP_ACTION_BODIES:
            if key.casefold() == requested_key:
                return key, requested_name
        for key in _PICK_UP_ACTION_BODIES:
            if self._pickup_name(key) == requested_name:
                return key, requested_name
        return None, None

    def _action_fishing(self, waypoint: Waypoint) -> None:
        config: dict = {}
        raw = waypoint.action_params.strip()
        if raw.startswith("{"):
            try:
                config = json.loads(raw)
            except json.JSONDecodeError:
                self.log("[pathing] fishing action_params 不是有效 JSON，使用默认参数")
        elif raw:
            config["targetCatches"] = max(1, int(self._number(raw, 1)))
        TaskDispatcher(self.ctx, log=self.log).run_auto_fishing_task(config)

    def _action_use_gadget(self, waypoint: Waypoint) -> None:
        self.log("[pathing] 执行：使用小道具")
        self.ctx.input.key_press("Z")
        raw = waypoint.action_params.strip()
        if "not_wait" in raw.lower():
            # Keep BetterGI's historical not_wait contract: invoke again
            # immediately and only retain the common action settle delay.
            self.ctx.input.key_press("Z")
            self.ctx.sleep(300)
            return

        try:
            max_wait_seconds = float(raw) if raw else 100.0
        except ValueError:
            max_wait_seconds = 100.0
        if not math.isfinite(max_wait_seconds):
            max_wait_seconds = 100.0
        max_wait_seconds = max(0.0, max_wait_seconds)

        cooldown = 0.0
        capture = getattr(self.ctx, "capture_bgr", None)
        if callable(capture):
            try:
                frame = capture()
                if isinstance(frame, np.ndarray):
                    cooldown = read_gadget_cooldown(
                        frame, getattr(self.ctx, "transform", None)
                    )
            except Exception as error:
                self.log(f"[pathing] 小道具冷却识别失败，立即重试：{error}")

        if cooldown > 100:
            self.log(
                f"[pathing] 小道具冷却识别异常（{cooldown:.1f}秒），立即重试"
            )
        elif cooldown > 0:
            wait_seconds = min(cooldown, max_wait_seconds)
            if cooldown > max_wait_seconds:
                self.log(
                    f"[pathing] 小道具冷却 {cooldown:.1f}秒，使用最大等待"
                    f" {max_wait_seconds:.1f}秒"
                )
            else:
                self.log(f"[pathing] 等待小道具冷却 {cooldown:.1f}秒")
            self.ctx.sleep(int(wait_seconds * 1000) + 100)

        self.ctx.input.key_press("Z")
        self.ctx.sleep(300)

    def _action_up_down_grab_leaf(self, waypoint: Waypoint) -> None:
        direction = -1.0 if waypoint.action_params.strip().lower() == "down" else 1.0
        vertical_movement = direction * 1000
        consecutive_detections = 0
        for cycle in range(40):
            if cycle and cycle % 10 == 0:
                vertical_movement = -vertical_movement
            frame = self.ctx.capture_bgr()
            if self._leaf_prompt_visible(frame):
                consecutive_detections += 1
                if consecutive_detections >= 2:
                    self.log("[pathing] 连续检测到四叶印，执行交互")
                    self.ctx.input.key_press("F")
                    self.ctx.sleep(200)
                    self._middle_click()
                    self.ctx.input.key_press("SPACE")
                    return
                self.ctx.sleep(150)
                continue
            consecutive_detections = 0
            self.ctx.input.move_camera_by(0, vertical_movement)
            self.ctx.sleep(100)
        self._middle_click()
        self.ctx.sleep(300)
        self.log("[pathing] 未检测到四叶印，已恢复视角")

    def _leaf_prompt_visible(self, bgr: np.ndarray) -> bool:
        points = ((1500, 1000), (1508, 1041), (1500, 987), (1500, 1010))
        shifted = (0, 120, -104)
        for offset in shifted:
            hits = 0
            for ref_x, ref_y in points:
                x, y = self.ctx.transform.to_device(ref_x + offset, ref_y)
                ix, iy = int(round(x)), int(round(y))
                if 0 <= iy < bgr.shape[0] and 0 <= ix < bgr.shape[1]:
                    pixel = bgr[iy, ix]
                    if bool(np.all(pixel >= 245)):
                        hits += 1
            if hits >= 3:
                return True
        return False

    def _action_log_output(self, waypoint: Waypoint) -> None:
        self.log(f"[pathing] {waypoint.action_params}")

    def _action_force_tp(self, waypoint: Waypoint) -> None:
        # PathingExecutor handles this action before movement so it can reuse
        # the waypoint coordinate as the teleport target.
        self.log("[pathing] force_tp 已由传送流程处理")

    def _action_exit_and_relogin(self, waypoint: Waypoint) -> None:
        from ..engine.genshin_api import GenshinApi

        self.log("[pathing] 退出并重新登录原神")
        GenshinApi(self.ctx, log=self.log).relogin()

    def _action_set_time(self, waypoint: Waypoint) -> None:
        from ..engine.genshin_api import GenshinApi

        raw = waypoint.action_params.strip()
        matched = re.fullmatch(
            r"(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<skip>true|false))?",
            raw,
            flags=re.IGNORECASE,
        )
        if matched is None:
            raise ValueError(
                "set_time action_params 必须为 H:mm、HH:mm 或 HH:mm:true/false"
            )
        hour = int(matched.group("hour"))
        minute = int(matched.group("minute"))
        if not 0 <= hour <= 24 or not 0 <= minute <= 59:
            raise ValueError("set_time 时间必须为 hour 0-24、minute 0-59")
        skip_value = matched.group("skip")
        # BetterGI's SetTimeHandler defaults to skipping the clock animation.
        skip_animation = skip_value is None or skip_value.lower() == "true"
        self.log(
            f"[pathing] 设置游戏时间 {hour:02d}:{minute:02d}"
            f"（跳过动画：{'是' if skip_animation else '否'}）"
        )
        if not GenshinApi(self.ctx, log=self.log).setTime(
            hour, minute, skip_animation
        ):
            raise RuntimeError("set_time 失败：未能打开或确认游戏时间界面")

    def _action_wonderland_cycle(self, waypoint: Waypoint) -> None:
        from ..engine.genshin_api import GenshinApi

        self.log("[pathing] 进入并退出千星奇域")
        if not GenshinApi(self.ctx, log=self.log).wonderlandCycle():
            raise RuntimeError("wonderland_cycle 失败：未能完成千星奇域进入退出流程")
