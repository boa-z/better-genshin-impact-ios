"""BetterGI ``CombatScenes`` and ``Avatar`` JavaScript host compatibility.

The desktop implementation discovers the team with YOLO/OCR and then routes
combat actions through Windows input. On iOS, configured ``party_slots`` are
the authoritative team model and every action is sent through InputSimulator,
so DeviceHub profiles keep held movement and buttons on the device side.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Mapping

from ..combat.hud import is_skill_ready


def _unwrap(value: Any) -> Any:
    return getattr(value, "__wrapped__", value)


def _cancelled(token: Any) -> bool:
    token = _unwrap(token)
    if token is None:
        return False
    for name in ("isCancellationRequested", "IsCancellationRequested", "cancelled"):
        try:
            value = getattr(token, name)
            return bool(value() if callable(value) else value)
        except (AttributeError, TypeError):
            continue
    return False


class Avatar:
    def __init__(self, scenes: "CombatScenes", name: str, index: int,
                 name_rect: Any = None, manual_skill_cd: float = -1):
        self.combat_scenes = scenes
        self.ctx = scenes.ctx
        self.name = str(name)
        self.index = int(index)
        self.combat_avatar = {
            "Name": self.name, "SkillCd": 0, "SkillHoldCd": 0,
        }
        self.manual_skill_cd = float(manual_skill_cd)
        self.last_skill_time = 0.0
        self.is_burst_ready = True
        self.name_rect = name_rect
        self.index_rect = None
        self.ct = None

    def _check_cancel(self) -> bool:
        return _cancelled(self.ct)

    def switch(self) -> None:
        if not self._check_cancel():
            self.ctx.input.switch_party_slot(self.index)
            self.combat_scenes.last_active_avatar_index = self.index
            self.ctx.sleep(600)

    def try_switch(self, try_times: int = 4, need_log: bool = True) -> bool:
        for _ in range(max(1, int(try_times))):
            if self._check_cancel():
                return False
            self.switch()
            if self.is_active(None):
                return True
        if need_log:
            self.combat_scenes.log(f"[combat] 切换到 {self.name} 失败")
        return False

    def switch_without_cts(self) -> None:
        token, self.ct = self.ct, None
        try:
            self.switch()
        finally:
            self.ct = token

    def use_skill(self, hold_press: bool = False) -> None:
        if self._check_cancel():
            return
        self.ctx.input.key_press("E", hold_ms=900 if bool(hold_press) else 80)
        self.ctx.sleep(200)
        self.after_use_skill()

    def after_use_skill(self, _given_region: Any = None) -> float:
        self.last_skill_time = time.monotonic()
        return self.get_skill_cd_seconds()

    def use_burst(self) -> None:
        if self._check_cancel() or not self.is_burst_ready:
            return
        self.ctx.input.key_press("Q")
        self.ctx.sleep(2000)
        self.is_burst_ready = False

    def attack(self, milliseconds: int = 0) -> None:
        remain = max(0, int(milliseconds))
        while True:
            if self._check_cancel():
                return
            self.ctx.input.attack()
            self.ctx.sleep(200)
            if remain <= 0:
                return
            remain -= 200

    def charge(self, milliseconds: int = 0) -> None:
        if not self._check_cancel():
            self.ctx.input.charged_attack(max(1, int(milliseconds) or 1000))

    def dash(self, milliseconds: int = 0) -> None:
        if not self._check_cancel():
            self.ctx.input.key_press("LSHIFT", hold_ms=max(1, int(milliseconds) or 200))

    def jump(self) -> None:
        if not self._check_cancel():
            self.ctx.input.key_press("SPACE")

    def walk(self, direction: str, milliseconds: int) -> None:
        key = str(direction).upper()
        if key not in {"W", "A", "S", "D"} or self._check_cancel():
            return
        self.ctx.input.key_down(key)
        try:
            self.ctx.sleep(max(0, int(milliseconds)))
        finally:
            self.ctx.input.key_up(key)

    def move_camera(self, x: float, y: float) -> None:
        self.ctx.input.move_camera_by(float(x), float(y))

    def wait(self, milliseconds: int) -> None:
        self.ctx.sleep(max(0, int(milliseconds)))

    def ready(self) -> None:
        for _ in range(20):
            if self._check_cancel() or is_skill_ready(self.ctx):
                return
            self.ctx.sleep(150)

    def is_skill_ready(self, _print_log: bool = False) -> bool:
        return self.get_skill_cd_seconds() <= 0

    def get_skill_cd_seconds(self) -> float:
        if self.manual_skill_cd <= 0 or self.last_skill_time <= 0:
            return 0.0
        return max(0.0, self.manual_skill_cd - (time.monotonic() - self.last_skill_time))

    def is_active(self, _region: Any = None) -> bool:
        active = getattr(self.ctx.input, "_active_slot", None)
        if active is None:
            active = self.combat_scenes.last_active_avatar_index
        return int(active) == self.index

    def is_active_no_index_rect(self, region: Any = None) -> bool:
        return self.is_active(region)

    def wait_skill_cd(self, token: Any = None) -> None:
        while not self.is_skill_ready():
            if _cancelled(token):
                return
            self.ctx.sleep(min(200, int(self.get_skill_cd_seconds() * 1000) + 1))

    def mouse_down(self, key: str = "left") -> None:
        value = str(key).lower()
        if value == "left":
            self.ctx.input.attack_down()
        elif value == "right":
            self.ctx.input.button_down("sprint")

    def mouse_up(self, key: str = "left") -> None:
        value = str(key).lower()
        if value == "left":
            self.ctx.input.attack_up()
        elif value == "right":
            self.ctx.input.button_up("sprint")

    def click(self, key: str = "left") -> None:
        value = str(key).lower()
        if value == "left":
            self.ctx.input.attack()
        elif value == "right":
            self.ctx.input.key_press("LSHIFT")

    def move_by(self, x: float, y: float) -> None:
        self.move_camera(x, y)

    def scroll(self, amount: int) -> None:
        self.ctx.input.vertical_scroll(float(amount))

    def key_down(self, key: str) -> None:
        self.ctx.input.key_down(str(key))

    def key_up(self, key: str) -> None:
        self.ctx.input.key_up(str(key))

    def key_press(self, key: str) -> None:
        self.ctx.input.key_press(str(key))

    @staticmethod
    def parse_action_scheduler_by_cd(avatar_name: str, value: str) -> float | None:
        name, text = str(avatar_name), str(value)
        if not name or not text:
            return None
        result: float | None = None
        for item in text.split(";"):
            fields = item.split(",", 1)
            if fields[0] != name:
                continue
            if len(fields) == 1:
                result = -1.0
            else:
                try:
                    result = float(fields[1])
                except ValueError:
                    result = -1.0
        return result

    Name = property(lambda self: self.name)
    Index = property(lambda self: self.index)
    combatAvatar = property(lambda self: self.combat_avatar)
    CombatAvatar = property(lambda self: self.combat_avatar)
    manualSkillCd = property(lambda self: self.manual_skill_cd,
                             lambda self, value: setattr(self, "manual_skill_cd", float(value)))
    ManualSkillCd = property(lambda self: self.manual_skill_cd,
                             lambda self, value: setattr(self, "manual_skill_cd", float(value)))
    lastSkillTime = property(lambda self: self.last_skill_time,
                             lambda self, value: setattr(self, "last_skill_time", float(value)))
    LastSkillTime = property(lambda self: self.last_skill_time,
                             lambda self, value: setattr(self, "last_skill_time", float(value)))
    isBurstReady = property(lambda self: self.is_burst_ready,
                            lambda self, value: setattr(self, "is_burst_ready", bool(value)))
    IsBurstReady = property(lambda self: self.is_burst_ready,
                            lambda self, value: setattr(self, "is_burst_ready", bool(value)))
    nameRect = property(lambda self: self.name_rect,
                        lambda self, value: setattr(self, "name_rect", value))
    NameRect = property(lambda self: self.name_rect,
                        lambda self, value: setattr(self, "name_rect", value))
    indexRect = property(lambda self: self.index_rect,
                         lambda self, value: setattr(self, "index_rect", value))
    IndexRect = property(lambda self: self.index_rect,
                         lambda self, value: setattr(self, "index_rect", value))
    Ct = property(lambda self: self.ct, lambda self, value: setattr(self, "ct", value))
    combatScenes = property(lambda self: self.combat_scenes)
    CombatScenes = property(lambda self: self.combat_scenes)
    trySwitch = try_switch
    Switch = switch
    TrySwitch = try_switch
    switchWithoutCts = switch_without_cts
    SwitchWithoutCts = switch_without_cts
    useSkill = use_skill
    UseSkill = use_skill
    afterUseSkill = after_use_skill
    AfterUseSkill = after_use_skill
    useBurst = use_burst
    UseBurst = use_burst
    Attack = attack
    Charge = charge
    Dash = dash
    Jump = jump
    Walk = walk
    moveCamera = move_camera
    MoveCamera = move_camera
    Wait = wait
    Ready = ready
    isSkillReady = is_skill_ready
    IsSkillReady = is_skill_ready
    getSkillCdSeconds = get_skill_cd_seconds
    GetSkillCdSeconds = get_skill_cd_seconds
    isActive = is_active
    IsActive = is_active
    isActiveNoIndexRect = is_active_no_index_rect
    IsActiveNoIndexRect = is_active_no_index_rect
    waitSkillCd = wait_skill_cd
    WaitSkillCd = wait_skill_cd
    mouseDown = mouse_down
    MouseDown = mouse_down
    mouseUp = mouse_up
    MouseUp = mouse_up
    Click = click
    moveBy = move_by
    MoveBy = move_by
    Scroll = scroll
    keyDown = key_down
    KeyDown = key_down
    keyUp = key_up
    KeyUp = key_up
    keyPress = key_press
    KeyPress = key_press
    parseActionSchedulerByCd = parse_action_scheduler_by_cd
    ParseActionSchedulerByCd = parse_action_scheduler_by_cd


class CombatScenes:
    def __init__(self, ctx, party_slots: Mapping[str, int] | None = None,
                 log: Callable[[str], None] = print,
                 to_collection: Callable[[list[Avatar]], Any] = lambda values: values):
        self.ctx = ctx
        self.party_slots = {
            str(name): int(slot) for name, slot in (party_slots or {}).items()
            if 1 <= int(slot) <= 4
        }
        self.log = log
        self.to_collection = to_collection
        self.avatars: list[Avatar] = []
        self.last_active_avatar_index = int(getattr(ctx.input, "_active_slot", 1))
        self.current_multi_game_status = None
        self.expected_team_avatar_num = len(self.party_slots) or 4

    @staticmethod
    def _team_names(config: Any) -> list[str]:
        config = _unwrap(config)
        if config is None:
            return []
        if isinstance(config, Mapping):
            raw = config.get("teamNames", config.get("TeamNames", ""))
        else:
            raw = getattr(config, "teamNames", getattr(config, "TeamNames", ""))
        return [name.strip() for name in re.split(r"[,;，]", str(raw)) if name.strip()]

    def initialize_team(self, _image_region: Any, auto_fight_config: Any = None):
        configured_names = self._team_names(auto_fight_config)
        if configured_names:
            slots = {name: index for index, name in enumerate(configured_names, 1)}
        else:
            slots = self.party_slots
        self.avatars = [
            Avatar(self, name, slot) for name, slot in sorted(slots.items(), key=lambda item: item[1])
        ]
        self.expected_team_avatar_num = len(self.avatars) or self.expected_team_avatar_num
        if self.avatars:
            self.log("[combat] 队伍：" + ", ".join(avatar.name for avatar in self.avatars))
        return self

    def initialize_team_silent(self, image_region: Any, auto_fight_config: Any = None):
        return self.initialize_team(image_region, auto_fight_config)

    def get_avatars(self):
        return self.to_collection(list(self.avatars))

    def refresh_team_avatar_index_rect_list(self, _image_region: Any) -> bool:
        return bool(self.avatars)

    def classify_avatar_cn_name(self, _image: Any, index: int):
        name = self.classify_avatar_name(_image, index)
        return {"Item1": name, "Item2": ""}

    def classify_avatar_name(self, _image: Any, index: int) -> str:
        avatar = self.select_avatar(int(index))
        return avatar.name if avatar else "未知角色"

    def check_team_initialized(self) -> bool:
        if not self.avatars:
            raise RuntimeError("队伍尚未初始化；请配置 party.json 或 TeamNames")
        return True

    def update_action_scheduler_by_cd(self, config: str) -> list[str]:
        for avatar in self.avatars:
            value = Avatar.parse_action_scheduler_by_cd(avatar.name, str(config))
            if value is not None:
                avatar.manual_skill_cd = value
        return [avatar.name for avatar in self.avatars]

    def select_avatar(self, value: Any) -> Avatar | None:
        raw = _unwrap(value)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return next((avatar for avatar in self.avatars if avatar.index == int(raw)), None)
        name = str(raw)
        return next((avatar for avatar in self.avatars if avatar.name == name), None)

    def current_avatar(self, _force: bool = False, _region: Any = None,
                       _ct: Any = None) -> str | None:
        index = self.get_active_avatar_index(_region, None)
        avatar = self.select_avatar(index)
        return avatar.name if avatar else None

    def get_active_avatar_index(self, _image_region: Any = None,
                                _context: Any = None) -> int:
        value = int(getattr(self.ctx.input, "_active_slot", self.last_active_avatar_index))
        self.last_active_avatar_index = value
        return value

    def initialize_team_old_ocr(self, content: Any):
        return self.initialize_team(content)

    @staticmethod
    def error_ocr_correction(name: str) -> str:
        return str(name)

    def before_task(self, token: Any = None) -> None:
        for avatar in self.avatars:
            avatar.ct = token

    def after_task(self) -> None:
        self.ctx.input.release_all()

    def dispose(self) -> None:
        self.after_task()

    avatarCount = property(lambda self: len(self.avatars))
    AvatarCount = avatarCount
    lastActiveAvatarIndex = property(
        lambda self: self.last_active_avatar_index,
        lambda self, value: setattr(self, "last_active_avatar_index", int(value)),
    )
    LastActiveAvatarIndex = property(
        lambda self: self.last_active_avatar_index,
        lambda self, value: setattr(self, "last_active_avatar_index", int(value)),
    )
    CurrentMultiGameStatus = property(
        lambda self: self.current_multi_game_status,
        lambda self, value: setattr(self, "current_multi_game_status", value),
    )
    currentMultiGameStatus = CurrentMultiGameStatus
    expectedTeamAvatarNum = property(lambda self: self.expected_team_avatar_num)
    ExpectedTeamAvatarNum = property(lambda self: self.expected_team_avatar_num)
    initializeTeam = initialize_team
    InitializeTeam = initialize_team
    initializeTeamSilent = initialize_team_silent
    InitializeTeamSilent = initialize_team_silent
    getAvatars = get_avatars
    GetAvatars = get_avatars
    refreshTeamAvatarIndexRectList = refresh_team_avatar_index_rect_list
    RefreshTeamAvatarIndexRectList = refresh_team_avatar_index_rect_list
    classifyAvatarCnName = classify_avatar_cn_name
    ClassifyAvatarCnName = classify_avatar_cn_name
    classifyAvatarName = classify_avatar_name
    ClassifyAvatarName = classify_avatar_name
    checkTeamInitialized = check_team_initialized
    CheckTeamInitialized = check_team_initialized
    updateActionSchedulerByCd = update_action_scheduler_by_cd
    UpdateActionSchedulerByCd = update_action_scheduler_by_cd
    selectAvatar = select_avatar
    SelectAvatar = select_avatar
    currentAvatar = current_avatar
    CurrentAvatar = current_avatar
    getActiveAvatarIndex = get_active_avatar_index
    GetActiveAvatarIndex = get_active_avatar_index
    initializeTeamOldOcr = initialize_team_old_ocr
    InitializeTeamOldOcr = initialize_team_old_ocr
    errorOcrCorrection = error_ocr_correction
    ErrorOcrCorrection = error_ocr_correction
    beforeTask = before_task
    BeforeTask = before_task
    afterTask = after_task
    AfterTask = after_task
    Dispose = dispose
