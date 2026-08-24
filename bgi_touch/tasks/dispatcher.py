"""BetterGI-compatible SoloTask dispatcher shared by JS, WebUI and CLI."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..combat.dsl import CombatExecutor
from ..engine.context import GameContext


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        if key in obj:
            return obj[key]
        wanted = key.replace("_", "").lower()
        for candidate, value in obj.items():
            if str(candidate).replace("_", "").lower() == wanted:
                return value
        return default
    try:
        value = getattr(obj, key)
    except (AttributeError, TypeError):
        wanted = key.replace("_", "").lower()
        for candidate in dir(obj):
            if candidate.replace("_", "").lower() == wanted:
                value = getattr(obj, candidate)
                break
        else:
            return default
    return default if value is None else value


def _requested(token: Any) -> bool:
    if token is None:
        return False
    try:
        method = getattr(token, "isCancellationRequested", None)
        if callable(method) and method():
            return True
    except Exception:
        pass
    try:
        if bool(getattr(token, "cancelled", False)):
            return True
    except Exception:
        pass
    return False


def _boolean(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _tuple4(value: Any, default: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return tuple(int(part) for part in value)  # type: ignore[return-value]
        except (TypeError, ValueError):
            pass
    return default


def _points(value: Any) -> dict[int, tuple[float, float]] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[int, tuple[float, float]] = {}
    for raw_key, raw_point in value.items():
        if not isinstance(raw_point, (list, tuple)) or len(raw_point) != 2:
            raise ValueError("触控点配置必须是 [x, y] 数组")
        try:
            result[int(raw_key)] = (float(raw_point[0]), float(raw_point[1]))
        except (TypeError, ValueError) as error:
            raise ValueError(f"触控点配置无效：{raw_key}") from error
    return result


class TaskDispatcher:
    """Run migrated BetterGI tasks with a single cancellation contract."""

    IMPLEMENTED = frozenset({
        "AutoFight", "AutoWood", "AutoDomain", "AutoCook", "AutoFishing", "AutoOpenChest",
        "AutoBoss", "AutoLeyLine", "AutoLeyLineOutcrop", "AutoEat", "AutoMusicGame", "AutoAlbum",
        "AutoGeniusInvokation", "AutoStygianOnslaught", "QuickSereniteaPot",
        "QuickClaimReward", "QuickBuy", "UseRedemptionCode", "AutoArtifactSalvage",
        "CountInventoryItem", "GetGridIcons", "InventoryCountComparison",
        "CharacterDevelopment", "OneDragon", "ScriptGroup", "MusicPlayer", "Shell",
        "OneKeyExpedition",
        "AutoTrack",
    })

    def __init__(
        self,
        ctx: GameContext,
        party_slots: dict[str, int] | None = None,
        log: Callable[[str], None] = print,
        cancelled: Callable[[], bool] | None = None,
        strategy_roots: list[str | Path] | None = None,
        restrict_strategy_roots: bool = False,
    ):
        self.ctx = ctx
        self.party_slots = party_slots or {}
        self.log = log
        self.cancelled = cancelled or (lambda: False)
        default_root = Path(__file__).resolve().parents[2] / "scripts" / "combat"
        self.strategy_roots = [
            Path(value).expanduser().resolve()
            for value in (strategy_roots or [default_root])
        ]
        self.restrict_strategy_roots = bool(restrict_strategy_roots)

    def _is_cancelled(self, token: Any = None) -> bool:
        return bool(self.cancelled()) or _requested(token)

    def _callback(self, token: Any = None) -> Callable[[], bool]:
        return lambda: self._is_cancelled(token)

    def _resolve_strategy(self, value: Any, *, allow_directory: bool = False) -> str | None:
        if value is None or not str(value).strip():
            return None
        raw = Path(str(value)).expanduser()
        candidates = [] if self.restrict_strategy_roots else (
            [raw] if raw.is_absolute() else [Path.cwd() / raw]
        )
        candidates.extend(root / raw for root in self.strategy_roots)
        if not raw.suffix:
            candidates.extend(
                root / f"{raw}.txt" for root in self.strategy_roots
            )
        for candidate in candidates:
            resolved = candidate.resolve()
            if self.restrict_strategy_roots and not any(
                resolved.is_relative_to(root) for root in self.strategy_roots
            ):
                continue
            if resolved.is_file() or (allow_directory and resolved.is_dir()):
                return str(resolved)
        return None if self.restrict_strategy_roots else str(value)

    def run_task(self, task: Any, ct: Any = None) -> Any:
        name = str(_value(task, "name", task))
        cfg = _value(task, "config", {}) or {}
        if name == "AutoFight":
            return self.run_auto_fight_task(cfg, ct)
        if name == "AutoWood":
            from .auto_wood import AutoWoodTask
            return AutoWoodTask(
                self.ctx,
                rounds=int(_value(cfg, "woodRoundNum", _value(cfg, "rounds", 1)) or 1),
                per_round_attacks=int(_value(cfg, "perRoundAttacks", 8) or 8),
                relogin_between=bool(_value(cfg, "reloginBetween", False)),
                log=self.log,
            ).run(cancelled=self._callback(ct))
        if name == "AutoDomain":
            return self.run_auto_domain_task(cfg, ct)
        if name == "AutoCook":
            return self.run_auto_cook_task(cfg, ct)
        if name in ("AutoFishing", "AutoFish"):
            return self.run_auto_fishing_task(cfg, ct)
        if name in ("AutoOpenChest", "OpenChest"):
            return self.run_auto_open_chest_task(cfg, ct)
        if name == "AutoEat":
            return self.run_auto_eat_task(cfg, ct)
        if name in ("AutoMusicGame", "AutoMusic"):
            return self.run_auto_music_game_task(cfg, ct)
        if name in ("AutoAlbum", "AutoMusicAlbum"):
            return self.run_auto_album_task(cfg, ct)
        if name == "AutoGeniusInvokation":
            return self.run_auto_genius_invokation_task(cfg, ct)
        if name == "AutoStygianOnslaught":
            return self.run_auto_stygian_onslaught_task(cfg, ct)
        if name == "AutoBoss":
            return self.run_auto_boss_task(cfg, ct)
        if name in ("AutoLeyLine", "AutoLeyLineOutcrop"):
            return self.run_auto_leyline_task(cfg, ct)
        if name in ("QuickSereniteaPot", "SereniteaPot"):
            return self.run_quick_serenitea_pot_task(cfg, ct)
        if name in ("QuickClaimReward", "OneKeyClaimReward"):
            return self.run_quick_claim_reward_task(cfg, ct)
        if name in ("OneKeyExpedition", "Expedition"):
            return self.run_one_key_expedition_task(cfg, ct)
        if name in ("AutoTrack", "QuestTracking"):
            return self.run_auto_track_task(cfg, ct)
        if name in ("QuickBuy", "BuyMax"):
            return self.run_quick_buy_task(cfg, ct)
        if name in ("UseRedemptionCode", "UseRedeemCode", "AutoRedeemCode"):
            return self.run_use_redemption_code_task(cfg, ct)
        if name in ("AutoArtifactSalvage", "ArtifactSalvage"):
            return self.run_auto_artifact_salvage_task(cfg, ct)
        if name == "CountInventoryItem":
            return self.run_count_inventory_item_task(cfg, ct)
        if name == "GetGridIcons":
            return self.run_get_grid_icons_task(cfg, ct)
        if name == "InventoryCountComparison":
            return self.run_inventory_count_comparison_task(cfg, ct)
        if name in ("CharacterDevelopment", "CharacterDevelopmentTask"):
            return self.run_character_development_task(cfg, ct)
        if name in ("OneDragon", "OneDragonFlow"):
            return self.run_one_dragon_task(cfg, ct)
        if name in ("ScriptGroup", "ScriptGroupTask"):
            return self.run_script_group_task(cfg, ct)
        if name in ("MusicPlayer", "PlayMusic", "AutoYuanQin"):
            return self.run_music_player_task(cfg, ct)
        if name == "Shell":
            return self.run_shell_task(cfg, ct)
        raise NotImplementedError(
            f"SoloTask {name} 尚未移植；当前已支持 {', '.join(sorted(self.IMPLEMENTED))}"
        )

    def run_auto_fight_task(self, param: Any = None, ct: Any = None) -> bool:
        from ..combat.finish import FightFinishConfig
        from .auto_fight import AutoFightTask
        strategy = self._resolve_strategy(_value(param, "combatStrategyPath", None))
        timeout = _value(param, "timeout", None)
        # BetterGI AutoFightParam.Timeout is expressed in seconds.
        timeout_s = float(timeout) if timeout else 120
        return AutoFightTask(
            self.ctx,
            combat_strategy_path=strategy,
            timeout_s=timeout_s,
            party_slots=self.party_slots,
            log=self.log,
            fight_finish_detect_enabled=_boolean(
                _value(param, "fightFinishDetectEnabled", True), True,
            ),
            finish_detect_config=FightFinishConfig.from_mapping(
                _value(param, "finishDetectConfig", {}) or {}
            ),
        ).run(cancelled=self._callback(ct))

    def run_one_dragon_task(self, param: Any = None, ct: Any = None) -> dict[str, Any]:
        from .one_dragon import OneDragonFlowTask

        source = _value(param, "configFile", _value(param, "path", None))
        if source is None:
            nested = _value(param, "flowConfig", _value(param, "oneDragonConfig", None))
            source = nested if nested is not None else param
        config = OneDragonFlowTask.load_config(source)
        return OneDragonFlowTask(
            self.ctx,
            config,
            self,
            continue_on_error=bool(_value(param, "continueOnError", True)),
            close_game_on_completion=bool(
                _value(param, "closeGameOnCompletion", True)
            ),
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_script_group_task(self, param: Any = None, ct: Any = None) -> dict[str, Any]:
        from .execution_records import ExecutionRecordStore
        from .script_group import ScriptGroupRoots, ScriptGroupRunner
        from .task_progress import TaskProgressStore

        sources = _value(param, "configFiles", _value(
            param, "paths", _value(param, "configFile", _value(param, "path", None))
        ))
        if isinstance(sources, (str, Path)):
            sources = [sources]
        if not isinstance(sources, (list, tuple)) or not sources:
            raise ValueError("ScriptGroup 需要 configFile/configFiles")
        roots = ScriptGroupRoots.build(
            javascript=_value(param, "javascriptRoot", _value(param, "scriptRoot", None)),
            key_mouse=_value(param, "keyMouseRoot", _value(param, "macroRoot", None)),
            pathing=_value(param, "pathingRoot", None),
        )
        store = TaskProgressStore(_value(param, "progressDirectory", None), log=self.log)
        runner = ScriptGroupRunner.load(
            self.ctx,
            [str(value) for value in sources],
            roots=roots,
            party_slots=self.party_slots,
            progress_store=store,
            execution_store=ExecutionRecordStore(
                _value(param, "executionRecordDirectory", _value(param, "recordsDirectory", None))
            ),
            continue_on_error=bool(_value(param, "continueOnError", True)),
            cancelled=self._callback(ct),
            log=self.log,
        )
        return runner.run(resume=_value(param, "resume", _value(param, "progress", None)))

    def run_music_player_task(self, param: Any = None, ct: Any = None) -> dict[str, Any]:
        from .music_player import MusicPlayerTask

        sources = _value(param, "files", _value(
            param, "sources", _value(param, "file", _value(param, "path", None))
        ))
        if isinstance(sources, (str, Path)):
            sources = [sources]
        if not isinstance(sources, (list, tuple)) or not sources:
            raise ValueError("MusicPlayer 需要 file/files 或 path/sources")
        custom_bpm = _value(param, "customBpm", None)
        if not bool(_value(param, "useCustomBpm", custom_bpm is not None)):
            custom_bpm = None
        return MusicPlayerTask(
            self.ctx,
            [str(value) for value in sources],
            layout_path=_value(param, "layoutPath", _value(param, "musicLayout", None)),
            playback_mode=str(_value(param, "playbackMode", "Sequential") or "Sequential"),
            speed=float(_value(param, "speed", 1.0) or 1.0),
            custom_bpm=None if custom_bpm is None else float(custom_bpm),
            transpose=int(_value(param, "transpose", 0) or 0),
            auto_switch_instrument=bool(_value(param, "autoSwitchInstrument", False)),
            start_position_s=float(_value(param, "startPositionSeconds", 0) or 0),
            search_text=str(_value(param, "searchText", "") or ""),
            format_filter=str(_value(
                param, "formatFilter", _value(param, "selectedFormatFilter", "")
            ) or ""),
            instrument_filter=str(_value(
                param, "instrumentFilter", _value(param, "selectedInstrumentFilter", "")
            ) or ""),
            start_index=int(_value(param, "startIndex", 0) or 0),
            start_track=_value(param, "startTrack", _value(param, "selectedTrack", None)),
            loop_count=int(_value(param, "loopCount", 1) or 1),
            max_instrument_pages=int(_value(param, "maxInstrumentPages", 20) or 20),
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_shell_task(self, param: Any = None, ct: Any = None) -> dict[str, Any]:
        from .shell_task import ShellTask

        command = param if isinstance(param, str) else _value(
            param, "command", _value(param, "shell", "")
        )
        raw_timeout = _value(param, "timeoutSeconds", _value(param, "timeout", None))
        raw_no_window = _value(param, "noWindow", None)
        raw_output = _value(param, "output", None)
        return ShellTask(
            str(command or ""),
            config_path=_value(param, "shellConfigPath", None),
            timeout_s=None if raw_timeout is None else float(raw_timeout),
            no_window=None if raw_no_window is None else bool(raw_no_window),
            output=None if raw_output is None else bool(raw_output),
            disable=bool(_value(param, "disable", False)),
            working_directory=_value(param, "workingDirectory", None),
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_auto_domain_task(self, param: Any = None, ct: Any = None) -> dict[str, int]:
        from .auto_domain import AutoDomainTask
        return AutoDomainTask(
            self.ctx,
            rounds=int(_value(param, "domainRoundNum", _value(param, "rounds", 1)) or 1),
            combat_strategy_path=self._resolve_strategy(
                _value(param, "combatStrategyPath", None)
            ),
            use_condensed_resin=bool(_value(param, "useCondensedResin", True)),
            reward_recognition_enabled=bool(
                _value(param, "rewardRecognitionEnabled", False)
            ),
            reward_max_pages=int(_value(param, "rewardMaxPages", 3) or 3),
            party_slots=self.party_slots,
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_auto_cook_task(self, param: Any = None, ct: Any = None) -> bool:
        from .auto_cook import AutoCookTask
        return AutoCookTask(
            self.ctx,
            check_interval_ms=int(_value(param, "checkIntervalMs", 400) or 400),
            stop_on_recover=bool(_value(param, "stopTaskWhenRecoverButtonDetected", True)),
            idle_timeout_s=float(_value(param, "idleTimeoutSeconds", 15) or 15),
            timeout_s=float(_value(param, "timeoutSeconds", 900) or 900),
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_auto_fishing_task(self, param: Any = None, ct: Any = None) -> bool:
        from .auto_fishing import AutoFishingTask
        target_catches = _value(
            param, "targetCatches", _value(param, "fishCount", 0)
        )
        return AutoFishingTask(
            self.ctx,
            target_catches=int(target_catches or 0),
            timeout_s=float(_value(
                param, "wholeProcessTimeoutSeconds", _value(param, "timeoutSeconds", 300)
            ) or 300),
            idle_timeout_s=float(_value(param, "idleTimeoutSeconds", 20) or 20),
            # BetterGI's SoloTask defaults to the full behaviour tree. The
            # iOS extension still permits disabling automatic casting explicitly.
            auto_throw_rod_enabled=bool(_value(param, "autoThrowRodEnabled", True)),
            throw_rod_timeout_s=float(_value(
                param, "throwRodTimeOutTimeoutSeconds",
                _value(param, "autoThrowRodTimeOut", 15),
            ) or 15),
            fishing_time_policy=_value(param, "fishingTimePolicy", 0),
            coop=bool(_value(param, "isCoop", _value(param, "coop", False))),
            quit_on_finish=bool(_value(param, "quitOnFinish", True)),
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_auto_open_chest_task(self, param: Any = None, ct: Any = None) -> bool:
        from .auto_open_chest import AutoOpenChestTask
        return AutoOpenChestTask(
            self.ctx,
            timeout_s=float(_value(param, "timeoutSeconds", 60) or 60),
            idle_timeout_s=float(_value(param, "idleTimeoutSeconds", 4) or 4),
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_auto_eat_task(self, param: Any = None, ct: Any = None) -> bool:
        from .auto_eat import AutoEatTask

        food_name = _value(param, "foodName", None)
        food_effect = _value(param, "foodEffectType", None)
        if food_name is not None:
            food_name = str(food_name).strip() or None
        if food_name is not None and food_effect is not None:
            raise ValueError("不能同时指定 foodName 和 foodEffectType")
        if food_effect is not None:
            defaults = _value(param, "defaultFoodNames", _value(param, "foodNames", {}))
            if not isinstance(defaults, Mapping):
                defaults = {}
            effect_names = {
                1: "ATKBoostingDish",
                2: "AdventurersDish",
                3: "DEFBoostingDish",
            }
            effect_key = effect_names.get(food_effect, str(food_effect))
            food_name = defaults.get(effect_key, defaults.get(str(food_effect)))
            if food_name is None:
                raise ValueError(
                    "foodEffectType 需要 defaultFoodNames/foodNames 配置，"
                    "且只支持攻击、冒险、防御类料理"
                )
            food_name = str(food_name).strip() or None
            if food_name is None:
                raise ValueError("foodEffectType 对应的料理名称为空")

        check_interval = _value(
            param,
            "checkIntervalMs",
            _value(param, "checkInterval", 150),
        )
        eat_interval_ms = _value(
            param,
            "eatIntervalMs",
            _value(param, "eatInterval", 1000),
        )
        return AutoEatTask(
            self.ctx,
            food_name=food_name,
            check_interval_ms=int(check_interval or 150),
            eat_interval_s=float(eat_interval_ms or 1000) / 1000.0,
            duration_s=float(_value(param, "durationSeconds", 0) or 0),
            health_roi=_tuple4(
                _value(param, "healthRoi", None),
                (720, 900, 480, 140),
            ),
            min_width_ref=int(_value(param, "minWidthRef", 55) or 55),
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_auto_music_game_task(self, param: Any = None, ct: Any = None) -> bool:
        from .auto_music import AutoMusicGameTask

        lane_x = _value(param, "laneX", None)
        keys = _value(param, "keys", None)
        return AutoMusicGameTask(
            self.ctx,
            lane_x=lane_x if lane_x is not None else (417, 628, 844, 1061, 1277, 1493),
            lane_y=int(_value(param, "laneY", 921) or 921),
            keys=keys if keys is not None else ("A", "S", "D", "J", "K", "L"),
            threshold=int(_value(param, "threshold", 220) or 220),
            poll_interval_ms=int(_value(param, "pollIntervalMs", 35) or 35),
            timeout_s=float(_value(param, "timeoutSeconds", 900) or 900),
            idle_timeout_s=float(_value(param, "idleTimeoutSeconds", 8) or 8),
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_auto_album_task(self, param: Any = None, ct: Any = None) -> bool:
        from .auto_album import AutoAlbumTask

        music_level = _value(
            param,
            "musicLevel",
            _value(param, "difficulty", _value(param, "difficulties", "传说")),
        )
        return AutoAlbumTask(
            self.ctx,
            music_level=music_level,
            must_canorus_level=bool(
                _value(param, "mustCanorusLevel", _value(param, "onlyGrandOde", False))
            ),
            song_count=int(_value(param, "songCount", 13) or 13),
            track_timeout_s=float(_value(param, "trackTimeoutSeconds", 900) or 900),
            timeout_s=float(_value(param, "timeoutSeconds", 7200) or 7200),
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_auto_genius_invokation_task(self, param: Any = None, ct: Any = None) -> bool:
        from .auto_tcg import AutoGeniusInvokationTask

        strategy = _value(param, "strategy", None)
        if strategy is None:
            strategy_path = _value(param, "strategyPath", None)
            if strategy_path:
                path = Path(str(strategy_path)).expanduser()
                if not path.is_file():
                    project_root = Path(__file__).resolve().parents[2]
                    path = project_root / "scripts" / "tcg" / path.name
                if path.is_file():
                    strategy = path.read_text(encoding="utf-8")
        if not strategy:
            raise ValueError("AutoGeniusInvokation 需要 strategy 或 strategyPath")
        return AutoGeniusInvokationTask(
            self.ctx,
            str(strategy),
            character_points=_points(_value(param, "characterPoints", None)),
            skill_points=_points(_value(param, "skillPoints", None)),
            max_commands=_value(param, "maxCommands", None),
            max_rounds=int(_value(param, "maxRounds", 20) or 20),
            timeout_s=float(_value(param, "timeoutSeconds", 900) or 900),
            asset_dir=_value(param, "assetDir", None) or None,
            card_config_path=_value(param, "cardConfigPath", None) or None,
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_auto_stygian_onslaught_task(self, param: Any = None, ct: Any = None) -> bool:
        from .auto_stygian import AutoStygianOnslaughtTask

        route = self._resolve_route(param, kind="stygian")
        strategy = _value(
            param,
            "combatScriptBagPath",
            _value(param, "combatStrategyPath", None),
        )
        strategy = self._resolve_strategy(strategy, allow_directory=True)
        raw_priority = _value(
            param, "resinPriorityList", ("浓缩树脂", "原粹树脂")
        )
        if isinstance(raw_priority, str):
            resin_priority = [
                item.strip() for item in raw_priority.replace("，", ",").split(",")
                if item.strip()
            ]
        else:
            try:
                resin_priority = [str(item) for item in raw_priority]
            except TypeError:
                resin_priority = ["浓缩树脂", "原粹树脂"]
        salvage_options = {
            "star": int(
                _value(param, "artifactSalvageStar", _value(param, "maxArtifactStar", 4))
                or 4
            ),
            "javascript": _value(param, "artifactSalvageJavaScript", None),
            "artifact_set_filter": _value(param, "artifactSetFilter", None),
            "max_num_to_check": int(_value(param, "maxNumToCheck", 100) or 100),
            "recognition_failure_policy": str(
                _value(param, "recognitionFailurePolicy", "Skip") or "Skip"
            ),
            "confirm_quick_salvage": bool(
                _value(param, "confirmQuickSalvage", False)
            ),
            "confirm_salvage": bool(
                _value(
                    param,
                    "confirmArtifactSalvage",
                    _value(param, "confirmSalvage", False),
                )
            ),
        }
        return AutoStygianOnslaughtTask(
            self.ctx,
            route_path=route,
            boss_num=int(_value(param, "bossNum", 1) or 1),
            combat_strategy_path=str(strategy) if strategy else None,
            timeout_s=float(
                _value(param, "timeoutSeconds", _value(param, "timeout", 360))
                or 360
            ),
            party_slots=self.party_slots,
            auto_artifact_salvage=bool(
                _value(param, "autoArtifactSalvage", False)
            ),
            specify_resin_use=bool(_value(param, "specifyResinUse", False)),
            resin_priority_list=resin_priority,
            original_resin_use_count=int(
                _value(param, "originalResinUseCount", 0) or 0
            ),
            condensed_resin_use_count=int(
                _value(param, "condensedResinUseCount", 0) or 0
            ),
            transient_resin_use_count=int(
                _value(param, "transientResinUseCount", 0) or 0
            ),
            fragile_resin_use_count=int(
                _value(param, "fragileResinUseCount", 0) or 0
            ),
            fight_team_name=str(_value(param, "fightTeamName", "") or ""),
            artifact_salvage_options=salvage_options,
            max_battle_failures=int(_value(param, "maxBattleFailures", 20) or 20),
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def _resolve_route(self, param: Any, *, kind: str, name: str = "") -> str | None:
        from pathlib import Path

        explicit = _value(param, "routePath", _value(param, "pathingFile", None))
        if explicit:
            path = Path(str(explicit)).expanduser()
            if path.is_file():
                return str(path)
            project_root = Path(__file__).resolve().parents[2]
            candidate = project_root / "assets" / "pathing" / kind / path.name
            if candidate.is_file():
                return str(candidate)
        if kind == "boss" and name:
            root = Path(__file__).resolve().parents[2] / "assets" / "pathing" / "boss"
            for candidate in (root / f"{name}前往.json", root / f"{name}.json"):
                if candidate.is_file():
                    return str(candidate)
            matches = sorted(root.glob(f"*{name}*前往.json"))
            if matches:
                return str(matches[0])
        return None

    def run_auto_boss_task(self, param: Any = None, ct: Any = None) -> dict[str, int]:
        from .auto_encounter import AutoBossTask

        boss_name = str(_value(param, "bossName", "") or "")
        explicit_route = _value(param, "routePath", _value(param, "pathingFile", None))
        route = (
            self._resolve_route(
                {"routePath": explicit_route}, kind="boss", name=boss_name
            )
            if explicit_route else None
        )
        strategy = self._resolve_strategy(
            _value(param, "combatStrategyPath", None)
        )
        if not strategy:
            strategy_name = str(_value(param, "strategyName", "") or "")
            if strategy_name and strategy_name != "根据队伍自动选择":
                strategy = self._resolve_strategy(strategy_name)
        timeout = _value(param, "timeout", 240)
        return AutoBossTask(
            self.ctx,
            boss_name=boss_name,
            route_path=route,
            specify_run_count=bool(_value(param, "specifyRunCount", False)),
            rounds=int(_value(param, "runCount", _value(param, "rounds", 1)) or 1),
            use_transient_resin=bool(_value(param, "useTransientResin", False)),
            use_fragile_resin=bool(_value(param, "useFragileResin", False)),
            revive_retry_count=int(_value(param, "reviveRetryCount", 3) or 0),
            return_to_statue_after_each_round=bool(
                _value(param, "returnToStatueAfterEachRound", False)
            ),
            reward_recognition_enabled=bool(
                _value(param, "rewardRecognitionEnabled", False)
            ),
            reward_max_pages=int(_value(param, "rewardMaxPages", 3) or 3),
            team_name=str(_value(param, "teamName", "") or ""),
            combat_strategy_path=str(strategy) if strategy else None,
            timeout_s=float(timeout),
            party_slots=self.party_slots,
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_auto_leyline_task(self, param: Any = None, ct: Any = None) -> bool:
        from .auto_leyline import AutoLeyLineOutcropTask

        route = self._resolve_route(param, kind="leyline")
        fight_config = _value(param, "fightConfig", {}) or {}
        strategy = _value(
            param, "combatStrategyPath",
            _value(fight_config, "combatStrategyPath", None),
        )
        strategy = self._resolve_strategy(strategy)
        if not strategy:
            strategy_name = str(_value(fight_config, "strategyName", "") or "")
            if strategy_name:
                strategy = self._resolve_strategy(strategy_name)
        timeout = _value(
            fight_config, "timeout", _value(param, "timeout", 120)
        )
        return AutoLeyLineOutcropTask(
            self.ctx,
            route_path=route,
            count=int(_value(param, "count", _value(param, "rounds", 1)) or 1),
            country=str(_value(param, "country", "蒙德") or "蒙德"),
            ley_line_type=str(
                _value(param, "leyLineOutcropType", _value(param, "type", "启示之花"))
                or "启示之花"
            ),
            open_mode_count_min=bool(_value(param, "openModeCountMin", False)),
            resin_exhaustion_mode=bool(_value(param, "isResinExhaustionMode", False)),
            use_adventurer_handbook=bool(_value(param, "useAdventurerHandbook", False)),
            friendship_team=str(_value(param, "friendshipTeam", "") or ""),
            team=str(_value(param, "team", "") or ""),
            use_fragile_resin=bool(_value(param, "useFragileResin", False)),
            use_transient_resin=bool(_value(param, "useTransientResin", False)),
            scan_drops_after_reward_enabled=bool(
                _value(param, "scanDropsAfterRewardEnabled", False)
            ),
            scan_drops_after_reward_seconds=int(
                _value(param, "scanDropsAfterRewardSeconds", 12) or 0
            ),
            combat_strategy_path=str(strategy) if strategy else None,
            timeout_s=float(timeout),
            party_slots=self.party_slots,
            one_dragon_mode=bool(_value(param, "oneDragonMode", False)),
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_quick_serenitea_pot_task(self, param: Any = None, ct: Any = None) -> bool:
        from .quick_serenitea import QuickSereniteaPotTask

        return QuickSereniteaPotTask(
            self.ctx,
            timeout_s=float(_value(param, "timeoutSeconds", 35) or 35),
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_quick_claim_reward_task(self, param: Any = None, ct: Any = None) -> int:
        from .quick_claim import QuickClaimRewardTask

        mode = str(_value(param, "mode", _value(param, "hotkeyMode", "点按一次")) or "")
        scroll = bool(_value(param, "scrollDown", _value(param, "scrollDownEnabled", False)))
        return QuickClaimRewardTask(
            self.ctx,
            max_clicks=int(_value(param, "maxClicks", 30) or 30),
            scroll_down=scroll or mode in ("按住持续", "hold", "continuous"),
            max_scrolls=int(_value(param, "maxScrolls", 3) or 3),
            timeout_s=float(_value(param, "timeoutSeconds", 30) or 30),
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_one_key_expedition_task(self, param: Any = None, ct: Any = None) -> bool:
        from .expedition import OneKeyExpeditionTask

        return OneKeyExpeditionTask(
            self.ctx,
            collect_retries=int(_value(param, "collectRetries", 2) or 2),
            redispatch_retries=int(_value(param, "redispatchRetries", 3) or 3),
            timeout_s=float(_value(param, "timeoutSeconds", 12) or 12),
            close_page=_boolean(_value(param, "closePage", True), True),
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_auto_track_task(self, param: Any = None, ct: Any = None) -> bool:
        from .auto_track import AutoTrackTask

        return AutoTrackTask(
            self.ctx,
            timeout_s=float(_value(param, "timeoutSeconds", 120) or 120),
            far_distance_m=int(_value(param, "farDistance", 150) or 150),
            arrival_distance_m=int(_value(param, "arrivalDistance", 3) or 3),
            teleport_when_far=_boolean(
                _value(param, "teleportWhenFar", True), True,
            ),
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_quick_buy_task(self, param: Any = None, ct: Any = None) -> bool:
        from .quick_buy import QuickBuyTask

        shop = _value(param, "serenitea", _value(param, "isSereniteaPot", None))
        return QuickBuyTask(
            self.ctx,
            serenitea=None if shop is None else bool(shop),
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_use_redemption_code_task(self, param: Any = None, ct: Any = None) -> dict[str, bool]:
        from .redeem_code import UseRedemptionCodeTask

        codes = _value(param, "codes", _value(param, "list", _value(param, "code", None)))
        return UseRedemptionCodeTask(
            self.ctx,
            codes,
            timeout_s=float(_value(param, "timeoutSeconds", 120) or 120),
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_auto_artifact_salvage_task(self, param: Any = None, ct: Any = None) -> dict:
        from .artifact_salvage import AutoArtifactSalvageTask

        return AutoArtifactSalvageTask(
            self.ctx,
            star=int(_value(param, "star", _value(param, "maxArtifactStar", 4)) or 4),
            javascript=_value(param, "javaScript", _value(param, "javascript", None)),
            artifact_set_filter=_value(param, "artifactSetFilter", None),
            max_num_to_check=int(_value(param, "maxNumToCheck", 100) or 100),
            recognition_failure_policy=str(
                _value(param, "recognitionFailurePolicy", "Skip") or "Skip"
            ),
            confirm_quick_salvage=bool(
                _value(param, "confirmQuickSalvage", _value(param, "confirmLowStarSalvage", False))
            ),
            confirm_salvage=bool(_value(param, "confirmSalvage", False)),
            max_pages=int(_value(param, "maxPages", 12) or 12),
            timeout_s=float(_value(param, "timeoutSeconds", 300) or 300),
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_count_inventory_item_task(self, param: Any = None, ct: Any = None) -> Any:
        from .inventory_grid import CountInventoryItemTask

        category = _value(param, "gridScreenName", _value(param, "category", None))
        if category is None:
            raise ValueError("CountInventoryItem 需要 gridScreenName")
        item_names = _value(param, "itemNames", None)
        if isinstance(item_names, str):
            item_names = [item_names]
        return CountInventoryItemTask(
            self.ctx,
            category,
            item_name=_value(param, "itemName", None),
            item_names=item_names,
            icon_recognition_mode=_value(param, "iconRecognitionMode", "GridIcon"),
            max_pages=int(_value(param, "maxPages", 100) or 100),
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_get_grid_icons_task(self, param: Any = None, ct: Any = None) -> list[str]:
        from .inventory_grid import GetGridIconsTask

        category = _value(
            param, "gridScreenName", _value(param, "gridName", _value(param, "category", None))
        )
        if category is None:
            raise ValueError("GetGridIcons 需要 gridScreenName/gridName")
        max_num = _value(param, "maxNumToGet", None)
        return GetGridIconsTask(
            self.ctx,
            category,
            star_as_suffix=bool(_value(param, "starAsSuffix", False)),
            max_num_to_get=int(max_num) if max_num is not None else None,
            max_pages=int(_value(param, "maxPages", 100) or 100),
            output_dir=_value(param, "outputDirectory", _value(param, "outputDir", None)),
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_inventory_count_comparison_task(self, param: Any = None, ct: Any = None) -> str:
        from .inventory_grid import InventoryCountComparisonTask

        return InventoryCountComparisonTask(
            self.ctx,
            _value(param, "target", "CurrentPage"),
            max_pages=int(_value(param, "maxPages", 100) or 100),
            output_dir=_value(param, "outputDirectory", _value(param, "outputDir", None)),
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_character_development_task(self, param: Any = None, ct: Any = None) -> Any:
        from .character_development import CharacterDevelopmentTask

        task = CharacterDevelopmentTask(
            self.ctx,
            max_pages=int(_value(param, "maxPages", 30) or 30),
            timeout_s=float(_value(param, "timeoutSeconds", 900) or 900),
            log=self.log,
        )
        categories = _value(param, "categories", None)
        name = _value(param, "characterName", _value(param, "name", None))
        names = _value(param, "characterNames", _value(param, "names", None))
        if name is not None and names is not None:
            raise ValueError("characterName 和 characterNames 不能同时使用")
        if name is not None:
            results = task.run([str(name)], categories, cancelled=self._callback(ct))
            return results[0] if results else None
        if names is None:
            raise ValueError("CharacterDevelopment 需要 characterName 或 characterNames")
        if isinstance(names, str):
            raise ValueError("characterNames 必须是数组")
        return task.run(names, categories, cancelled=self._callback(ct))

    def run_combat_script(self, script: str, avatar: str | None = None) -> Any:
        return CombatExecutor.for_context(
            self.ctx, party_slots=self.party_slots, log=self.log
        ).run(str(script))

    def add_timer(self, timer: Any) -> None:
        name = str(_value(timer, "name", timer))
        self.ctx.triggers.clear()
        config = _value(timer, "config", {}) or {}
        if name == "AutoPick":
            force_interaction = _value(
                config,
                "forceInteraction",
                _value(config, "force_interaction", False),
            )
            self.ctx.enable_trigger(
                name,
                force_interaction=bool(force_interaction),
                mode=_value(config, "mode", "Whitelist"),
                whitelist=_value(
                    config,
                    "whitelist",
                    _value(config, "whiteList", _value(config, "pickWhitelist", None)),
                ),
                blacklist=_value(
                    config,
                    "blacklist",
                    _value(config, "blackList", _value(config, "pickBlacklist", None)),
                ),
                fuzzy_blacklist=_value(
                    config,
                    "fuzzyBlacklist",
                    _value(config, "fuzzy_blacklist", None),
                ),
                whitelist_exclusions=_value(
                    config,
                    "whitelistExclusions",
                    _value(config, "doNotPickList", None),
                ),
                blacklist_mode_pick_enabled=bool(_value(
                    config, "blacklistModePickEnabled", False
                )),
                whitelist_mode_do_not_pick_enabled=bool(_value(
                    config, "whitelistModeDoNotPickEnabled", True
                )),
            )
        elif name in ("AutoEat", "自动吃药"):
            self.ctx.enable_trigger(
                "AutoEat",
                check_interval_ms=int(
                    _value(config, "checkIntervalMs", _value(config, "checkInterval", 150))
                    or 150
                ),
                eat_interval_ms=int(
                    _value(config, "eatIntervalMs", _value(config, "eatInterval", 8000))
                    or 8000
                ),
                health_roi=_tuple4(_value(config, "healthRoi", None), (720, 900, 480, 140)),
                min_width_ref=int(_value(config, "minWidthRef", 55) or 55),
            )
        elif name == "AutoSkip":
            raw_priorities = str(_value(config, "customPriorityOptions", "") or "")
            priority_texts = [
                value.strip()
                for value in raw_priorities.replace("\r", "\n").replace(";", "\n").split("\n")
                if value.strip()
            ] if bool(_value(config, "customPriorityOptionsEnabled", False)) else []
            self.ctx.enable_trigger(
                name,
                click_option=str(
                    _value(config, "clickChatOption", "优先选择第一个选项")
                    or "优先选择第一个选项"
                ),
                priority_texts=priority_texts,
                quickly_skip=bool(
                    _value(config, "quicklySkipConversationsEnabled", True)
                ),
                skip_built_in_options=bool(
                    _value(config, "skipBuiltInClickOptions", False)
                ),
                after_choose_delay_ms=max(
                    0, int(_value(config, "afterChooseOptionSleepDelay", 0) or 0)
                ),
                before_confirm_delay_ms=max(
                    0, int(_value(config, "beforeClickConfirmDelay", 0) or 0)
                ),
                close_popup_pages=bool(_value(config, "closePopupPagedEnabled", True)),
                auto_re_explore_enabled=_boolean(
                    _value(config, "autoReExploreEnabled", True), True,
                ),
            )
        elif name in ("MapMask", "地图遮罩"):
            self.ctx.enable_trigger(
                "MapMask",
                map_name=str(
                    _value(config, "mapName", _value(config, "map_name", "Teyvat"))
                    or "Teyvat"
                ),
                mini_map_enabled=bool(
                    _value(
                        config,
                        "miniMapMaskEnabled",
                        _value(config, "mini_map_enabled", True),
                    )
                ),
            )
        elif name in ("SkillCd", "技能冷却"):
            self.ctx.enable_trigger(
                "SkillCd",
                party_slots=self.party_slots or None,
                custom_cd_list=_value(config, "customCdList", []),
                trigger_on_skill_use=bool(_value(config, "triggerOnSkillUse", False)),
                hide_when_zero=bool(_value(config, "hideWhenZero", False)),
                p_x=float(_value(config, "pX", 1520.0) or 0),
                p_y=float(_value(config, "pY", 245.0) or 0),
                gap=float(_value(config, "gap", 91.2) or 0),
                scale=float(_value(config, "scale", 1.0) or 0),
                background_normal_color=str(
                    _value(config, "backgroundNormalColor", "#FFFFFFFF") or "#FFFFFFFF"
                ),
                text_normal_color=str(
                    _value(config, "textNormalColor", "#DA4A23FF") or "#DA4A23FF"
                ),
                background_ready_color=str(
                    _value(config, "backgroundReadyColor", "#FFFFFFFF") or "#FFFFFFFF"
                ),
                text_ready_color=str(
                    _value(config, "textReadyColor", "#5DCC17FF") or "#5DCC17FF"
                ),
            )
        elif name in ("GameLoading", "自动开门"):
            self.ctx.enable_trigger(
                "GameLoading",
                timeout_s=float(_value(config, "timeoutSeconds", 300) or 300),
            )
        elif name in ("QuickTeleport", "快速传送"):
            self.ctx.enable_trigger(
                "QuickTeleport",
                teleport_list_click_delay_ms=int(
                    _value(config, "teleportListClickDelay", 200) or 0
                ),
                wait_teleport_panel_delay_ms=int(
                    _value(config, "waitTeleportPanelDelay", 50) or 0
                ),
                hotkey_tp_enabled=bool(_value(config, "hotkeyTpEnabled", False)),
            )
        else:
            self.ctx.enable_trigger(name)

    def add_trigger(self, trigger: Any) -> None:
        self.ctx.enable_trigger(str(_value(trigger, "name", trigger)))

    def clear_all_triggers(self) -> None:
        self.ctx.triggers.clear()
