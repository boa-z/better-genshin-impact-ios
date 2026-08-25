"""BetterGI OneDragonFlowConfig orchestration for migrated iOS tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping

from ..engine.context import GENSHIN_BUNDLE_ID, GameContext
from ..engine.genshin_api import GenshinApi


def _get(obj: Mapping[str, Any], key: str, default: Any = None) -> Any:
    wanted = key.replace("_", "").casefold()
    for candidate, value in obj.items():
        if str(candidate).replace("_", "").casefold() == wanted:
            return value
    return default


def _bool(value: Any, default: bool = False) -> bool:
    """Coerce JSON/config variants without treating ``"false"`` as true."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "yes", "on", "是"}:
            return True
        if normalized in {"0", "false", "no", "off", "否", ""}:
            return False
    return default


@dataclass(frozen=True)
class OneDragonItem:
    id: str
    name: str
    enabled: bool


def parse_one_dragon_items(config: Mapping[str, Any]) -> list[OneDragonItem]:
    enabled = _get(config, "taskEnabledList", {}) or {}
    definitions = _get(config, "taskDefinitions", {}) or {}
    order = _get(config, "taskOrder", []) or list(enabled)
    if not isinstance(enabled, Mapping) or not isinstance(definitions, Mapping):
        raise ValueError("一条龙 TaskEnabledList/TaskDefinitions 必须是对象")
    if not isinstance(order, list):
        raise ValueError("一条龙 TaskOrder 必须是数组")
    legacy = not definitions
    items = []
    for raw_id in order:
        task_id = str(raw_id)
        if task_id not in enabled:
            continue
        name = task_id if legacy else definitions.get(task_id)
        if name is None:
            continue
        items.append(OneDragonItem(task_id, str(name), _bool(enabled[task_id])))
    next_task_id = str(_get(config, "nextTaskId", "") or "")
    if next_task_id:
        start = next((index for index, item in enumerate(items) if item.id == next_task_id), None)
        if start is not None:
            items = items[start:]
    return items


class OneDragonFlowTask:
    BUILTIN_NAMES = frozenset({
        "领取邮件", "合成树脂", "自动秘境", "自动首领讨伐", "自动幽境危战",
        "领取每日奖励", "领取尘歌壶奖励", "自动地脉花",
    })

    def __init__(
        self,
        ctx: GameContext,
        config: Mapping[str, Any],
        dispatcher: Any,
        *,
        continue_on_error: bool = True,
        close_game_on_completion: bool = True,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.config = dict(config)
        self.dispatcher = dispatcher
        self.continue_on_error = _bool(continue_on_error, True)
        self.close_game_on_completion = _bool(close_game_on_completion, True)
        self.log = log
        self.genshin = GenshinApi(ctx, log=log)

    @classmethod
    def load_config(cls, value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        path = Path(str(value)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"一条龙配置不存在：{path}")
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("一条龙配置根节点必须是对象")
        return data

    def _task_config(self, item: OneDragonItem) -> dict[str, Any]:
        configs = _get(self.config, "taskConfigs", {}) or {}
        if not isinstance(configs, Mapping):
            return {}
        value = configs.get(item.id, configs.get(item.name, {}))
        return dict(value) if isinstance(value, Mapping) else {}

    def _run_builtin(self, item: OneDragonItem) -> Any:
        task_config = self._task_config(item)
        if item.name == "领取邮件":
            return self.genshin.claimMailRewards()
        if item.name == "合成树脂":
            country = _get(
                task_config,
                "country",
                default=_get(self.config, "craftingBenchCountry", "枫丹"),
            )
            options: dict[str, Any] = {}
            min_keep = _get(
                task_config,
                "minResinToKeep",
                default=_get(self.config, "minResinToKeep", None),
            )
            if min_keep is not None:
                options["min_resin_to_keep"] = int(min_keep)
            timeout = _get(
                task_config,
                "timeoutSeconds",
                default=_get(self.config, "craftingBenchTimeoutSeconds", None),
            )
            if timeout is not None:
                options["timeout_s"] = float(timeout)
            return self.genshin.goCraftResin(country, **options)
        if item.name == "自动秘境":
            task_config.setdefault("domainRoundNum", 1)
            task_config.setdefault("partyName", _get(self.config, "partyName", ""))
            task_config.setdefault("domainName", _get(self.config, "domainName", ""))
            return self.dispatcher.run_auto_domain_task(task_config)
        if item.name == "自动首领讨伐":
            task_config.setdefault("bossName", _get(self.config, "autoBossName", ""))
            task_config.setdefault(
                "strategyName", _get(self.config, "autoBossStrategyName", "根据队伍自动选择")
            )
            task_config.setdefault("teamName", _get(self.config, "autoBossTeamName", ""))
            task_config.setdefault(
                "specifyRunCount", _bool(_get(self.config, "autoBossSpecifyRunCount", False))
            )
            task_config.setdefault("runCount", _get(self.config, "autoBossRunCount", 1))
            task_config.setdefault(
                "useTransientResin", _bool(_get(self.config, "autoBossUseTransientResin", False))
            )
            task_config.setdefault(
                "useFragileResin", _bool(_get(self.config, "autoBossUseFragileResin", False))
            )
            task_config.setdefault(
                "reviveRetryCount", _get(self.config, "autoBossReviveRetryCount", 3)
            )
            task_config.setdefault(
                "returnToStatueAfterEachRound",
                _bool(_get(self.config, "autoBossReturnToStatueAfterEachRound", False)),
            )
            task_config.setdefault(
                "rewardRecognitionEnabled",
                _bool(_get(self.config, "autoBossRewardRecognitionEnabled", False)),
            )
            task_config.setdefault("timeout", _get(self.config, "autoBossTimeout", 240))
            return self.dispatcher.run_auto_boss_task(task_config)
        if item.name == "自动幽境危战":
            return self.dispatcher.run_auto_stygian_onslaught_task(task_config)
        if item.name == "领取每日奖励":
            country = _get(
                task_config,
                "country",
                _get(self.config, "adventurersGuildCountry", "枫丹"),
            )
            guild_options: dict[str, Any] = {}
            party_name = str(_get(
                task_config,
                "dailyRewardPartyName",
                _get(self.config, "dailyRewardPartyName", ""),
            ) or "").strip()
            if party_name:
                guild_options["daily_reward_party_name"] = party_name
            only_do_once = _get(
                task_config,
                "onlyDoOnce",
                _get(task_config, "only_do_once", None),
            )
            if only_do_once is not None:
                guild_options["only_do_once"] = _bool(
                    only_do_once,
                    False,
                )
            guild = self.genshin.goToAdventurersGuild(country, **guild_options)
            battle_pass = self.genshin.claimBattlePassRewards()
            return guild and battle_pass
        if item.name == "领取尘歌壶奖励":
            # QuickSereniteaPot only deploys the gadget and enters/leaves the
            # realm.  The OneDragon item is the upstream reward job and must
            # use the full NPC/shop flow instead.
            reward_task = getattr(
                self.dispatcher, "run_serenitea_pot_rewards_task", None
            )
            if not callable(reward_task):
                raise RuntimeError("dispatcher 未提供尘歌壶奖励任务")
            return reward_task(task_config)
        if item.name == "自动地脉花":
            day_name = (
                datetime.now().astimezone() - timedelta(hours=4)
            ).strftime("%A")
            if not _bool(_get(self.config, f"leyLineRun{day_name}", True), True):
                self.log(f"[OneDragon] {day_name} 未启用自动地脉花，跳过")
                return True
            configured_count = int(_get(self.config, "leyLineRunCount", 0) or 0)
            if configured_count > 0:
                task_config.setdefault("count", configured_count)
            else:
                task_config.setdefault("count", 1)
            daily_type = str(_get(self.config, f"leyLine{day_name}Type", "") or "")
            daily_country = str(_get(self.config, f"leyLine{day_name}Country", "") or "")
            if daily_type:
                task_config.setdefault("leyLineOutcropType", daily_type)
            if daily_country:
                task_config.setdefault("country", daily_country)
            task_config.setdefault(
                "isResinExhaustionMode",
                _bool(_get(self.config, "leyLineResinExhaustionMode", False)),
            )
            task_config.setdefault(
                "openModeCountMin", _bool(_get(self.config, "leyLineOpenModeCountMin", False))
            )
            task_config.setdefault(
                "oneDragonMode", _bool(_get(self.config, "leyLineOneDragonMode", False))
            )
            return self.dispatcher.run_auto_leyline_task(task_config)
        raise KeyError(item.name)

    def _run_item(self, item: OneDragonItem) -> Any:
        if item.name in self.BUILTIN_NAMES:
            return self._run_builtin(item)
        config = self._task_config(item)
        task_name = str(config.pop("taskName", item.name))
        if task_name in self.dispatcher.IMPLEMENTED and task_name != "OneDragon":
            return self.dispatcher.run_task({"name": task_name, "config": config})
        raise NotImplementedError(
            f"一条龙配置组「{item.name}」尚未转换；请在 taskConfigs 中指定 taskName"
        )

    def _finish(self) -> None:
        action = str(_get(self.config, "completionAction", "") or "")
        if action not in {"关闭游戏", "关闭游戏和软件", "关机"}:
            return
        if not self.close_game_on_completion:
            self.log(f"[OneDragon] 已跳过完成动作：{action}")
            return
        try:
            self.ctx.device.stop_app(GENSHIN_BUNDLE_ID)
            self.log("[OneDragon] 完成动作：已关闭原神")
        except Exception as error:
            mode = self.ctx.device.background_current_app()
            self.log(f"[OneDragon] stop_app 失败（{error}），已通过 {mode} 挂起原神")
        if action == "关机":
            self.log("[OneDragon] iOS 移植不会执行宿主机关机")

    def run(self, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        items = parse_one_dragon_items(self.config)
        enabled = [item for item in items if item.enabled]
        result: dict[str, Any] = {
            "name": str(_get(self.config, "name", "") or ""),
            "completed": [],
            "skipped": [item.id for item in items if not item.enabled],
            "failed": {},
            "results": {},
        }
        self.log(f"[OneDragon] 启动，共 {len(enabled)} 个启用任务")
        for index, item in enumerate(enabled, start=1):
            if cancelled and cancelled():
                result["cancelled"] = True
                break
            self.log(f"[OneDragon] {index}/{len(enabled)}：{item.name} ({item.id})")
            try:
                task_result = self._run_item(item)
                result["results"][item.id] = task_result
                if task_result is False:
                    raise RuntimeError("任务返回失败")
                result["completed"].append(item.id)
            except Exception as error:
                result["failed"][item.id] = str(error)
                self.log(f"[OneDragon] {item.name} 失败：{error}")
                if not self.continue_on_error:
                    break
            self.ctx.sleep(1000)

        # BetterGI performs this notification-only check after the configured
        # one-dragon items. Keep it optional for lightweight custom
        # dispatchers that predate this task.
        raw_check_enabled = _get(self.config, "checkDailyRewardsEnabled", True)
        check_enabled = _bool(raw_check_enabled, True)
        check_rewards = getattr(self.dispatcher, "run_check_rewards_task", None)
        if (
            check_enabled
            and callable(check_rewards)
            and not (cancelled and cancelled())
        ):
            try:
                result["results"]["dailyRewardCheck"] = check_rewards(
                    _get(self.config, "checkDailyRewardsConfig", {}) or {},
                    cancelled,
                )
            except Exception as error:
                self.log(f"[OneDragon] 检查每日奖励失败：{error}")
        self._finish()
        self.log(
            f"[OneDragon] 完成 {len(result['completed'])}/{len(enabled)}，"
            f"失败 {len(result['failed'])}"
        )
        return result
