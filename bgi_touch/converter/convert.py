"""bettergi-scripts-list 脚本 → 本项目可用格式的转换器。

支持四类输入（自动识别）：
- JS 脚本包（含 manifest.json 的目录）→ 复制 + 兼容性扫描报告 COMPAT.md，
  内嵌键鼠宏资产同时转出 .touch.json 预览
- pathing JSON（含 positions 数组）→ 校验 + 统计 + 复制
- 键鼠宏 JSON（含 macroEvents）→ 触控时间线 .touch.json
- 战斗策略 .txt → DSL 校验 + 复制
"""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
from pathlib import Path

from ..combat.dsl import parse_combat_script
from ..macro.keymouse import convert_keymouse
from ..pathing.model import PathingTask

# JS 兼容性扫描：本移植版已实现 / 部分实现 / 未实现的 API
SUPPORTED = [
    "sleep", "log.", "click", "keyPress", "keyDown", "keyUp", "captureGameRegion",
    "RecognitionObject", "ImageRegion", "AutoSkipConfig", "TemplateMatch", "ocr",
    "findMulti", ".find(", "isEmpty", "isExist",
    "file.read", "file.write", "file.mkdir", "renamePathSync", "readTextSync",
    "readImageMat", "settings.", "notification.",
    "moveMouseBy", "keyMouseScript", "inputText", "http.request", "genshin.returnMainUi",
    "genshin.relogin", "chooseTalkOption", "runCombatScript", "leftButtonClick",
    "genshin.tp", "genshin.moveMapTo", "genshin.tpToStatueOfTheSeven",
    "genshin.teleportToStatue",
    "getPositionFromBigMap", "getPositionFromMap", "genshin.getBigMapZoomLevel",
    "genshin.claimMailRewards", "genshin.claimBattlePassRewards",
    "genshin.claimEncounterPointsRewards",
    "pathingScript",
    "dispatcher.addTimer", "dispatcher.addTrigger",
    "dispatcher.runAutoEatTask", "dispatcher.runAutoMusicGameTask",
    "dispatcher.runAutoAlbumTask", "dispatcher.runAutoAlbum",
    "dispatcher.runAutoGeniusInvokationTask", "dispatcher.runAutoStygianOnslaughtTask",
    "dispatcher.runQuickSereniteaPotTask", "dispatcher.runQuickClaimRewardTask",
    "dispatcher.runQuickBuyTask",
    "dispatcher.runUseRedemptionCodeTask",
    "dispatcher.runAutoArtifactSalvageTask",
    "dispatcher.runCountInventoryItemTask", "CountInventoryItemParam",
    "characterDevelopmentTask.getCharacter", "characterDevelopmentTask.getMultiCharacters",
]
PARTIAL = {
    "moveMouseTo": "触控无指针，移动为空操作（点击时直接给坐标即可）",
    "rightButton": "近似映射为 E 技能",
    "middleButton": "PC 重置视角，触控端为空操作",
    "verticalScroll": "触控端为空操作",
    "dispatcher.runTask": "已支持 AutoFight/AutoWood/AutoDomain/AutoCook/AutoFishing/AutoOpenChest/AutoBoss/AutoLeyLine/AutoEat/AutoMusicGame/AutoAlbum/AutoGeniusInvokation/AutoStygianOnslaught/QuickSereniteaPot/QuickClaimReward/QuickBuy/UseRedemptionCode/AutoArtifactSalvage/CountInventoryItem/GetGridIcons/InventoryCountComparison/CharacterDevelopment/OneDragon/Shell；AutoBoss/地脉已内置官方路线，幽境已内置活动导航状态机",
    "runAutoFightTask": "支持 combatStrategyPath；掉落拾取等附加参数忽略",
    "runAutoBossTask": "支持 BetterGI AutoBoss 官方路线、树脂策略、征讨之花导航、奖励汇总和 routePath/pathingFile 覆盖",
    "runAutoLeyLine": "支持 269 个官方地脉点位、632 份路线、树脂耗尽策略、奖励树脂优先级和 routePath/pathingFile 覆盖",
    "genshin.switchParty": "iOS 端按 config/party.json 切换当前角色槽位，不重组完整队伍",
    "genshin.setBigMapZoomLevel": "使用触控 pinch 近似 BetterGI 的 1.0-6.0 缩放等级",
    "genshin.autoFishing": "已迁移找鱼、选饵、抛竿、提竿、拉条与全天/昼夜时间策略闭环",
    "dispatcher.runAutoMusicGameTask": "已迁移六轨像素检测；需要先进入千音雅集曲目界面",
    "dispatcher.runAutoAlbumTask": "已迁移专辑导航、完成标志跳过和难度循环；需要先进入主题专辑界面",
    "dispatcher.runAutoGeniusInvokationTask": "已迁移完整回合状态机、投骰/骰子识别、元素调和、异常状态与阵亡切换；需准备七圣召唤识别资产",
    "PostMessage": "iOS 无后台窗口消息，按键与点击映射到 DeviceHub 前台触控输入",
    "dispatcher.runAutoStygianOnslaughtTask": "已迁移活动导航、困难模式、Boss 选择、战斗结果、地脉花、树脂优先级与继续/退出状态机；routePath 为可选入口覆盖",
    "dispatcher.runQuickSereniteaPotTask": "使用背包小道具页与原版模板快速进入/离开尘歌壶",
    "dispatcher.runQuickClaimRewardTask": "使用原版领取模板并将滚轮操作转换为触控上滑",
    "dispatcher.runUseRedemptionCodeTask": "使用 OCR 导航设置界面，并通过 DeviceHub 输入兑换码",
    "dispatcher.runAutoArtifactSalvageTask": "支持低星快速选择、五星 OCR+JavaScript 筛选；最终分解需要显式确认参数",
    "Shell": "跨平台宿主命令已支持；必须先在 config/shell.json 显式启用，且应只运行可信脚本",
    "dispatcher.runCountInventoryItemTask": "遍历 iOS 背包网格，通过详情名称和数量区域 OCR 返回单项或多项计数",
}
UNSUPPORTED = {
    "KeyMouseHook": "全局键鼠钩子为 Windows 专有",
}


def detect_kind(path: Path) -> str:
    if path.is_dir():
        return "js" if (path / "manifest.json").exists() else "unknown"
    if path.suffix.lower() == ".json":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return "unknown"
        if isinstance(raw, dict) and "macroEvents" in raw:
            return "keymouse"
        if isinstance(raw, dict) and "positions" in raw:
            return "pathing"
        return "unknown"
    if path.suffix.lower() == ".txt":
        return "combat"
    return "unknown"


def convert_keymouse_file(src: Path, out_dir: Path) -> Path:
    macro = json.loads(src.read_text(encoding="utf-8"))
    events, warnings = convert_keymouse(macro)
    out = out_dir / (src.stem + ".touch.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source": str(src),
        "format": "bgi-touch-macro/1",
        "warnings": warnings,
        "events": [dataclasses.asdict(e) for e in events],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def convert_pathing_file(src: Path, out_dir: Path) -> tuple[Path, dict]:
    task = PathingTask.load(src)
    out = out_dir / src.name
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, out)
    return out, task.summary()


def convert_combat_file(src: Path, out_dir: Path) -> tuple[Path, int]:
    lines = parse_combat_script(src.read_text(encoding="utf-8"))
    out = out_dir / src.name
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, out)
    return out, len(lines)


def scan_js_compat(pkg_dir: Path) -> dict:
    findings: dict[str, list[str]] = {"partial": [], "unsupported": []}
    for js in pkg_dir.rglob("*.js"):
        text = js.read_text(encoding="utf-8", errors="replace")
        rel = str(js.relative_to(pkg_dir))
        for pattern, why in PARTIAL.items():
            if re.search(re.escape(pattern), text, re.IGNORECASE):
                findings["partial"].append(f"`{rel}` 使用 `{pattern}` — {why}")
        for pattern, why in UNSUPPORTED.items():
            if re.search(re.escape(pattern), text, re.IGNORECASE):
                findings["unsupported"].append(f"`{rel}` 使用 `{pattern}` — {why}")
    return findings


def convert_js_package(src: Path, out_dir: Path) -> tuple[Path, dict]:
    dest = out_dir / src.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    findings = scan_js_compat(dest)

    macro_count = 0
    for asset in dest.rglob("*.json"):
        try:
            raw = json.loads(asset.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(raw, dict) and "macroEvents" in raw:
            convert_keymouse_file(asset, asset.parent)
            macro_count += 1

    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    verdict = "❌ 存在未移植依赖，可能无法完整运行" if findings["unsupported"] else (
        "⚠️ 部分能力为近似实现" if findings["partial"] else "✅ 未发现不兼容 API")
    report = [
        f"# 兼容性报告：{manifest.get('name', src.name)}",
        "",
        f"- 来源：`{src}`",
        f"- 结论：{verdict}",
        f"- 内嵌键鼠宏：{macro_count} 个（已转出 .touch.json）",
        "",
    ]
    if findings["unsupported"]:
        report += ["## 未移植的依赖", *[f"- {f}" for f in sorted(set(findings["unsupported"]))], ""]
    if findings["partial"]:
        report += ["## 近似/部分实现", *[f"- {f}" for f in sorted(set(findings["partial"]))], ""]
    (dest / "COMPAT.md").write_text("\n".join(report), encoding="utf-8")
    return dest, {"verdict": verdict, **{k: len(set(v)) for k, v in findings.items()}, "macros": macro_count}


def convert_any(src: str | Path, out_dir: str | Path) -> dict:
    src, out_dir = Path(src), Path(out_dir)
    kind = detect_kind(src)
    if kind == "js":
        dest, info = convert_js_package(src, out_dir / "js")
        return {"kind": kind, "output": str(dest), **info}
    if kind == "keymouse":
        out = convert_keymouse_file(src, out_dir / "keymouse")
        return {"kind": kind, "output": str(out)}
    if kind == "pathing":
        out, summary = convert_pathing_file(src, out_dir / "pathing")
        return {"kind": kind, "output": str(out), **summary}
    if kind == "combat":
        out, n = convert_combat_file(src, out_dir / "combat")
        return {"kind": kind, "output": str(out), "lines": n}
    raise ValueError(f"无法识别脚本类型: {src}")
