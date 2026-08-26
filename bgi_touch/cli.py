"""bgi-touch 命令行入口。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _context(args):
    from .engine.context import GameContext
    from .input.layout import DEFAULT_LAYOUT
    return GameContext(
        mcp_url=args.url,
        layout_path=args.layout or DEFAULT_LAYOUT,
        keymap_profile=None if args.no_keymap_profile else args.keymap_profile,
        keymap_profile_path=args.keymap_profile_file,
        devicehub_config_path=args.devicehub_config,
        device_id=args.device_id,
        game_bundle_id=getattr(args, "game_bundle_id", None),
    )


def _load_party() -> dict[str, int]:
    p = PROJECT_ROOT / "config" / "party.json"
    if p.exists():
        return {k: int(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
    return {}


def cmd_status(args) -> int:
    ctx = _context(args)
    st = ctx.device.status()
    try:
        app = ctx.device.app_status(ctx.game_bundle_id)
    except Exception as e:  # 设备侧查询偶发超时，不影响主要状态
        app = {"error": str(e)}
    print(json.dumps({"device": st, "genshin": app,
                      "transform": {"w": ctx.transform.device_width, "h": ctx.transform.device_height,
                                    "scale": round(ctx.transform.scale, 4)}},
                     ensure_ascii=False, indent=2))
    ctx.close()
    return 0


def cmd_screenshot(args) -> int:
    ctx = _context(args)
    png = ctx.device.screenshot_png()
    out = Path(args.output)
    out.write_bytes(png)
    print(f"已保存 {out}（{len(png)} 字节）")
    ctx.close()
    return 0


def cmd_launch(args) -> int:
    ctx = _context(args)
    ctx.launch_game()
    print("原神已启动")
    ctx.close()
    return 0


def cmd_close_game(args) -> int:
    ctx = _context(args)
    try:
        ctx.device.stop_app(ctx.game_bundle_id)
        print("原神已停止")
    except Exception as e:
        mode = ctx.device.background_current_app()
        print(f"MCP 无法终止当前 App（{e}），已通过 {mode} 将原神移出前台并挂起")
    finally:
        ctx.close()
    return 0


def cmd_calibrate(args) -> int:
    """截图并叠加当前布局的按钮标注，输出到文件用于人工校准。"""
    import cv2
    ctx = _context(args)
    bgr = ctx.capture_bgr()
    h, w = bgr.shape[:2]
    for name, (nx, ny) in ctx.layout.buttons.items():
        x, y = int(nx * w), int(ny * h)
        cv2.circle(bgr, (x, y), 14, (0, 0, 255), 2)
        cv2.putText(bgr, name, (x + 16, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    jx = int(ctx.layout.joystick_center[0] * w)
    jy = int(ctx.layout.joystick_center[1] * h)
    cv2.circle(bgr, (jx, jy), int(ctx.layout.joystick_radius_n * w), (0, 255, 0), 2)
    nx, ny, nw, nh = ctx.layout.camera_region
    cv2.rectangle(bgr, (int(nx * w), int(ny * h)), (int((nx + nw) * w), int((ny + nh) * h)), (255, 160, 0), 2)
    cv2.imwrite(args.output, bgr)
    print(f"已保存标注截图 {args.output}；对照游戏画面修改 config/controls/genshin-default.json")
    ctx.close()
    return 0


def cmd_convert(args) -> int:
    from .converter.convert import convert_any
    out_dir = Path(args.output)
    ok = fail = 0
    for src in args.sources:
        try:
            info = convert_any(src, out_dir)
            print(json.dumps(info, ensure_ascii=False))
            ok += 1
        except Exception as e:  # 继续转换其余脚本
            print(f"[失败] {src}: {e}", file=sys.stderr)
            fail += 1
    print(f"完成：成功 {ok}，失败 {fail}，输出目录 {out_dir}")
    return 1 if fail and not ok else 0


def cmd_log_parse(args) -> int:
    """Analyze BetterGI log files without creating a DeviceHub context."""
    from .tasks.log_parse import (
        discover_log_files,
        load_diary_action_items,
        parse_log_files,
        render_text,
    )

    sources = []
    for raw in args.sources:
        path = Path(raw).expanduser()
        if path.is_dir():
            discovered = discover_log_files(path)
            if args.date:
                sources.extend((item[0], args.date) for item in discovered)
            else:
                sources.extend(discovered)
        else:
            sources.append((path, args.date) if args.date else path)
    if not sources:
        print("未找到可分析的 BetterGI 日志文件", file=sys.stderr)
        return 2

    diary_sources = [Path(item).expanduser() for item in getattr(args, "diary_file", []) or []]
    diary_cache_dir = getattr(args, "diary_cache_dir", None)
    if diary_cache_dir:
        cache_root = Path(diary_cache_dir).expanduser()
        game_uid = str(getattr(args, "game_uid", "") or "").strip()
        if game_uid:
            cache_root = cache_root / game_uid / "travelsdiarydetail"
        elif (cache_root / "travelsdiarydetail").is_dir():
            cache_root = cache_root / "travelsdiarydetail"
        diary_sources.extend(sorted(cache_root.rglob("*.json")))
    try:
        mora_items = load_diary_action_items(diary_sources) if diary_sources else []
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"旅行札记缓存读取失败：{error}", file=sys.stderr)
        return 2

    report = parse_log_files(sources, mora_items=mora_items)
    if args.format == "text":
        print(render_text(report))
    elif args.format == "html":
        print(report.to_html(
            title=getattr(args, "title", "日志分析"),
            include_faults=not getattr(args, "no_faults", False),
        ), end="")
    else:
        print(report.to_json())
    return 0


def cmd_travel_diary(args) -> int:
    """Refresh HoYoverse travel-diary cache without connecting to a device."""

    from .tasks.travel_diary import (
        MoraStatistics,
        TravelDiaryError,
        TravelDiaryStore,
        TravelDiaryUpdater,
        cookie_from_environment,
        load_today_action_items,
    )

    cookie = cookie_from_environment()
    if not cookie:
        print(
            "未设置 BGI_MIYOUSHE_COOKIE；为避免 Cookie 进入命令历史，"
            "旅行札记命令不接受 --cookie 参数",
            file=sys.stderr,
        )
        return 2
    store = TravelDiaryStore(args.cache_dir)
    updater = TravelDiaryUpdater(
        store=store,
        tz=args.server_timezone,
        log=lambda message: print(message, file=sys.stderr),
    )
    try:
        update = updater.update(cookie, role_index=args.role_index)
        items = load_today_action_items(
            store,
            update.game_info.game_uid,
            tz=args.server_timezone,
        )
    except (TravelDiaryError, IndexError, OSError, ValueError) as error:
        print(f"旅行札记更新失败：{error}", file=sys.stderr)
        return 1

    statistics = MoraStatistics(tuple(items))
    print(json.dumps({
        "gameInfo": update.game_info.to_dict(),
        "updatedMonths": [list(item) for item in update.updated_months],
        "reusedMonths": [list(item) for item in update.reused_months],
        "cacheRoot": str(store.root),
        "today": {
            "items": [item.to_dict() for item in items],
            "eliteStatistics": statistics.elite_statistics,
            "eliteGameStatistics": statistics.elite_game_statistics,
            "eliteMora": statistics.elite_mora,
            "smallMonsterStatistics": statistics.small_monster_statistics,
            "smallMonsterMora": statistics.small_monster_mora,
            "totalMoraKillingMonsters": statistics.total_mora_killing_monsters,
            "otherMora": statistics.other_mora,
            "allMora": statistics.all_mora,
        },
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_combat(args) -> int:
    ctx = _context(args)
    try:
        path = Path(args.file)
        if path.suffix.casefold() == ".json":
            from .tasks.auto_fight import AutoFightTask
            result = AutoFightTask(
                ctx,
                str(path),
                timeout_s=args.timeout,
                party_slots=_load_party(),
            ).run()
            return 0 if result else 1
        from .combat.dsl import CombatExecutor
        executor = CombatExecutor.for_context(ctx, party_slots=_load_party())
        executor.run(path.read_text(encoding="utf-8"))
        return 0
    finally:
        ctx.close()


def cmd_task(args) -> int:
    from .tasks.dispatcher import TaskDispatcher
    from .notification import NotificationService

    if args.config_file:
        config = json.loads(Path(args.config_file).read_text(encoding="utf-8"))
    else:
        config = json.loads(args.config)
    if args.name == "Shell":
        result = TaskDispatcher(None).run_task({"name": args.name, "config": config})
        print(json.dumps({"task": args.name, "result": result}, ensure_ascii=False))
        return 0 if result.get("status") not in {"failed", "timeout"} else 1
    ctx = _context(args)
    notifications = NotificationService.load(args.notification_config)
    try:
        result = TaskDispatcher(
            ctx,
            party_slots=_load_party(),
            notification_service=notifications,
        ).run_task(
            {"name": args.name, "config": config}
        )
        print(json.dumps({"task": args.name, "result": result}, ensure_ascii=False))
        return 0 if result is not False else 1
    finally:
        notifications.close()
        ctx.close()


def cmd_macro(args) -> int:
    from .macro.keymouse import MacroPlayer, load_keymouse
    ctx = _context(args)
    try:
        raw = load_keymouse(args.file)
        if raw.get("format") == "bgi-touch-macro/1":
            print("提示：.touch.json 是预览格式；回放请直接使用原始宏 JSON")
            return 2
        MacroPlayer(ctx.input, sleep=ctx.sleep).play(raw)
        return 0
    finally:
        ctx.close()


def cmd_run(args) -> int:
    from .engine.js_runtime import JsScriptRuntime
    ctx = _context(args)
    overrides = {}
    for kv in args.set or []:
        k, _, v = kv.partition("=")
        overrides[k] = v
    rt = JsScriptRuntime(
        ctx,
        args.script_dir,
        settings=overrides,
        party_slots=_load_party(),
        pathing_root=args.pathing_root,
        notification_config_path=args.notification_config,
    )
    try:
        rt.run()
    finally:
        ctx.close()
    return 0


def cmd_notify(args) -> int:
    """Send one notification without connecting to DeviceHub."""
    from .notification import NotificationService

    service = NotificationService.load(args.notification_config)
    try:
        try:
            queued = service.notify_now(
                args.event,
                args.message,
                result="Fail" if args.error else "Success",
            )
        except Exception as error:
            print(f"Gotify 通知发送失败：{error}", file=sys.stderr)
            return 1
    finally:
        service.close()
    if not queued:
        print("通知未发送：Gotify 未启用、事件未订阅或配置无效", file=sys.stderr)
        return 2
    print("Gotify 通知已发送")
    return 0


def cmd_pathing(args) -> int:
    from .pathing.executor import PathingExecutor
    from .pathing.model import PathingTask
    task = PathingTask.load(args.file)
    task.validate()
    print(json.dumps(task.summary(), ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    ctx = _context(args)
    try:
        return 0 if PathingExecutor(ctx, party_slots=_load_party()).run(task) else 1
    finally:
        ctx.close()


def cmd_group(args) -> int:
    from .tasks.execution_records import ExecutionRecordStore
    from .tasks.script_group import ScriptGroupRoots, ScriptGroupRunner
    from .tasks.task_progress import TaskProgressStore

    roots = ScriptGroupRoots.build(
        javascript=args.script_root,
        key_mouse=args.macro_root,
        pathing=args.pathing_root,
    )
    groups = ScriptGroupRunner.load(
        None,
        args.files,
        roots=roots,
        party_slots=_load_party(),
        progress_store=TaskProgressStore(args.progress_dir),
        execution_store=ExecutionRecordStore(args.records_dir),
        continue_on_error=not args.stop_on_error,
    )
    if args.dry_run:
        print(json.dumps(groups.describe(), ensure_ascii=False, indent=2))
        return 0
    ctx = _context(args)
    groups.ctx = ctx
    try:
        result = groups.run(resume=args.resume)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["failed"] == 0 else 1
    finally:
        ctx.close()


def cmd_trigger(args) -> int:
    """长驻运行实时触发器（Ctrl-C 停止）。"""
    ctx = _context(args)
    if not (args.pick or args.skip or args.eat or args.fish or args.map_mask
            or args.skill_cd or args.quick_teleport):
        print("未指定触发器（--pick / --skip / --eat / --fish / --map-mask / --skill-cd / --quick-teleport）")
        ctx.close()
        return 2
    try:
        if args.pick:
            ctx.enable_trigger("AutoPick")
        if args.skip:
            ctx.enable_trigger("AutoSkip")
        if args.eat:
            ctx.enable_trigger("AutoEat")
        if args.fish:
            ctx.enable_trigger("AutoFish")
        if args.map_mask:
            ctx.enable_trigger("MapMask", map_name=args.map_name)
        if args.skill_cd:
            ctx.enable_trigger("SkillCd", party_slots=_load_party())
        if args.quick_teleport:
            ctx.enable_trigger("QuickTeleport")
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ctx.triggers.stop()
    finally:
        ctx.close()
    return 0


def cmd_reconnect(args) -> int:
    ctx = _context(args)
    ctx.device.reconnect_device()
    print("设备通道已重建")
    ctx.close()
    return 0


def cmd_web(args) -> int:
    from .webui.server import serve
    serve(host=args.host, port=args.port, mcp_url=args.url,
          devicehub_config_path=args.devicehub_config, device_id=args.device_id,
          notification_config_path=args.notification_config)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="bgi-touch",
                                     description="BetterGI 跨平台移植：经 devicehub-mask MCP 自动化 iPhone 端原神")
    parser.add_argument("--url", default=None, help="MCP 地址（优先于配置文件）")
    parser.add_argument("--devicehub-config", default=os.environ.get("BGI_DEVICEHUB_CONFIG"),
                        help="DeviceHub 配置文件（默认 config/devicehub.json）")
    parser.add_argument("--device-id", default=os.environ.get("BGI_DEVICE_ID"),
                        help="精确设备选择 ID/UDID（多设备环境推荐）")
    parser.add_argument(
        "--game-bundle-id",
        default=os.environ.get("BGI_GAME_BUNDLE_ID")
        or os.environ.get("BGI_GENSHIN_BUNDLE_ID"),
        help="原神 iOS Bundle ID（优先于配置与 DeviceHub profile）",
    )
    parser.add_argument("--notification-config", default=os.environ.get("BGI_NOTIFICATION_CONFIG"),
                        help="通知配置文件（默认 config/notification.json）")
    parser.add_argument("--layout", default=os.environ.get("BGI_LAYOUT_PATH"),
                        help="本地触控布局 JSON；支持 config/controls 下的 extends 覆盖")
    parser.add_argument("--keymap-profile", default=os.environ.get(
        "BGI_KEYMAP_PROFILE", "Genshin-Impact-fixed-16by9"),
                        help="DeviceHub profile 名称；传空值或 --no-keymap-profile 禁用")
    parser.add_argument("--keymap-profile-file", default=os.environ.get("BGI_KEYMAP_PROFILE_FILE"),
                        help="从本地 v2 JSON 读取 profile（优先于 MCP）")
    parser.add_argument("--no-keymap-profile", action="store_true",
                        help="禁用 DeviceHub 原生 game session，使用本地触控布局")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="设备与游戏状态")
    p = sub.add_parser("screenshot", help="截图保存")
    p.add_argument("-o", "--output", default="screenshot.png")
    sub.add_parser("launch", help="启动原神")
    sub.add_parser("close-game", help="停止原神；App Store 包不允许强杀时退回 Home 挂起")
    p = sub.add_parser("calibrate", help="输出带布局标注的截图用于校准触控坐标")
    p.add_argument("-o", "--output", default="calibrate.png")
    p = sub.add_parser("convert", help="转换 bettergi-scripts-list 脚本")
    p.add_argument("sources", nargs="+")
    p.add_argument("-o", "--output", default="scripts")
    p = sub.add_parser("log-parse", help="离线分析 BetterGI 日志（不连接设备）")
    p.add_argument("sources", nargs="+", help="日志文件或包含标准日志文件名的目录")
    p.add_argument("--date", help="手工导出的日志日期，格式 YYYY-MM-DD")
    p.add_argument("--format", choices=("json", "text", "html"), default="json")
    p.add_argument("--title", default="日志分析", help="HTML 报告标题")
    p.add_argument("--no-faults", action="store_true", help="HTML 报告隐藏故障列")
    p.add_argument(
        "--diary-file", action="append", default=[],
        help="本地旅行札记缓存 JSON，可重复指定；不发起网络请求",
    )
    p.add_argument(
        "--diary-cache-dir",
        help="旅行札记缓存根目录（配合 --game-uid 读取月度缓存）",
    )
    p.add_argument("--game-uid", help="旅行札记缓存中的原神 UID")
    p = sub.add_parser("travel-diary", help="更新米游社旅行札记缓存（不连接设备）")
    p.add_argument(
        "--cache-dir",
        help="旅行札记缓存根目录（默认 log/logparse）",
    )
    p.add_argument("--role-index", type=int, default=0, help="原神账号角色索引（默认 0）")
    p.add_argument(
        "--server-timezone",
        default="Asia/Shanghai",
        help="旅行札记统计时区名称或小时偏移（默认 Asia/Shanghai）",
    )
    p = sub.add_parser("combat", help="执行战斗策略 .txt/.json")
    p.add_argument("file")
    p.add_argument("--timeout", type=float, default=120, help="JSON 策略超时秒数")
    p = sub.add_parser("task", help="执行 BetterGI SoloTask（含自动吃药/音游/七圣召唤/幽境）")
    p.add_argument("name", help="SoloTask 名称")
    p.add_argument("--config", default="{}", help="内联 JSON 参数")
    p.add_argument("--config-file", help="从 JSON 文件读取参数")
    p = sub.add_parser("macro", help="回放键鼠宏 JSON（自动转触控）")
    p.add_argument("file")
    p = sub.add_parser("run", help="运行 JS 脚本包（BetterGI 兼容）")
    p.add_argument("script_dir")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="覆盖脚本 settings")
    p.add_argument("--pathing-root", help="订阅 Pathing 根目录（默认 scripts/pathing）")
    p = sub.add_parser("notify", help="发送一条 Gotify 测试通知（不连接设备）")
    p.add_argument("message")
    p.add_argument("--event", default="Test")
    p.add_argument("--error", action="store_true", help="标记为失败通知")
    p = sub.add_parser("pathing", help="解析/执行 pathing JSON")
    p.add_argument("file")
    p.add_argument("--dry-run", action="store_true", help="仅解析并输出统计")
    p = sub.add_parser("group", help="执行 BetterGI ScriptGroup 配置组（支持进度续跑）")
    p.add_argument("files", nargs="+", help="User/ScriptGroup 下的配置组 JSON")
    p.add_argument("--script-root", help="Javascript 项目根目录（默认 scripts/js）")
    p.add_argument("--macro-root", help="KeyMouse 项目根目录（默认 scripts/keymouse）")
    p.add_argument("--pathing-root", help="Pathing 项目根目录（默认 scripts/pathing）")
    p.add_argument("--progress-dir", help="TaskProgress 目录（默认 log/task_progress）")
    p.add_argument("--records-dir", help="完成记录目录（默认 log/ExecutionRecords）")
    p.add_argument("--resume", help="进度 JSON 路径或 14 位进度名称")
    p.add_argument("--stop-on-error", action="store_true", help="项目失败时停止配置组")
    p.add_argument("--dry-run", action="store_true", help="仅解析并输出项目顺序")
    p = sub.add_parser("web", help="启动 WebUI 控制台")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8899)
    sub.add_parser("reconnect", help="重建设备通道（触控失效时使用）")
    p = sub.add_parser("trigger", help="长驻实时触发器（含地图遮罩追踪）")
    p.add_argument("--pick", action="store_true", help="自动拾取")
    p.add_argument("--skip", action="store_true", help="自动剧情推进")
    p.add_argument("--eat", action="store_true", help="自动吃药")
    p.add_argument("--fish", action="store_true", help="自动钓鱼（进入钓鱼界面后接管提竿与拉条）")
    p.add_argument("--map-mask", action="store_true", help="地图遮罩与位置追踪")
    p.add_argument("--map-name", default="Teyvat", help="追踪地图名称（默认 Teyvat）")
    p.add_argument("--skill-cd", action="store_true", help="显示四人队伍元素战技冷却")
    p.add_argument("--quick-teleport", action="store_true", help="地图选点后自动确认传送")

    args = parser.parse_args()
    handlers = {"status": cmd_status, "screenshot": cmd_screenshot, "launch": cmd_launch,
                "close-game": cmd_close_game,
                "calibrate": cmd_calibrate, "convert": cmd_convert,
                "log-parse": cmd_log_parse, "travel-diary": cmd_travel_diary,
                "combat": cmd_combat,
                "task": cmd_task,
                "macro": cmd_macro, "run": cmd_run, "pathing": cmd_pathing,
                "group": cmd_group, "notify": cmd_notify, "web": cmd_web,
                "trigger": cmd_trigger, "reconnect": cmd_reconnect}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
