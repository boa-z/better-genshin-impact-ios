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
    )


def _load_party() -> dict[str, int]:
    p = PROJECT_ROOT / "config" / "party.json"
    if p.exists():
        return {k: int(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
    return {}


def cmd_status(args) -> int:
    from .engine.context import GENSHIN_BUNDLE_ID
    ctx = _context(args)
    st = ctx.device.status()
    try:
        app = ctx.device.app_status(GENSHIN_BUNDLE_ID)
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
    from .engine.context import GENSHIN_BUNDLE_ID
    ctx = _context(args)
    try:
        ctx.device.stop_app(GENSHIN_BUNDLE_ID)
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


def cmd_combat(args) -> int:
    from .combat.dsl import CombatExecutor
    ctx = _context(args)
    executor = CombatExecutor.for_context(ctx, party_slots=_load_party())
    executor.run(Path(args.file).read_text(encoding="utf-8"))
    ctx.close()
    return 0


def cmd_task(args) -> int:
    from .tasks.dispatcher import TaskDispatcher

    if args.config_file:
        config = json.loads(Path(args.config_file).read_text(encoding="utf-8"))
    else:
        config = json.loads(args.config)
    ctx = _context(args)
    try:
        result = TaskDispatcher(ctx, party_slots=_load_party()).run_task(
            {"name": args.name, "config": config}
        )
        print(json.dumps({"task": args.name, "result": result}, ensure_ascii=False))
        return 0 if result is not False else 1
    finally:
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
    rt = JsScriptRuntime(ctx, args.script_dir, settings=overrides, party_slots=_load_party())
    try:
        rt.run()
    finally:
        ctx.close()
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


def cmd_trigger(args) -> int:
    """长驻运行实时触发器（Ctrl-C 停止）。"""
    ctx = _context(args)
    if args.pick:
        ctx.enable_trigger("AutoPick")
    if args.skip:
        ctx.enable_trigger("AutoSkip")
    if not (args.pick or args.skip):
        print("未指定触发器（--pick / --skip）")
        return 2
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ctx.triggers.stop()
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
    serve(host=args.host, port=args.port)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="bgi-touch",
                                     description="BetterGI 跨平台移植：经 devicehub-mask MCP 自动化 iPhone 端原神")
    parser.add_argument("--url", default=None, help="MCP 地址（默认 http://127.0.0.1:8009/mcp）")
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
    p = sub.add_parser("combat", help="执行战斗策略 .txt")
    p.add_argument("file")
    p = sub.add_parser("task", help="执行 BetterGI SoloTask（AutoFight/AutoWood/AutoDomain/AutoCook/AutoFishing/AutoOpenChest）")
    p.add_argument("name", help="SoloTask 名称")
    p.add_argument("--config", default="{}", help="内联 JSON 参数")
    p.add_argument("--config-file", help="从 JSON 文件读取参数")
    p = sub.add_parser("macro", help="回放键鼠宏 JSON（自动转触控）")
    p.add_argument("file")
    p = sub.add_parser("run", help="运行 JS 脚本包（BetterGI 兼容）")
    p.add_argument("script_dir")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="覆盖脚本 settings")
    p = sub.add_parser("pathing", help="解析/执行 pathing JSON")
    p.add_argument("file")
    p.add_argument("--dry-run", action="store_true", help="仅解析并输出统计")
    p = sub.add_parser("web", help="启动 WebUI 控制台")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8899)
    sub.add_parser("reconnect", help="重建设备通道（触控失效时使用）")
    p = sub.add_parser("trigger", help="长驻实时触发器（自动拾取/自动剧情）")
    p.add_argument("--pick", action="store_true", help="自动拾取")
    p.add_argument("--skip", action="store_true", help="自动剧情推进")

    args = parser.parse_args()
    if args.url is None:
        args.url = os.environ.get("BGI_MCP_URL", "http://127.0.0.1:8009/mcp")
    handlers = {"status": cmd_status, "screenshot": cmd_screenshot, "launch": cmd_launch,
                "close-game": cmd_close_game,
                "calibrate": cmd_calibrate, "convert": cmd_convert, "combat": cmd_combat,
                "task": cmd_task,
                "macro": cmd_macro, "run": cmd_run, "pathing": cmd_pathing, "web": cmd_web,
                "trigger": cmd_trigger, "reconnect": cmd_reconnect}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
