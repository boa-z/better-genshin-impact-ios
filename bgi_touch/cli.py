"""bgi-touch 命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_party() -> dict[str, int]:
    p = PROJECT_ROOT / "config" / "party.json"
    if p.exists():
        return {k: int(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
    return {}


def cmd_status(args) -> int:
    from .engine.context import GENSHIN_BUNDLE_ID, GameContext
    ctx = GameContext(mcp_url=args.url)
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
    from .engine.context import GameContext
    ctx = GameContext(mcp_url=args.url)
    png = ctx.device.screenshot_png()
    out = Path(args.output)
    out.write_bytes(png)
    print(f"已保存 {out}（{len(png)} 字节）")
    ctx.close()
    return 0


def cmd_launch(args) -> int:
    from .engine.context import GameContext
    ctx = GameContext(mcp_url=args.url)
    ctx.launch_game()
    print("原神已启动")
    ctx.close()
    return 0


def cmd_calibrate(args) -> int:
    """截图并叠加当前布局的按钮标注，输出到文件用于人工校准。"""
    import cv2
    from .engine.context import GameContext
    ctx = GameContext(mcp_url=args.url)
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
    from .engine.context import GameContext
    ctx = GameContext(mcp_url=args.url)
    executor = CombatExecutor(ctx.input, sleep=ctx.sleep, party_slots=_load_party())
    executor.run(Path(args.file).read_text(encoding="utf-8"))
    ctx.close()
    return 0


def cmd_macro(args) -> int:
    from .engine.context import GameContext
    from .macro.keymouse import MacroPlayer, load_keymouse
    ctx = GameContext(mcp_url=args.url)
    raw = load_keymouse(args.file)
    if raw.get("format") == "bgi-touch-macro/1":
        print("提示：.touch.json 是预览格式；回放请直接使用原始宏 JSON")
        return 2
    MacroPlayer(ctx.input, sleep=ctx.sleep).play(raw)
    ctx.close()
    return 0


def cmd_run(args) -> int:
    from .engine.context import GameContext
    from .engine.js_runtime import JsScriptRuntime
    ctx = GameContext(mcp_url=args.url)
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
    from .engine.context import GameContext
    from .pathing.executor import PathingExecutor
    from .pathing.model import PathingTask
    task = PathingTask.load(args.file)
    print(json.dumps(task.summary(), ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    ctx = GameContext(mcp_url=args.url)
    PathingExecutor(ctx, party_slots=_load_party()).run(task)
    ctx.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="bgi-touch",
                                     description="BetterGI 跨平台移植：经 devicehub-mask MCP 自动化 iPhone 端原神")
    parser.add_argument("--url", default=None, help="MCP 地址（默认 http://127.0.0.1:8009/mcp）")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="设备与游戏状态")
    p = sub.add_parser("screenshot", help="截图保存")
    p.add_argument("-o", "--output", default="screenshot.png")
    sub.add_parser("launch", help="启动原神")
    p = sub.add_parser("calibrate", help="输出带布局标注的截图用于校准触控坐标")
    p.add_argument("-o", "--output", default="calibrate.png")
    p = sub.add_parser("convert", help="转换 bettergi-scripts-list 脚本")
    p.add_argument("sources", nargs="+")
    p.add_argument("-o", "--output", default="scripts")
    p = sub.add_parser("combat", help="执行战斗策略 .txt")
    p.add_argument("file")
    p = sub.add_parser("macro", help="回放键鼠宏 JSON（自动转触控）")
    p.add_argument("file")
    p = sub.add_parser("run", help="运行 JS 脚本包（BetterGI 兼容）")
    p.add_argument("script_dir")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="覆盖脚本 settings")
    p = sub.add_parser("pathing", help="解析/执行 pathing JSON")
    p.add_argument("file")
    p.add_argument("--dry-run", action="store_true", help="仅解析并输出统计")

    args = parser.parse_args()
    if args.url is None:
        import os
        args.url = os.environ.get("BGI_MCP_URL", "http://127.0.0.1:8009/mcp")
    handlers = {"status": cmd_status, "screenshot": cmd_screenshot, "launch": cmd_launch,
                "calibrate": cmd_calibrate, "convert": cmd_convert, "combat": cmd_combat,
                "macro": cmd_macro, "run": cmd_run, "pathing": cmd_pathing}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
