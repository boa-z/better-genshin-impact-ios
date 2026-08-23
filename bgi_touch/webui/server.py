"""WebUI 控制台后端。

单进程 FastAPI：懒加载共享 GameContext（设备调用内部有锁，截图轮询与任务
线程可并发）；同一时刻只允许一个自动化任务在后台线程运行；日志入环形缓冲
供前端轮询。默认只绑定 127.0.0.1。
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response

from ..converter.convert import convert_any
from ..engine.context import GENSHIN_BUNDLE_ID, GameContext

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="bgi-touch console")

# ---- 日志环形缓冲 ----

_logs: deque[dict] = deque(maxlen=800)
_log_seq = 0
_log_lock = threading.Lock()


def weblog(msg: str) -> None:
    global _log_seq
    with _log_lock:
        _log_seq += 1
        _logs.append({"i": _log_seq, "t": datetime.now().strftime("%H:%M:%S"), "msg": str(msg)})
    print(msg)


# ---- 共享 GameContext ----

_ctx: GameContext | None = None
_ctx_lock = threading.Lock()
_mcp_url: str | None = None
_devicehub_config_path: str | Path | None = None
_preview_refresh_lock = threading.Lock()
_preview_refresh_thread: threading.Thread | None = None
_PREVIEW_STALE_S = 1.25


def get_ctx() -> GameContext:
    global _ctx
    with _ctx_lock:
        if _ctx is None:
            weblog("[web] 连接设备…")
            _ctx = GameContext(mcp_url=_mcp_url,
                               devicehub_config_path=_devicehub_config_path)
            weblog(f"[web] 设备已连接 {_ctx.transform.device_width}x{_ctx.transform.device_height}")
        return _ctx


def _err(e: Exception, code: int = 503) -> JSONResponse:
    return JSONResponse({"error": str(e)}, status_code=code)


# ---- 后台任务（同时只跑一个）----

class TaskCancelled(Exception):
    pass


class TaskRunner:
    def __init__(self):
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.info: dict = {"state": "idle"}
        self._js_runtime = None

    def _sleep(self, ms: float) -> None:
        if self.stop_event.wait(ms / 1000):
            raise TaskCancelled()

    def status(self) -> dict:
        return dict(self.info)

    def is_running(self) -> bool:
        with self.lock:
            return bool(self.thread and self.thread.is_alive())

    def start(self, kind: str, path: str, settings: dict | None = None) -> None:
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise RuntimeError("已有任务在运行，请先停止")
            self.stop_event.clear()
            self.info = {"state": "running", "kind": kind, "path": path,
                         "started": datetime.now().strftime("%H:%M:%S")}
            self.thread = threading.Thread(target=self._run, args=(kind, path, settings or {}),
                                           daemon=True, name="bgi-task")
            self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self._js_runtime is not None:
            self._js_runtime.cancelled = True
        if _ctx is not None:
            _ctx.input.release_all()

    def _party(self) -> dict[str, int]:
        p = PROJECT_ROOT / "config" / "party.json"
        if p.exists():
            return {k: int(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
        return {}

    def _run(self, kind: str, path: str, settings: dict) -> None:
        ctx = None
        try:
            ctx = get_ctx()
            weblog(f"[task] 开始 {kind}: {path}")
            if kind == "js":
                from ..engine.js_runtime import JsScriptRuntime
                rt = JsScriptRuntime(ctx, path, settings=settings, log=weblog,
                                     party_slots=self._party())
                self._js_runtime = rt
                rt.run()
            elif kind == "combat":
                from ..combat.dsl import CombatExecutor
                CombatExecutor.for_context(ctx, party_slots=self._party(), log=weblog,
                                           sleep=self._sleep).run(Path(path).read_text(encoding="utf-8"))
            elif kind == "macro":
                from ..macro.keymouse import MacroPlayer, load_keymouse
                MacroPlayer(ctx.input, sleep=self._sleep, log=weblog).play(load_keymouse(path))
            elif kind == "pathing":
                from ..pathing.executor import PathingExecutor
                from ..pathing.model import PathingTask
                PathingExecutor(ctx, party_slots=self._party(), log=weblog).run(PathingTask.load(path))
            elif kind == "task":
                from ..tasks.dispatcher import TaskDispatcher
                task_path = Path(path)
                if task_path.exists():
                    task = json.loads(task_path.read_text(encoding="utf-8"))
                else:
                    task = {"name": path, "config": settings}
                TaskDispatcher(ctx, party_slots=self._party(), log=weblog,
                               cancelled=self.stop_event.is_set).run_task(task)
            else:
                raise ValueError(f"未知任务类型 {kind}")
            self.info = {**self.info, "state": "done", "ended": datetime.now().strftime("%H:%M:%S")}
            weblog(f"[task] 完成 {kind}")
        except (TaskCancelled, Exception) as e:  # noqa: BLE001 —— 任务失败需回报前端
            cancelled = isinstance(e, TaskCancelled) or "ScriptCancelled" in type(e).__name__
            self.info = {**self.info, "state": "cancelled" if cancelled else "error",
                         "error": None if cancelled else str(e),
                         "ended": datetime.now().strftime("%H:%M:%S")}
            weblog(f"[task] {'已停止' if cancelled else f'出错: {e}'}")
        finally:
            self._js_runtime = None
            if ctx is not None:
                ctx.input.release_all()


runner = TaskRunner()


def _capture_is_busy(ctx: GameContext) -> bool:
    """Do not create a competing screenshot producer during automation."""
    loop = getattr(ctx, "_trigger_loop", None)
    return runner.is_running() or bool(loop and loop.active)


def _refresh_preview_in_background(ctx: GameContext) -> None:
    global _preview_refresh_thread
    try:
        if not _capture_is_busy(ctx):
            ctx.capture_bgr()
    except Exception:
        # Preview refresh is opportunistic; the next request can use the last frame
        # or retry once the device is idle.
        pass
    finally:
        with _preview_refresh_lock:
            _preview_refresh_thread = None


def _schedule_preview_refresh(ctx: GameContext) -> None:
    global _preview_refresh_thread
    if _capture_is_busy(ctx):
        return
    with _preview_refresh_lock:
        if _preview_refresh_thread and _preview_refresh_thread.is_alive():
            return
        _preview_refresh_thread = threading.Thread(
            target=_refresh_preview_in_background,
            args=(ctx,),
            daemon=True,
            name="web-preview-refresh",
        )
        _preview_refresh_thread.start()


def _preview_frame(ctx: GameContext) -> tuple[object, float]:
    frame, age = ctx.cached_frame()
    if frame is None:
        # GameContext normally seeds the cache during initialization. Keep a
        # synchronous fallback for direct/test construction.
        return ctx.capture_bgr().copy(), 0.0
    if age > _PREVIEW_STALE_S:
        _schedule_preview_refresh(ctx)
    return frame, age


# ---- 页面与状态 ----

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def api_status():
    try:
        ctx = get_ctx()
        st = ctx.device.status()
        try:
            game = ctx.device.app_status(GENSHIN_BUNDLE_ID)
        except Exception as e:
            game = {"error": str(e)}
        return {"device": st, "game": game, "task": runner.status(),
                "transform": {"w": ctx.transform.device_width, "h": ctx.transform.device_height,
                              "scale": round(ctx.transform.scale, 4)}}
    except Exception as e:
        return _err(e)


@app.get("/api/screenshot")
def api_screenshot(annotate: int = 0, w: int = 1408, q: int = 70):
    try:
        ctx = get_ctx()
        img, frame_age = _preview_frame(ctx)
        if annotate:
            _draw_layout(ctx, img)
        if 0 < w < img.shape[1]:
            h = int(img.shape[0] * w / img.shape[1])
            img = cv2.resize(img, (w, h))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, max(30, min(95, q))])
        if not ok:
            raise RuntimeError("JPEG 编码失败")
        return Response(buf.tobytes(), media_type="image/jpeg",
                        headers={
                            "Cache-Control": "no-store",
                            "X-Screenshot-Frame-Age-Ms": str(round(frame_age * 1000)),
                            "X-Screenshot-Cached": "1",
                        })
    except Exception as e:
        return _err(e)


def _draw_layout(ctx: GameContext, img) -> None:
    h, w = img.shape[:2]
    for name, (nx, ny) in ctx.layout.buttons.items():
        x, y = int(nx * w), int(ny * h)
        cv2.circle(img, (x, y), 14, (0, 0, 255), 2)
        cv2.putText(img, name, (x + 16, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    jx, jy = ctx.layout.joystick_center
    cv2.circle(img, (int(jx * w), int(jy * h)), int(ctx.layout.joystick_radius_n * w), (0, 255, 0), 2)
    nx, ny, nw, nh = ctx.layout.camera_region
    cv2.rectangle(img, (int(nx * w), int(ny * h)), (int((nx + nw) * w), int((ny + nh) * h)), (255, 160, 0), 2)


# ---- 手动控制 ----

@app.post("/api/tap")
def api_tap(body: dict):
    try:
        ctx = get_ctx()
        x = float(body["nx"]) * ctx.transform.device_width
        y = float(body["ny"]) * ctx.transform.device_height
        ctx.device.tap(x, y, image_width=ctx.transform.device_width,
                       image_height=ctx.transform.device_height)
        return {"ok": True}
    except Exception as e:
        return _err(e)


@app.post("/api/swipe")
def api_swipe(body: dict):
    try:
        ctx = get_ctx()
        t = ctx.transform
        ctx.device.swipe(float(body["nx1"]) * t.device_width, float(body["ny1"]) * t.device_height,
                         float(body["nx2"]) * t.device_width, float(body["ny2"]) * t.device_height,
                         duration_ms=int(body.get("duration_ms", 300)),
                         image_width=t.device_width, image_height=t.device_height)
        return {"ok": True}
    except Exception as e:
        return _err(e)


@app.post("/api/key")
def api_key(body: dict):
    try:
        ctx = get_ctx()
        key, action = str(body["key"]), str(body.get("action", "press"))
        if action == "down":
            ctx.input.key_down(key)
        elif action == "up":
            ctx.input.key_up(key)
        else:
            ctx.input.key_press(key)
        return {"ok": True}
    except Exception as e:
        return _err(e)


@app.post("/api/button")
def api_button(body: dict):
    try:
        get_ctx().input.tap_button(str(body["name"]))
        return {"ok": True}
    except Exception as e:
        return _err(e)


@app.post("/api/attack")
def api_attack():
    try:
        get_ctx().input.attack()
        return {"ok": True}
    except Exception as e:
        return _err(e)


@app.post("/api/party")
def api_party(body: dict):
    try:
        get_ctx().input.switch_party_slot(int(body["slot"]))
        return {"ok": True}
    except Exception as e:
        return _err(e)


@app.post("/api/camera")
def api_camera(body: dict):
    try:
        get_ctx().input.move_camera_by(float(body.get("dx", 0)), float(body.get("dy", 0)))
        return {"ok": True}
    except Exception as e:
        return _err(e)


@app.post("/api/launch")
def api_launch():
    try:
        get_ctx().launch_game()
        weblog("[web] 已启动原神")
        return {"ok": True}
    except Exception as e:
        return _err(e)


@app.post("/api/release")
def api_release():
    try:
        get_ctx().input.release_all()
        return {"ok": True}
    except Exception as e:
        return _err(e)


# ---- 脚本管理 ----

@app.get("/api/scripts")
def api_scripts():
    out = {"js": [], "combat": [], "keymouse": [], "pathing": [], "task": []}
    for name in (
        "AutoFight", "AutoWood", "AutoDomain", "AutoCook", "AutoFishing",
        "AutoOpenChest", "AutoBoss", "AutoLeyLine", "AutoLeyLineOutcrop", "AutoEat",
        "AutoMusicGame", "AutoAlbum",
        "AutoGeniusInvokation", "AutoStygianOnslaught", "QuickSereniteaPot",
        "QuickClaimReward", "UseRedemptionCode",
        "AutoArtifactSalvage",
        "CountInventoryItem", "GetGridIcons", "InventoryCountComparison",
    ):
        out["task"].append({"path": name, "name": name})
    js_dir = SCRIPTS_DIR / "js"
    if js_dir.is_dir():
        for d in sorted(js_dir.iterdir()):
            mf = d / "manifest.json"
            if mf.exists():
                try:
                    m = json.loads(mf.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    m = {}
                compat = (d / "COMPAT.md")
                verdict = ""
                if compat.exists():
                    for line in compat.read_text(encoding="utf-8").splitlines():
                        if line.startswith("- 结论："):
                            verdict = line.removeprefix("- 结论：")
                            break
                out["js"].append({"path": str(d), "name": m.get("name", d.name),
                                  "version": m.get("version", ""), "verdict": verdict})
    for kind, pattern in (("combat", "*.txt"), ("keymouse", "*.json"), ("pathing", "*.json")):
        d = SCRIPTS_DIR / kind
        if d.is_dir():
            for f in sorted(d.glob(pattern)):
                if kind == "keymouse" and f.name.endswith(".touch.json"):
                    continue
                out[kind].append({"path": str(f), "name": f.stem})
    return out


@app.get("/api/task")
def api_task():
    return runner.status()


@app.post("/api/run")
def api_run(body: dict):
    try:
        kind, path = str(body["kind"]), str(body["path"])
        target = Path(path).resolve()
        if kind != "task" and not target.exists():
            return _err(FileNotFoundError(f"路径不存在: {path}"), 404)
        settings = body.get("settings") or {}
        runner.start(kind, path if kind == "task" else str(target), settings)
        return {"ok": True}
    except RuntimeError as e:
        return _err(e, 409)
    except Exception as e:
        return _err(e, 400)


@app.post("/api/stop")
def api_stop():
    runner.stop()
    weblog("[web] 请求停止任务")
    return {"ok": True}


@app.post("/api/convert")
def api_convert(body: dict):
    try:
        src = str(body["source"]).strip()
        info = convert_any(src, SCRIPTS_DIR)
        weblog(f"[convert] {src} → {info.get('output')}")
        return info
    except Exception as e:
        return _err(e, 400)


@app.get("/api/triggers")
def api_triggers():
    try:
        ctx = get_ctx()
        loop = ctx.triggers
        return {"active": [t.name for t in loop.triggers]}
    except Exception as e:
        return _err(e)


@app.post("/api/triggers")
def api_triggers_set(body: dict):
    """body: {"AutoPick": true, "AutoSkip": false, "AutoEat": false}"""
    try:
        ctx = get_ctx()
        for name in ("AutoPick", "AutoSkip", "AutoEat"):
            if name not in body:
                continue
            if body[name]:
                ctx.enable_trigger(name)
            else:
                ctx.triggers.remove(name)
                weblog(f"[trigger] 停用 {name}")
        if not ctx.triggers.triggers:
            ctx.triggers.stop()
        return {"active": [t.name for t in ctx.triggers.triggers]}
    except Exception as e:
        return _err(e)


@app.get("/api/logs")
def api_logs(after: int = 0):
    with _log_lock:
        return {"logs": [l for l in _logs if l["i"] > after], "seq": _log_seq}


def serve(
    host: str = "127.0.0.1",
    port: int = 8899,
    *,
    mcp_url: str | None = None,
    devicehub_config_path: str | Path | None = None,
) -> None:
    import uvicorn

    global _mcp_url, _devicehub_config_path
    _mcp_url = mcp_url
    _devicehub_config_path = devicehub_config_path
    weblog(f"[web] 控制台 http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
