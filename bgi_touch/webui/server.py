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
from urllib.parse import quote

import cv2
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response

from ..converter.convert import convert_any
from ..engine.context import GENSHIN_BUNDLE_ID, GameContext
from ..engine.html_mask import html_mask_manager

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


def _shutdown_context() -> None:
    """Release automation input, the MCP session, and owned headless process."""
    global _ctx
    runner.stop()
    with _ctx_lock:
        ctx = _ctx
        _ctx = None
    if ctx is not None:
        ctx.close()


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

    def enqueue_key_mouse_event(self, event: dict) -> bool:
        runtime = self._js_runtime
        if runtime is None:
            return False
        return bool(runtime.enqueue_key_mouse_event(event))

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
    if _ctx is None:
        return {
            "connected": False,
            "device": {"status": "disconnected", "orientation": "unknown"},
            "game": {"status": "unknown"},
            "task": runner.status(),
            "transform": {"w": 0, "h": 0, "scale": 0},
        }
    try:
        ctx = _ctx
        st = ctx.device.status()
        try:
            game = ctx.device.app_status(GENSHIN_BUNDLE_ID)
        except Exception as e:
            game = {"error": str(e)}
        return {"connected": True, "device": st, "game": game, "task": runner.status(),
                "transform": {"w": ctx.transform.device_width, "h": ctx.transform.device_height,
                              "scale": round(ctx.transform.scale, 4)}}
    except Exception as e:
        return _err(e)


@app.get("/api/screenshot")
def api_screenshot(annotate: int = 0, w: int = 1408, q: int = 70):
    if _ctx is None:
        return _err(RuntimeError("设备尚未连接，请先点击“连接设备”"), 409)
    try:
        ctx = _ctx
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

@app.post("/api/connect")
def api_connect():
    try:
        ctx = get_ctx()
        return {
            "ok": True,
            "w": ctx.transform.device_width,
            "h": ctx.transform.device_height,
        }
    except Exception as e:
        return _err(e)

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
        "QuickClaimReward", "QuickBuy", "UseRedemptionCode",
        "AutoArtifactSalvage",
        "CountInventoryItem", "GetGridIcons", "InventoryCountComparison",
        "CharacterDevelopment",
        "OneDragon",
        "Shell",
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


@app.post("/api/key-mouse-hook/event")
def api_key_mouse_hook_event(body: dict):
    """Forward WebUI controls to BetterGI KeyMouseHook without device I/O."""
    try:
        return {"ok": True, "accepted": runner.enqueue_key_mouse_event(body)}
    except (TypeError, ValueError) as error:
        return _err(error, 400)


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
    if _ctx is None:
        return {"active": [], "connected": False}
    try:
        ctx = _ctx
        loop = ctx.triggers
        return {"active": [t.name for t in loop.triggers]}
    except Exception as e:
        return _err(e)


@app.post("/api/triggers")
def api_triggers_set(body: dict):
    """Toggle realtime triggers without creating extra capture producers."""
    try:
        ctx = get_ctx()
        for name in ("AutoPick", "AutoSkip", "AutoEat", "SkillCd"):
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


# ---- BetterGI 技能冷却提示 ----

def _skill_cd_trigger():
    if _ctx is None:
        return None
    loop = getattr(_ctx, "_trigger_loop", None)
    return None if loop is None else loop.get("SkillCd")


def _inactive_skill_cd_state() -> dict:
    return {
        "active": False,
        "scene": "disabled",
        "visible": False,
        "activeSlot": 1,
        "team": [],
        "config": {
            "pX": 1520.0,
            "pY": 245.0,
            "gap": 91.2,
            "scale": 1.0,
            "backgroundNormalColor": "#FFFFFFFF",
            "textNormalColor": "#DA4A23FF",
            "backgroundReadyColor": "#FFFFFFFF",
            "textReadyColor": "#5DCC17FF",
            "hideWhenZero": False,
            "triggerOnSkillUse": False,
        },
        "sequence": 0,
        "ageMs": 0,
    }


@app.get("/api/skill-cd")
def api_skill_cd():
    """Return the cached cooldown snapshot without connecting or capturing."""

    trigger = _skill_cd_trigger()
    return _inactive_skill_cd_state() if trigger is None else trigger.state.snapshot()


@app.post("/api/skill-cd")
def api_skill_cd_set(body: dict):
    try:
        enabled = bool(body.get("enabled", True))
        if not enabled:
            if _ctx is not None:
                loop = getattr(_ctx, "_trigger_loop", None)
                if loop is not None:
                    loop.remove("SkillCd")
                    if not loop.triggers:
                        loop.stop()
            return _inactive_skill_cd_state()

        ctx = get_ctx()
        ctx.enable_trigger(
            "SkillCd",
            party_slots=body.get("partySlots"),
            custom_cd_list=body.get("customCdList", []),
            trigger_on_skill_use=bool(body.get("triggerOnSkillUse", False)),
            hide_when_zero=bool(body.get("hideWhenZero", False)),
            p_x=body.get("pX", 1520.0),
            p_y=body.get("pY", 245.0),
            gap=body.get("gap", 91.2),
            scale=body.get("scale", 1.0),
            background_normal_color=body.get("backgroundNormalColor", "#FFFFFFFF"),
            text_normal_color=body.get("textNormalColor", "#DA4A23FF"),
            background_ready_color=body.get("backgroundReadyColor", "#FFFFFFFF"),
            text_ready_color=body.get("textReadyColor", "#5DCC17FF"),
        )
        trigger = ctx.triggers.get("SkillCd")
        return trigger.state.snapshot() if trigger is not None else _inactive_skill_cd_state()
    except (TypeError, ValueError) as error:
        return _err(error, 400)
    except Exception as error:
        return _err(error)


# ---- BetterGI 地图遮罩 / 地图追踪 ----

def _map_mask_trigger():
    if _ctx is None:
        return None
    loop = getattr(_ctx, "_trigger_loop", None)
    return None if loop is None else loop.get("MapMask")


def _inactive_map_mask_state() -> dict:
    return {
        "active": False,
        "mapName": "Teyvat",
        "layer": 0,
        "scene": "disabled",
        "inBigMapUi": False,
        "positionValid": False,
        "position": None,
        "viewport": None,
        "error": None,
        "sequence": 0,
        "ageMs": 0,
    }


@app.get("/api/map-mask")
def api_map_mask():
    """Return cached tracking state without connecting or capturing a frame."""

    trigger = _map_mask_trigger()
    return _inactive_map_mask_state() if trigger is None else trigger.state.snapshot()


@app.post("/api/map-mask")
def api_map_mask_set(body: dict):
    try:
        enabled = bool(body.get("enabled", True))
        if not enabled:
            if _ctx is not None:
                loop = getattr(_ctx, "_trigger_loop", None)
                if loop is not None:
                    loop.remove("MapMask")
                    if not loop.triggers:
                        loop.stop()
            return _inactive_map_mask_state()

        ctx = get_ctx()
        ctx.enable_trigger(
            "MapMask",
            map_name=str(body.get("mapName", "Teyvat") or "Teyvat"),
            mini_map_enabled=bool(body.get("miniMapMaskEnabled", True)),
        )
        trigger = ctx.triggers.get("MapMask")
        return trigger.state.snapshot() if trigger is not None else _inactive_map_mask_state()
    except ValueError as error:
        return _err(error, 400)
    except Exception as error:
        return _err(error)


@app.get("/api/map-mask/maps")
def api_map_mask_maps():
    from ..triggers.map_mask import map_catalog

    try:
        return {"maps": map_catalog()}
    except Exception as error:
        return _err(error)


@app.get("/api/map-mask/maps/{map_name}/layers/{layer}/image")
def api_map_mask_image(map_name: str, layer: int):
    from ..triggers.map_mask import map_image_path

    try:
        path = map_image_path(map_name, layer)
        if path is None:
            return _err(FileNotFoundError(f"地图图像不存在: {map_name} layer {layer}"), 404)
        return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})
    except ValueError as error:
        return _err(error, 404)
    except Exception as error:
        return _err(error)


@app.get("/api/logs")
def api_logs(after: int = 0):
    with _log_lock:
        return {"logs": [l for l in _logs if l["i"] > after], "seq": _log_seq}


# ---- BetterGI HTML 遮罩 ----

def _mask_bridge(window_id: str) -> str:
    identifier = json.dumps(str(window_id), ensure_ascii=False)
    return f"""
<script>
(() => {{
  const windowId = {identifier};
  const callbacks = new Map();
  let sequence = 0;
  function post(url, data, requestId) {{
    parent.postMessage({{type:'bgi-html-mask', windowId, url, data, requestId}}, location.origin);
  }}
  function postKey(type, event) {{
    parent.postMessage({{type:'bgi-key-mouse-hook', event:{{
      type, key:event.key, code:event.code, repeat:event.repeat,
      altKey:event.altKey, ctrlKey:event.ctrlKey,
      shiftKey:event.shiftKey, metaKey:event.metaKey
    }}}}, location.origin);
  }}
  window.addEventListener('keydown', event => postKey('keyDown', event), true);
  window.addEventListener('keyup', event => postKey('keyUp', event), true);
  window.htmlMask = {{
    onMessage: null,
    send(url, data) {{ post(url, data ?? {{}}, null); }},
    request(url, data) {{
      const requestId = '__req_' + (++sequence);
      return new Promise((resolve, reject) => {{
        callbacks.set(requestId, {{resolve, reject}});
        post(url, data ?? {{}}, requestId);
      }});
    }},
    _dispatch(message) {{
      const callback = message.requestId && callbacks.get(message.requestId);
      if (callback) {{
        callbacks.delete(message.requestId);
        callback.resolve(message);
        return;
      }}
      if (typeof this.onMessage !== 'function') return;
      const result = this.onMessage(message);
      if (message.requestId && result !== undefined) {{
        Promise.resolve(result).then(data =>
          post('/__response__', data ?? null, message.requestId));
      }}
    }}
  }};
  window.addEventListener('message', event => {{
    if (event.origin !== location.origin) return;
    const envelope = event.data || {{}};
    if (envelope.type === 'bgi-html-mask-host' && envelope.windowId === windowId) {{
      window.htmlMask._dispatch(envelope.message);
    }}
  }});
}})();
</script>
"""


@app.get("/api/html-masks/events")
def api_html_mask_events(after: int = 0):
    return html_mask_manager.events(after=after, timeout_s=20)


@app.post("/api/html-masks/{window_id}/message")
def api_html_mask_message(window_id: str, body: dict):
    try:
        html_mask_manager.post_from_html(
            window_id,
            str(body.get("url", "")),
            body.get("data"),
            str(body["requestId"]) if body.get("requestId") else None,
        )
        return {"ok": True}
    except Exception as e:
        return _err(e, 404)


@app.post("/api/html-masks/{window_id}/close")
def api_html_mask_close(window_id: str):
    return {"ok": html_mask_manager.close(window_id)}


@app.get("/api/html-masks/{window_id}/files/{asset_path:path}")
def api_html_mask_file(window_id: str, asset_path: str):
    try:
        source = html_mask_manager.resolve_asset(window_id, asset_path)
        if source.suffix.lower() not in {".html", ".htm"}:
            return FileResponse(source)
        content = source.read_text(encoding="utf-8", errors="replace")
        parent = Path(asset_path).parent
        encoded_parent = "/".join(quote(part, safe="") for part in parent.parts)
        base_url = f"/api/html-masks/{quote(window_id, safe='')}/files/"
        if encoded_parent and encoded_parent != ".":
            base_url += encoded_parent.rstrip("/") + "/"
        injection = f'<base href="{base_url}">' + _mask_bridge(window_id)
        lowered = content.lower()
        head_at = lowered.find("<head")
        head_end = content.find(">", head_at) + 1 if head_at >= 0 else 0
        content = (
            content[:head_end] + injection + content[head_end:]
            if head_end > 0 else injection + content
        )
        return Response(content, media_type="text/html",
                        headers={"Cache-Control": "no-store"})
    except PermissionError as e:
        return _err(e, 403)
    except FileNotFoundError as e:
        return _err(e, 404)
    except Exception as e:
        return _err(e)


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
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        _shutdown_context()
