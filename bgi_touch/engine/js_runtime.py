"""BetterGI 兼容 JS 脚本运行时（pythonmonkey/SpiderMonkey 宿主）。

实现 bettergi.d.ts 的 API 表面，使 bettergi-scripts-list repo/js 的脚本
尽量原样运行。要点：

- 原版 ClearScript 大小写不敏感绑定（keyPress/KeyPress 混用）：所有宿主
  对象经 JS 侧 Proxy 包装做大小写回退查找。
- 原版的同步 C# 方法在这里是阻塞的 Python 函数；`await sleep()` 等价于
  阻塞后立即 resolve 的 Promise，时序语义保持一致。
- file/http 按 manifest 沙箱化（脚本目录内读写、URL 白名单）。
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from ..macro.keymouse import MacroPlayer, load_keymouse
from ..pathing.executor import PathingExecutor
from ..pathing.model import PathingTask
from .context import GameContext
from .recognition import ImageRegion, Mat, Point2f, RecognitionObject, Region

CASE_PROXY = """
(function (target) {
  if (target === null || typeof target !== 'object' && typeof target !== 'function') return target;
  const cache = {};
  return new Proxy(target, {
    get(t, prop, recv) {
      if (typeof prop !== 'string' || prop in t) return Reflect.get(t, prop, recv);
      if (!(prop in cache)) {
        const lower = prop.toLowerCase();
        cache[prop] = Object.getOwnPropertyNames(t).find(k => k.toLowerCase() === lower)
          ?? (t.constructor ? Object.getOwnPropertyNames(Object.getPrototypeOf(t) ?? {}).find(k => k.toLowerCase() === lower) : undefined)
          ?? null;
      }
      return cache[prop] === null ? undefined : Reflect.get(t, cache[prop], recv);
    }
  });
})
"""

_SCRIPT_ERROR_MARKER = "__BGI_SCRIPT_ERROR__"


def _repair_js_text(text: str) -> str:
    """Repair PythonMonkey's UTF-8 text decoded as Latin-1."""
    value = str(text)
    if not value or any(ord(char) > 255 for char in value):
        return value
    try:
        repaired = value.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return value
    mojibake = "ÃÂÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞß" \
                "äåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ"
    old_score = sum(value.count(char) for char in mojibake)
    new_score = sum(repaired.count(char) for char in mojibake)
    return repaired if new_score < old_score else value


def _script_error_message(value: str) -> str:
    text = _repair_js_text(value)
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), text)
    if "Python RuntimeError: " in first_line:
        return first_line.split("Python RuntimeError: ", 1)[1]
    if first_line.startswith("Error: "):
        return first_line.removeprefix("Error: ")
    return first_line


def _unwrap_async_iife(source: str) -> str:
    """Await the common BetterGI ``(async function () { ... })();`` form."""
    stripped = source.strip()
    start = stripped.find("(async function")
    if start < 0:
        start = stripped.find("(async () =>")
    if start < 0:
        return source
    prefix = stripped[:start]
    if not re.fullmatch(r"(?:\s|//[^\n]*(?:\n|$)|/\*.*?\*/)*", prefix, re.S):
        return source
    end_marker = "})();"
    if not stripped.endswith(end_marker):
        return source
    open_brace = stripped.find("{", start)
    if open_brace < 0 or open_brace >= len(stripped) - len(end_marker):
        return source
    return stripped[open_brace + 1:-len(end_marker)]


class ScriptCancelled(Exception):
    pass


class JsScriptRuntime:
    def __init__(self, ctx: GameContext, script_dir: str | Path,
                 settings: dict | None = None, log: Callable[[str], None] = print,
                 party_slots: dict[str, int] | None = None):
        import pythonmonkey as pm

        self.pm = pm
        self.ctx = ctx
        self.script_dir = Path(script_dir).resolve()
        self.log = log
        self.cancelled = False
        self.manifest = self._load_manifest()
        self.settings = self._load_settings(settings or {})
        self.party_slots = party_slots or {}
        self._install_globals()

    # ---- manifest / settings ----

    def _load_manifest(self) -> dict:
        p = self.script_dir / "manifest.json"
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

    def _load_settings(self, overrides: dict) -> dict:
        values: dict[str, Any] = {}
        ui = self.manifest.get("settings_ui") or self.manifest.get("settingsUi")
        if ui and (self.script_dir / ui).exists():
            try:
                for item in json.loads((self.script_dir / ui).read_text(encoding="utf-8")):
                    if isinstance(item, dict) and item.get("name"):
                        values[item["name"]] = item.get("default")
            except (json.JSONDecodeError, TypeError):
                pass
        user = self.script_dir / "user-settings.json"
        if user.exists():
            values.update(json.loads(user.read_text(encoding="utf-8")))
        values.update(overrides)
        return values

    # ---- sandboxed helpers ----

    def _resolve(self, sub_path: str) -> Path:
        p = (self.script_dir / str(sub_path)).resolve()
        if not p.is_relative_to(self.script_dir):
            raise PermissionError(f"路径越出脚本目录: {sub_path}")
        return p

    def _check_cancel(self) -> None:
        if self.cancelled:
            raise ScriptCancelled()

    # ---- API implementations ----

    def _sleep(self, ms: float) -> None:
        # 分片睡眠，保证 cancelled 置位后能在 ~100ms 内中断长 sleep
        remain = max(0.0, float(ms)) / 1000
        while remain > 0:
            self._check_cancel()
            step = min(0.1, remain)
            time.sleep(step)
            remain -= step
        self._check_cancel()

    def _capture(self) -> Any:
        self._check_cancel()
        return self._wrap(ImageRegion(self.ctx, self.ctx.capture_bgr()))

    def _wrap(self, obj: Any) -> Any:
        """JS 侧大小写不敏感 Proxy 包装。"""
        return self._case_proxy(obj)

    def _http_request(self, method: str, url: str, body: Any = None, headers_json: str = "") -> dict:
        allowed = self.manifest.get("http_allowed_urls") or []
        ok = any(url.startswith(a.rstrip("*")) or a == "https://*" or a == "*" for a in allowed)
        if not ok:
            raise PermissionError(f"URL 未在 manifest http_allowed_urls 中声明: {url}")
        headers = json.loads(headers_json) if headers_json else {}
        data = None
        if body is not None:
            data = (body if isinstance(body, (bytes, str)) else json.dumps(body))
            data = data.encode() if isinstance(data, str) else data
        req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
        with urllib.request.urlopen(req, timeout=30) as resp:
            return {"status_code": resp.status, "headers": dict(resp.headers), "body": resp.read().decode("utf-8", "replace")}

    def _install_globals(self) -> None:
        pm = self.pm
        self._case_proxy = pm.eval(CASE_PROXY)
        g = pm.eval("globalThis")
        ctx, log, wrap = self.ctx, self.log, self._wrap

        def expose(name: str, value: Any, proxy: bool = True) -> None:
            v = wrap(value) if proxy and not callable(value) else value
            g[name] = v
            cap = name[0].upper() + name[1:]
            if cap != name:
                g[cap] = v

        # 基础
        expose("sleep", self._sleep)
        expose("getVersion", lambda: "0.1.0-touch")
        expose("captureGameRegion", self._capture)
        expose("getAvatars", lambda: list(self.party_slots.keys()))
        expose("inputText", lambda text: ctx.device.paste_text(str(text)))
        expose("setGameMetrics", lambda w, h, dpi=1: None)
        expose("getGameMetrics", lambda: [
            ctx.transform.device_width,
            ctx.transform.device_height,
            ctx.transform.scale,
        ])

        # 键鼠
        expose("keyDown", lambda k: ctx.input.key_down(str(k)))
        expose("keyUp", lambda k: ctx.input.key_up(str(k)))
        expose("keyPress", lambda k: ctx.input.key_press(str(k)))
        expose("click", lambda x, y: ctx.input.click_ref(float(x), float(y)))
        expose("moveMouseBy", lambda dx, dy: ctx.input.move_camera_by(float(dx), float(dy)))
        expose("moveMouseTo", lambda x, y: None)  # 无指针；点击时直接给坐标
        expose("leftButtonClick", lambda: ctx.input.attack())
        expose("leftButtonDown", lambda: ctx.input.attack_down())
        expose("leftButtonUp", lambda: ctx.input.attack_up())
        expose("rightButtonClick", lambda: ctx.input.key_press("R"))
        expose("rightButtonDown", lambda: ctx.input.button_down("aim"))
        expose("rightButtonUp", lambda: ctx.input.button_up("aim"))
        expose("middleButtonClick", lambda: None)
        expose("verticalScroll", lambda n: None)

        # log / notification / settings
        class _Log:
            @staticmethod
            def _fmt(msg, *args):
                s = str(msg)
                for a in args:
                    s = __import__("re").sub(r"\{[^}]*\}", str(a), s, count=1)
                return s
            def debug(self, m, *a): log(f"[debug] {_Log._fmt(m, *a)}")
            def info(self, m, *a): log(f"[info] {_Log._fmt(m, *a)}")
            def warn(self, m, *a): log(f"[warn] {_Log._fmt(m, *a)}")
            def error(self, m, *a): log(f"[error] {_Log._fmt(m, *a)}")
        expose("log", wrap(_Log()), proxy=False)

        class _Notification:
            def send(self, msg): log(f"[通知] {msg}")
            def error(self, msg): log(f"[通知-错误] {msg}")
        expose("notification", wrap(_Notification()), proxy=False)
        g["settings"] = pm.eval("(o) => o")(self.settings)

        # file（沙箱）
        rt = self

        class _File:
            def readTextSync(self, p): return rt._resolve(p).read_text(encoding="utf-8")
            def readText(self, p, cb=None):
                text = self.readTextSync(p)
                if cb:
                    cb(text)
                return text
            def writeTextSync(self, p, content, append=False):
                f = rt._resolve(p)
                f.parent.mkdir(parents=True, exist_ok=True)
                mode = "a" if append else "w"
                with open(f, mode, encoding="utf-8") as fh:
                    fh.write(str(content))
                return True
            def writeText(self, p, content, append=False): return self.writeTextSync(p, content, append)
            def readImageMatSync(self, p): return wrap(Mat.from_file(str(rt._resolve(p))))
            def readImageMatWithResizeSync(self, p, w, h, interp=1):
                import cv2 as _cv2
                m = Mat.from_file(str(rt._resolve(p)))
                return wrap(Mat(_cv2.resize(m.bgr, (int(w), int(h)))))
            def readPathSync(self, folder):
                base = rt._resolve(folder)
                return [str(p.relative_to(rt.script_dir)) for p in sorted(base.iterdir())]
            def isFolder(self, p): return rt._resolve(p).is_dir()
            def isFile(self, p): return rt._resolve(p).is_file()
            def isExists(self, p): return rt._resolve(p).exists()
        expose("file", wrap(_File()), proxy=False)

        # http
        class _Http:
            def request(self, method, url, body=None, headers_json=""):
                return rt._http_request(str(method), str(url), body, str(headers_json or ""))
        expose("http", wrap(_Http()), proxy=False)

        # 识别类型
        class _RO:
            @staticmethod
            def templateMatch(mat, x=None, y=None, w=None, h=None):
                m = mat if isinstance(mat, Mat) else getattr(mat, "__wrapped__", mat)
                return wrap(RecognitionObject.template_match(m, x, y, w, h))
            TemplateMatch = templateMatch
            @staticmethod
            def ocr(x, y, w, h): return wrap(RecognitionObject.ocr(x, y, w, h))
            Ocr = ocr
            @staticmethod
            def ocrMatch(x, y, w, h, *texts): return wrap(RecognitionObject.ocr_match(x, y, w, h, *[str(t) for t in texts]))
            OcrMatch = ocrMatch
            @property
            def ocrThis(self): return wrap(RecognitionObject.ocr_this())
            @property
            def OcrThis(self): return wrap(RecognitionObject.ocr_this())
        expose("RecognitionObject", wrap(_RO()), proxy=False)
        expose("Point2f", Point2f, proxy=False)

        # genshin 助手
        from .genshin_api import GenshinApi
        expose("genshin", wrap(GenshinApi(ctx, log)), proxy=False)

        # keyMouseScript / pathingScript
        player = MacroPlayer(ctx.input, sleep=ctx.sleep, log=log)

        class _KeyMouse:
            def run(self, j): player.play(json.loads(str(j)))
            def runFile(self, p): player.play(load_keymouse(rt._resolve(p)))
        expose("keyMouseScript", wrap(_KeyMouse()), proxy=False)

        pathing_exec = PathingExecutor(ctx, party_slots=self.party_slots, log=log)

        class _Pathing:
            def run(self, j): pathing_exec.run(PathingTask.parse(json.loads(str(j))))
            def runFile(self, p): pathing_exec.run(PathingTask.load(rt._resolve(p)))
            def runFileFromUser(self, p): pathing_exec.run(PathingTask.load(rt._resolve(p)))
        expose("pathingScript", wrap(_Pathing()), proxy=False)

        # dispatcher / 任务模型。JS、WebUI、CLI 共用同一个实现，避免任务
        # 名称和参数在不同入口逐渐分叉。
        from ..tasks.dispatcher import TaskDispatcher
        task_dispatcher = TaskDispatcher(
            ctx,
            party_slots=self.party_slots,
            log=log,
            cancelled=lambda: rt.cancelled,
        )

        class _CTS:
            def __init__(self): self._cancelled = False
            def cancel(self): self._cancelled = True
            def isCancellationRequested(self): return self._cancelled
            @property
            def token(self): return self

        class _Dispatcher:
            def runTask(self, task, ct=None):
                return task_dispatcher.run_task(task, ct)

            def runAutoDomainTask(self, param):
                return task_dispatcher.run_auto_domain_task(param)

            def runAutoFightTask(self, param):
                return task_dispatcher.run_auto_fight_task(param)
            def runAutoCookTask(self, param=None): return task_dispatcher.run_auto_cook_task(param)
            def runAutoFishingTask(self, param=None): return task_dispatcher.run_auto_fishing_task(param)
            def runAutoOpenChestTask(self, param=None): return task_dispatcher.run_auto_open_chest_task(param)
            def runAutoEatTask(self, param=None, ct=None):
                return task_dispatcher.run_auto_eat_task(param, ct)
            def runAutoMusicGameTask(self, param=None, ct=None):
                return task_dispatcher.run_auto_music_game_task(param, ct)
            def runAutoAlbumTask(self, param=None, ct=None):
                return task_dispatcher.run_auto_album_task(param, ct)
            def runAutoAlbum(self, param=None, ct=None):
                return task_dispatcher.run_auto_album_task(param, ct)
            def runAutoGeniusInvokationTask(self, param=None, ct=None):
                return task_dispatcher.run_auto_genius_invokation_task(param, ct)
            def runAutoStygianOnslaughtTask(self, param=None, ct=None):
                return task_dispatcher.run_auto_stygian_onslaught_task(param, ct)
            def runAutoBossTask(self, param=None): return task_dispatcher.run_auto_boss_task(param)
            def runAutoLeyLineTask(self, param=None): return task_dispatcher.run_auto_leyline_task(param)
            def runAutoLeyLineOutcropTask(self, param=None): return task_dispatcher.run_auto_leyline_task(param)
            def runQuickSereniteaPotTask(self, param=None, ct=None):
                return task_dispatcher.run_quick_serenitea_pot_task(param, ct)
            def runQuickClaimRewardTask(self, param=None, ct=None):
                return task_dispatcher.run_quick_claim_reward_task(param, ct)
            def runUseRedemptionCodeTask(self, param=None, ct=None):
                return task_dispatcher.run_use_redemption_code_task(param, ct)
            def runAutoArtifactSalvageTask(self, param=None, ct=None):
                return task_dispatcher.run_auto_artifact_salvage_task(param, ct)
            def runCombatScript(self, script, avatar=None):
                return task_dispatcher.run_combat_script(str(script), avatar)
            def addTimer(self, timer):
                try:
                    task_dispatcher.add_timer(timer)
                except ValueError as e:
                    log(f"[dispatcher] {e}")
            def addTrigger(self, trigger):
                try:
                    task_dispatcher.add_trigger(trigger)
                except ValueError as e:
                    log(f"[dispatcher] {e}")
            def clearAllTriggers(self):
                task_dispatcher.clear_all_triggers()
            def getLinkedCancellationTokenSource(self): return wrap(_CTS())
            def getLinkedCancellationToken(self): return wrap(_CTS())
        expose("dispatcher", wrap(_Dispatcher()), proxy=False)

        # 构造器类：脚本里 new RealtimeTimer("AutoPick") / new SoloTask("AutoFight")
        g["RealtimeTimer"] = pm.eval("(function(){ return function RealtimeTimer(name, cfg){ this.name = name; this.config = cfg; }; })()")
        g["SoloTask"] = pm.eval("(function(){ return function SoloTask(name, cfg){ this.name = name; this.config = cfg; }; })()")
        g["CancellationTokenSource"] = pm.eval("(function(){ return function CancellationTokenSource(){ this.cancelled = false; this.cancel = () => { this.cancelled = true; }; this.token = this; }; })()")

    # ---- run ----

    def run(self, entry: str | None = None) -> Any:
        main = entry or self.manifest.get("main") or "main.js"
        code = _unwrap_async_iife((self.script_dir / main).read_text(encoding="utf-8"))
        import asyncio

        async def _drive():
            # pythonmonkey 的 Promise/await 机制要求存在运行中的 asyncio 循环
            wrapper = (
                "(async () => {"
                "try {"
                f"{code}\n"
                "} catch (error) {"
                f"return {json.dumps(_SCRIPT_ERROR_MARKER)} + JSON.stringify({{"
                "message: String(error),"
                "stack: error && error.stack ? String(error.stack) : ''"
                "});"
                "}"
                "})()"
            )
            result = await self.pm.eval(wrapper)
            if isinstance(result, str) and result.startswith(_SCRIPT_ERROR_MARKER):
                raw = result[len(_SCRIPT_ERROR_MARKER):]
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    raise RuntimeError(_script_error_message(raw)) from None
                raw_message = str(payload.get("message", raw))
                if "ScriptCancelled" in raw_message:
                    raise ScriptCancelled() from None
                raise RuntimeError(_script_error_message(raw_message)) from None
            return result

        try:
            return asyncio.run(_drive())
        except ScriptCancelled:
            self.log("[runtime] 脚本已取消")
            return None
