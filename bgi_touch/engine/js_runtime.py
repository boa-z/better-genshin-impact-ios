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
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ..macro.keymouse import MacroPlayer, load_keymouse
from ..pathing.executor import PathingExecutor
from ..pathing.model import PathingTask
from .context import GameContext
from .keymouse_hook import KeyMouseHookManager
from .recognition import (
    DesktopRegion,
    GameCaptureRegion,
    ImageRegion,
    Mat,
    Point2f,
    RecognitionObject,
    Region,
    Scalar,
    SearchAnchorMode,
    SearchExpandRatio,
    SearchOptions,
)

CASE_PROXY = """
(function (target) {
  if (target === null || typeof target !== 'object' && typeof target !== 'function') return target;
  const cache = {};
  return new Proxy(target, {
    get(t, prop, recv) {
      if (prop === '__wrapped__') return t;
      if (typeof prop !== 'string' || prop in t) return Reflect.get(t, prop, recv);
      if (!(prop in cache)) {
        const lower = prop.toLowerCase();
        cache[prop] = Object.getOwnPropertyNames(t).find(k => k.toLowerCase() === lower)
          ?? (t.constructor ? Object.getOwnPropertyNames(Object.getPrototypeOf(t) ?? {}).find(k => k.toLowerCase() === lower) : undefined)
          ?? null;
      }
      return cache[prop] === null ? undefined : Reflect.get(t, cache[prop], recv);
    },
    set(t, prop, value, recv) {
      if (typeof prop !== 'string' || prop in t) return Reflect.set(t, prop, value, recv);
      const lower = prop.toLowerCase();
      const key = Object.getOwnPropertyNames(t).find(k => k.toLowerCase() === lower)
        ?? Object.getOwnPropertyNames(Object.getPrototypeOf(t) ?? {}).find(k => k.toLowerCase() === lower)
        ?? prop;
      return Reflect.set(t, key, value, recv);
    }
  });
})
"""

_SCRIPT_ERROR_MARKER = "__BGI_SCRIPT_ERROR__"
BETTERGI_COMPAT_VERSION = "0.63.2-alpha.4"


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
                 party_slots: dict[str, int] | None = None,
                 strategy_roots: list[str | Path] | None = None,
                 pathing_root: str | Path | None = None,
                 notification_config_path: str | Path | None = None):
        import pythonmonkey as pm

        self.pm = pm
        self.ctx = ctx
        self.script_dir = Path(script_dir).resolve()
        self.log = log
        self.cancelled = False
        self.manifest = self._load_manifest()
        self.settings = self._load_settings(settings or {})
        if party_slots is not None:
            self.party_slots = party_slots
        else:
            cached_party = getattr(ctx, "party_slots", None)
            self.party_slots = cached_party if isinstance(cached_party, dict) else {}
        default_strategy_root = Path(__file__).resolve().parents[2] / "scripts" / "combat"
        self.strategy_roots = [
            Path(value).expanduser().resolve()
            for value in (strategy_roots or [default_strategy_root])
        ]
        default_pathing_root = Path(__file__).resolve().parents[2] / "scripts" / "pathing"
        self.pathing_root = Path(
            pathing_root or default_pathing_root
        ).expanduser().resolve()
        from ..notification import NotificationService

        self._notification_service = NotificationService.load(
            notification_config_path, log=log,
        )
        self._key_mouse_hooks = KeyMouseHookManager(log=log)
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
        # Community scripts are authored for Windows and often keep `\\` in
        # manifest/file paths. Treat it as a separator on macOS as well.
        relative = str(sub_path).replace("\\", "/")
        p = (self.script_dir / relative).resolve()
        if not p.is_relative_to(self.script_dir):
            raise PermissionError(f"路径越出脚本目录: {sub_path}")
        return p

    def _resolve_pathing(self, sub_path: str) -> Path:
        """Resolve a BetterGI User/AutoPathing path inside the shared root."""
        relative = str(sub_path or ".").replace("\\", "/")
        p = (self.pathing_root / relative).resolve()
        if not p.is_relative_to(self.pathing_root):
            raise PermissionError(f"路径越出 Pathing 根目录: {sub_path}")
        return p

    def _check_cancel(self) -> None:
        if self.cancelled:
            raise ScriptCancelled()

    # ---- API implementations ----

    def _sleep(self, ms: float) -> None:
        # 分片睡眠，保证 cancelled 置位后能在 ~100ms 内中断长 sleep
        remain = max(0.0, float(ms)) / 1000
        self._drain_key_mouse_events()
        while remain > 0:
            self._check_cancel()
            step = min(0.1, remain)
            time.sleep(step)
            remain -= step
            self._drain_key_mouse_events()
        self._check_cancel()

    def _drain_key_mouse_events(self) -> int:
        manager = getattr(self, "_key_mouse_hooks", None)
        return manager.drain() if manager is not None else 0

    def enqueue_key_mouse_event(self, event: dict[str, Any]) -> bool:
        """Queue a WebUI control event without touching PythonMonkey threads."""
        value = dict(event)
        if "nx" in value and "ny" in value:
            transform = self.ctx.transform
            dev_x = float(value.pop("nx")) * transform.device_width
            dev_y = float(value.pop("ny")) * transform.device_height
            value["x"], value["y"] = transform.to_ref(dev_x, dev_y)
        return self._key_mouse_hooks.enqueue(value)

    def has_key_mouse_hooks(self) -> bool:
        return self._key_mouse_hooks.has_hooks()

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

    def _ensure_js_timers(self, global_object: Any) -> None:
        """Keep BetterGI's timer globals available in every JS runtime.

        PythonMonkey ships the timer implementation as a built-in module and
        normally installs it while importing ``pythonmonkey``.  Do the
        installation explicitly here as well: some supported PythonMonkey
        builds expose ``timers`` without eagerly copying its exports onto
        ``globalThis``.  Using the built-in implementation is important—the
        callbacks are scheduled by SpiderMonkey's own event loop and therefore
        never invoke a JS function from a Python worker thread.
        """
        names = (
            "setTimeout", "clearTimeout", "setImmediate", "clearImmediate",
            "setInterval", "clearInterval",
        )
        try:
            timers = self.pm.require("timers")
            for name in names:
                if self.pm.eval(
                    f"typeof globalThis[{json.dumps(name)}]"
                ) == "function":
                    continue
                global_object[name] = timers[name]
        except Exception as error:  # pragma: no cover - depends on the JS build
            self.log(f"[runtime] 无法安装 JS 计时器兼容：{error}")

    def _install_globals(self) -> None:
        pm = self.pm
        self._case_proxy = pm.eval(CASE_PROXY)
        g = pm.eval("globalThis")
        self._ensure_js_timers(g)
        ctx, log, wrap = self.ctx, self.log, self._wrap

        from ..input.layout import normalize_key
        from ..input.pointer import TouchPointer
        pointer = TouchPointer(ctx.input)
        self._pointer = pointer
        setattr(ctx, "_script_pointer", pointer)

        def expose(name: str, value: Any, proxy: bool = True) -> None:
            v = wrap(value) if proxy and not callable(value) else value
            g[name] = v
            cap = name[0].upper() + name[1:]
            if cap != name:
                g[cap] = v

        # 基础
        expose("sleep", self._sleep)
        expose("getVersion", lambda: BETTERGI_COMPAT_VERSION)
        expose("captureGameRegion", self._capture)

        def _get_avatars() -> list[str]:
            """Return the current mobile party without spawning a new stream.

            BetterGI's desktop implementation recognizes the party from the
            same capture that backs ``CaptureGameRegion``.  The iOS runtime
            must not make every ``getAvatars()`` call another DeviceHub
            screenshot request: community scripts commonly call it several
            times while building a route.  Prefer the caller-owned cached
            frame and only take one initial frame when no cache exists.
            """
            self._check_cancel()
            existing = getattr(ctx, "party_slots", None)
            if not isinstance(existing, dict):
                existing = self.party_slots
            else:
                existing = dict(existing)

            frame = None
            cached_frame = getattr(ctx, "cached_frame", None)
            if callable(cached_frame):
                try:
                    cached = cached_frame()
                    frame = cached[0] if isinstance(cached, (tuple, list)) else cached
                except Exception as error:
                    log(f"[combat] getAvatars 无法读取缓存帧：{error}")
            if frame is None:
                frame = ctx.capture_bgr()

            try:
                from .party_hud import recognize_party_slots

                result = recognize_party_slots(
                    ctx,
                    ImageRegion(ctx, frame),
                    existing=existing,
                    log=log,
                )
            except Exception as error:
                log(f"[combat] getAvatars 队伍识别失败（保留已有配置）：{error}")
                result = existing

            if result:
                # Keep the dictionary shared with GenshinApi so
                # clearPartyCache() and a later HUD refresh affect combat
                # hosts created by this same script runtime.
                self.party_slots.clear()
                self.party_slots.update(result)
                try:
                    setattr(ctx, "party_slots", self.party_slots)
                except Exception:
                    pass
            ordered = sorted(
                ((int(slot), str(name)) for name, slot in self.party_slots.items()),
                key=lambda item: item[0],
            )
            return [name for _slot, name in ordered]

        expose("getAvatars", _get_avatars)
        expose("inputText", lambda text: ctx.device.paste_text(str(text)))
        expose("setGameMetrics", pointer.set_metrics)
        expose("getGameMetrics", pointer.get_metrics)

        # 键鼠
        def _key_down(key):
            canonical = normalize_key(str(key))
            if canonical == "LBUTTON":
                pointer.left_down()
            elif canonical == "RBUTTON":
                ctx.input.button_down("sprint")
            elif canonical == "MBUTTON":
                ctx.input.button_down("elementalSight")
            else:
                pointer.clear_intent()
                ctx.input.key_down(str(key))

        def _key_up(key):
            canonical = normalize_key(str(key))
            if canonical == "LBUTTON":
                pointer.left_up()
            elif canonical == "RBUTTON":
                ctx.input.button_up("sprint")
            elif canonical == "MBUTTON":
                ctx.input.button_up("elementalSight")
            else:
                ctx.input.key_up(str(key))

        def _key_press(key):
            canonical = normalize_key(str(key))
            if canonical == "LBUTTON":
                pointer.left_click()
            elif canonical == "RBUTTON":
                ctx.input.key_press("LSHIFT")
            elif canonical == "MBUTTON":
                ctx.input.tap_button("elementalSight")
            else:
                pointer.clear_intent()
                ctx.input.key_press(str(key))

        expose("keyDown", _key_down)
        expose("keyUp", _key_up)
        expose("keyPress", _key_press)
        expose("click", pointer.click_at)
        expose("moveMouseBy", pointer.move_by)
        expose("moveMouseTo", pointer.move_to)
        expose("leftButtonClick", pointer.left_click)
        expose("leftButtonDown", pointer.left_down)
        expose("leftButtonUp", pointer.left_up)
        # Genshin's default PC right mouse binding is sprint, not aimed mode.
        expose("rightButtonClick", lambda: ctx.input.key_press("LSHIFT"))
        expose("rightButtonDown", lambda: ctx.input.button_down("sprint"))
        expose("rightButtonUp", lambda: ctx.input.button_up("sprint"))
        expose("middleButtonClick", lambda: ctx.input.tap_button("elementalSight"))
        expose("middleButtonDown", lambda: ctx.input.button_down("elementalSight"))
        expose("middleButtonUp", lambda: ctx.input.button_up("elementalSight"))
        expose("verticalScroll", lambda n: ctx.input.vertical_scroll(float(n)))

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
            def send(self, msg):
                log(f"[通知] {msg}")
                rt._notification_service.notify(
                    "JsNotification", str(msg), result="Success", from_js=True,
                )
            def error(self, msg):
                log(f"[通知-错误] {msg}")
                rt._notification_service.notify(
                    "JsNotification", str(msg), result="Fail", from_js=True,
                )
        expose("notification", wrap(_Notification()), proxy=False)
        g["settings"] = pm.eval("(o) => o")(self.settings)

        from .html_mask import HtmlMaskHost

        self._html_mask_host = HtmlMaskHost(
            self.script_dir,
            cancelled=lambda: self.cancelled,
        )
        expose("htmlMask", wrap(self._html_mask_host), proxy=False)

        def _create_key_mouse_hook():
            return wrap(self._key_mouse_hooks.create())

        g["KeyMouseHook"] = pm.eval(
            "factory => function KeyMouseHook() { return factory(); }"
        )(_create_key_mouse_hook)

        # file（沙箱）
        rt = self

        class _File:
            def readTextSync(self, p): return rt._resolve(p).read_text(encoding="utf-8")
            def readText(self, p, cb=None):
                try:
                    text = self.readTextSync(p)
                except Exception as error:
                    if cb:
                        cb(str(error), None)
                        return None
                    raise
                if cb:
                    cb(None, text)
                return text
            def writeTextSync(self, p, content, append=False):
                f = rt._resolve(p)
                f.parent.mkdir(parents=True, exist_ok=True)
                mode = "a" if append else "w"
                with open(f, mode, encoding="utf-8") as fh:
                    fh.write(str(content))
                return True
            def writeText(self, p, content, callback_or_append=False, append=False):
                callback = callback_or_append if callable(callback_or_append) else None
                should_append = append if callback else bool(callback_or_append)
                try:
                    result = self.writeTextSync(p, content, should_append)
                except Exception as error:
                    if callback:
                        callback(str(error), None)
                        return False
                    raise
                if callback:
                    callback(None, result)
                return result
            def readImageMatSync(self, p): return wrap(Mat.from_file(str(rt._resolve(p))))
            def readImageMatWithResizeSync(self, p, w, h, interp=1):
                import cv2 as _cv2
                m = Mat.from_file(str(rt._resolve(p)))
                return wrap(Mat(_cv2.resize(
                    m.bgr, (int(w), int(h)), interpolation=int(interp)
                )))
            def writeImageSync(self, p, mat):
                import cv2 as _cv2
                value = getattr(mat, "__wrapped__", mat)
                bgr = value.bgr if isinstance(value, Mat) else getattr(value, "bgr", None)
                if bgr is None:
                    raise TypeError("file.writeImageSync 需要 Mat")
                out = rt._resolve(p)
                out.parent.mkdir(parents=True, exist_ok=True)
                return bool(_cv2.imwrite(str(out), bgr))
            def readPathSync(self, folder):
                base = rt._resolve(folder)
                return [str(p.relative_to(rt.script_dir)) for p in sorted(base.iterdir())]
            def createDirectory(self, folder):
                try:
                    rt._resolve(folder).mkdir(parents=True, exist_ok=True)
                    return True
                except (OSError, ValueError, PermissionError) as error:
                    log(f"[file] CreateDirectory 失败: {error}")
                    return False
            # A few community packages feature-detect ``file.mkdir`` even
            # though BetterGI names the host method CreateDirectory.
            def mkdir(self, folder): return self.createDirectory(folder)
            def renamePathSync(self, old_path, new_path):
                try:
                    source = rt._resolve(old_path)
                    target = rt._resolve(new_path)
                    if not source.exists() or target.exists():
                        return False
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source.rename(target)
                    return True
                except (OSError, ValueError, PermissionError) as error:
                    log(f"[file] RenamePathSync 失败: {error}")
                    return False
            def isFolder(self, p): return rt._resolve(p).is_dir()
            def isFile(self, p): return rt._resolve(p).is_file()
            def isExists(self, p): return rt._resolve(p).exists()
        expose("file", wrap(_File()), proxy=False)

        class _StrategyFile:
            def __init__(self, root: Path):
                self.root = root

            def _resolve(self, path):
                target = (self.root / str(path)).resolve()
                if not target.is_relative_to(self.root):
                    raise PermissionError(f"策略路径越出根目录: {path}")
                return target

            def readPathSync(self, folder="./"):
                base = self._resolve(folder)
                if not base.is_dir():
                    return []
                return [
                    str(path.relative_to(self.root))
                    for path in sorted(base.iterdir())
                ]

            def isFolder(self, path): return self._resolve(path).is_dir()
            def isFile(self, path): return self._resolve(path).is_file()
            def isExists(self, path): return self._resolve(path).exists()

        strategy_root = self.strategy_roots[0]
        expose("strategyFile", wrap(_StrategyFile(strategy_root)), proxy=False)

        # http
        class _Http:
            def request(self, method, url, body=None, headers_json=""):
                return rt._http_request(str(method), str(url), body, str(headers_json or ""))
        expose("http", wrap(_Http()), proxy=False)

        class _ServerTime:
            @staticmethod
            def getServerTimeZoneOffset():
                offset = datetime.now().astimezone().utcoffset()
                return int(offset.total_seconds() * 1000) if offset else 0

        expose("ServerTime", wrap(_ServerTime()), proxy=False)

        # 识别类型。ClearScript HostType 同时可 ``new`` 且有静态工厂；原生 JS
        # 构造器保留这两个语义，再把实例交回大小写不敏感的 Python 宿主。
        def _create_recognition_object():
            return wrap(RecognitionObject())

        def _template_match(mat, x=None, y=None, w=None, h=None):
            value = getattr(mat, "__wrapped__", mat)
            return wrap(RecognitionObject.template_match(value, x, y, w, h))

        def _ocr(*args):
            return wrap(RecognitionObject.ocr(*args))

        def _ocr_match(x, y, w, h, *texts):
            return wrap(RecognitionObject.ocr_match(
                x, y, w, h, *[str(text) for text in texts]
            ))

        def _ocr_this():
            return wrap(RecognitionObject.ocr_this())

        recognition_object_type = pm.eval(r"""
            (factory, templateMatch, ocr, ocrMatch, ocrThis) => {
              function RecognitionObject() { return factory(); }
              RecognitionObject.templateMatch = templateMatch;
              RecognitionObject.TemplateMatch = templateMatch;
              RecognitionObject.ocr = ocr;
              RecognitionObject.Ocr = ocr;
              RecognitionObject.ocrMatch = ocrMatch;
              RecognitionObject.OcrMatch = ocrMatch;
              Object.defineProperties(RecognitionObject, {
                ocrThis: { get: ocrThis }, OcrThis: { get: ocrThis }
              });
              return RecognitionObject;
            }
        """)(_create_recognition_object, _template_match, _ocr, _ocr_match, _ocr_this)
        g["RecognitionObject"] = recognition_object_type

        def _create_search_options(anchor="Auto", box=None, expand=None, percent=None):
            return wrap(SearchOptions(anchor, box, expand, percent))

        g["SearchOptions"] = pm.eval(
            "factory => function SearchOptions(anchor, box, expand, percent) { "
            "return factory(anchor, box, expand, percent); }"
        )(_create_search_options)
        g["SearchExpandRatio"] = pm.eval(
            "factory => function SearchExpandRatio(...values) { "
            "return factory(...values); }"
        )(lambda *values: wrap(SearchExpandRatio(*values)))
        g["SearchAnchorMode"] = pm.eval(
            "Object.freeze({Auto:'Auto', TopLeft:'TopLeft', TopRight:'TopRight', "
            "BottomLeft:'BottomLeft', BottomRight:'BottomRight', Center:'Center'})"
        )

        point_factory = lambda x=0, y=0: wrap(Point2f(float(x), float(y)))

        def _point_from_point(x, y=None):
            return wrap(Point2f(float(x), float(y))) if y is not None else wrap(Point2f.from_point(x))

        def _point_from_vec2f(x, y=None):
            return wrap(Point2f(float(x), float(y))) if y is not None else wrap(Point2f.from_vec2f(x))

        point2f_type = pm.eval(r"""
            (factory, fromPoint, fromVec2f, distance, dotProduct, crossProduct) => {
              function Point2f(x = 0, y = 0) { return factory(x, y); }
              const component = (value, ...names) => {
                for (const name of names) {
                  if (value != null && value[name] != null) return value[name];
                }
                return 0;
              };
              Point2f.fromPoint = Point2f.FromPoint = point => fromPoint(
                component(point, 'X', 'x'), component(point, 'Y', 'y')
              );
              Point2f.fromVec2f = Point2f.FromVec2f = vec => fromVec2f(
                component(vec, 'Item0', 'item0'), component(vec, 'Item1', 'item1')
              );
              Point2f.distance = Point2f.Distance = distance;
              Point2f.dotProduct = Point2f.DotProduct = dotProduct;
              Point2f.crossProduct = Point2f.CrossProduct = crossProduct;
              return Point2f;
            }
        """)(point_factory, _point_from_point, _point_from_vec2f,
             Point2f.distance, Point2f.dot_product_static,
             Point2f.cross_product_static)
        g["Point2f"] = point2f_type

        def _create_mat(*args):
            values = [getattr(value, "__wrapped__", value) for value in args]
            if not values:
                return wrap(Mat())
            if len(values) == 1:
                value = values[0]
                if isinstance(value, Mat):
                    return wrap(value.clone())
                return wrap(Mat.from_array(value))
            if len(values) == 2 and all(
                isinstance(value, (int, float)) for value in values
            ):
                return wrap(Mat.zeros(int(values[0]), int(values[1]), 0))
            if len(values) == 3 and all(
                isinstance(value, (int, float)) for value in values
            ):
                return wrap(Mat.zeros(int(values[0]), int(values[1]), int(values[2])))
            raise TypeError("Mat 构造函数参数不受支持")

        def _mat_static(name, *args):
            factory = getattr(Mat, name)
            return wrap(factory(*args))

        mat_type = pm.eval(r"""
            (factory, fromArray, fromPixelData, imDecode, fromImageData,
             diag, zeros, ones, eye) => {
              function Mat(...args) { return factory(...args); }
              Object.assign(Mat, {
                fromArray, FromArray: fromArray,
                fromPixelData, FromPixelData: fromPixelData,
                imDecode, ImDecode: imDecode,
                fromImageData, FromImageData: fromImageData,
                diag, Diag: diag,
                zeros, Zeros: zeros,
                ones, Ones: ones,
                eye, Eye: eye
              });
              return Mat;
            }
        """)(
            _create_mat,
            lambda value: _mat_static("from_array", value),
            lambda *values: _mat_static("from_pixel_data", *values),
            lambda *values: _mat_static("im_decode", *values),
            lambda *values: _mat_static("from_image_data", *values),
            lambda *values: _mat_static("diag", *values),
            lambda *values: _mat_static("zeros", *values),
            lambda *values: _mat_static("ones", *values),
            lambda *values: _mat_static("eye", *values),
        )
        g["Mat"] = mat_type

        def _create_region(x=0, y=0, w=0, h=0, *_unused):
            dx, dy = ctx.transform.to_device(float(x), float(y))
            return wrap(Region(
                ctx, dx, dy,
                ctx.transform.scale_len(float(w)),
                ctx.transform.scale_len(float(h)),
            ))

        g["Region"] = pm.eval(
            "factory => function Region(x = 0, y = 0, w = 0, h = 0) { "
            "return factory(x, y, w, h); }"
        )(_create_region)

        def _create_desktop_region(width=None, height=None):
            return wrap(DesktopRegion(ctx, width, height))

        def _desktop_click(x, y, width=0, height=0):
            pointer.click_at(
                float(x) + float(width) / 2,
                float(y) + float(height) / 2,
            )

        def _desktop_move(x, y, width=0, height=0):
            pointer.move_to(
                float(x) + float(width) / 2,
                float(y) + float(height) / 2,
            )

        desktop_region_type = pm.eval(r"""
            (factory, click, move, moveBy) => {
              function DesktopRegion(width, height) {
                return factory(width, height);
              }
              Object.assign(DesktopRegion, {
                desktopRegionClick: click, DesktopRegionClick: click,
                desktopRegionMove: move, DesktopRegionMove: move,
                desktopRegionMoveBy: moveBy, DesktopRegionMoveBy: moveBy
              });
              return DesktopRegion;
            }
        """)(_create_desktop_region, _desktop_click, _desktop_move, pointer.move_by)
        g["DesktopRegion"] = desktop_region_type

        pm.eval(r"""
            (() => {
              function makeColor(a, r, g, b, name = '') {
                const value = { a:Number(a), r:Number(r), g:Number(g), b:Number(b), name:String(name) };
                Object.defineProperties(value, {
                  A:{get:()=>value.a}, R:{get:()=>value.r},
                  G:{get:()=>value.g}, B:{get:()=>value.b}, Name:{get:()=>value.name},
                  isEmpty:{get:()=>false}, IsEmpty:{get:()=>false}
                });
                return Object.freeze(value);
              }
              function Color(a = 255, r = 0, g = 0, b = 0) {
                if (arguments.length === 3) return makeColor(255, a, r, g);
                return makeColor(a, r, g, b);
              }
              Color.fromArgb = Color.FromArgb = (...args) => {
                if (args.length === 1) {
                  const argb = Number(args[0]) >>> 0;
                  return makeColor(argb >>> 24, argb >>> 16 & 255, argb >>> 8 & 255, argb & 255);
                }
                if (args.length === 3) return makeColor(255, ...args);
                if (args.length === 4) return makeColor(...args);
                throw new TypeError('Color.FromArgb 需要 1、3 或 4 个参数');
              };
              const palette = {
                Transparent:[0,255,255,255], Black:[255,0,0,0], White:[255,255,255,255],
                Red:[255,255,0,0], Green:[255,0,128,0], Blue:[255,0,0,255],
                Yellow:[255,255,255,0], Orange:[255,255,165,0],
                Coral:[255,255,127,80], Lime:[255,0,255,0],
                LimeGreen:[255,50,205,50], Cyan:[255,0,255,255],
                Magenta:[255,255,0,255], Gray:[255,128,128,128]
              };
              for (const [name, rgba] of Object.entries(palette)) {
                Object.defineProperty(Color, name, {value:makeColor(...rgba, name), enumerable:true});
              }
              function Pen(color = Color.Red, width = 1) {
                this.color = color; this.width = Number(width);
                Object.defineProperties(this, {
                  Color:{get:()=>this.color,set:v=>this.color=v},
                  Width:{get:()=>this.width,set:v=>this.width=Number(v)}
                });
              }
              Pen.prototype.dispose = Pen.prototype.Dispose = function () {};
              globalThis.Color = Color;
              globalThis.Pen = Pen;
            })()
        """)

        def _create_image_region(mat, x=0, y=0):
            value = getattr(mat, "__wrapped__", mat)
            bgr = value.bgr if isinstance(value, Mat) else getattr(value, "bgr", None)
            if bgr is None:
                raise TypeError("ImageRegion 构造函数需要 Mat")
            return wrap(ImageRegion(ctx, bgr, float(x), float(y)))

        def _create_game_capture_region(mat, x=0, y=0):
            value = getattr(mat, "__wrapped__", mat)
            bgr = value.bgr if isinstance(value, Mat) else getattr(value, "bgr", None)
            if bgr is None:
                raise TypeError("GameCaptureRegion 构造函数需要 Mat")
            return wrap(GameCaptureRegion(ctx, bgr, float(x), float(y)))

        # Python callables are not constructable with JS ``new``. Wrap the
        # sandboxed factory in a native JS constructor whose explicit object
        # return value becomes the constructed ImageRegion.
        g["ImageRegion"] = pm.eval(
            "factory => function ImageRegion(mat, x = 0, y = 0) { "
            "return factory(mat, x, y); }"
        )(_create_image_region)

        def _game_region_click(position):
            size = pm.eval("(w, h) => ({width:w,height:h,Width:w,Height:h})")(
                ctx.transform.device_width, ctx.transform.device_height,
            )
            point = position(size, ctx.transform.scale)
            ref_x, ref_y = ctx.transform.to_ref(float(point[0]), float(point[1]))
            pointer.click_at_ref(ref_x, ref_y)

        def _game_region_move(position):
            size = pm.eval("(w, h) => ({width:w,height:h,Width:w,Height:h})")(
                ctx.transform.device_width, ctx.transform.device_height,
            )
            point = position(size, ctx.transform.scale)
            ref_x, ref_y = ctx.transform.to_ref(float(point[0]), float(point[1]))
            pointer.move_to_ref(ref_x, ref_y)

        def _game_region_move_by(delta):
            size = pm.eval("(w, h) => ({width:w,height:h,Width:w,Height:h})")(
                ctx.transform.device_width, ctx.transform.device_height,
            )
            point = delta(size, ctx.transform.scale)
            pointer.move_by(
                float(point[0]) / ctx.transform.scale,
                float(point[1]) / ctx.transform.scale,
            )

        def _game_region_1080p_click(x, y):
            pointer.click_at_ref(float(x), float(y))

        def _game_region_1080p_move(x, y):
            pointer.move_to_ref(float(x), float(y))

        game_capture_region_type = pm.eval(r"""
            (factory, click, move, moveBy, click1080, move1080) => {
              function GameCaptureRegion(mat, x = 0, y = 0) {
                return factory(mat, x, y);
              }
              Object.assign(GameCaptureRegion, {
                gameRegionClick: click, GameRegionClick: click,
                gameRegionMove: move, GameRegionMove: move,
                gameRegionMoveBy: moveBy, GameRegionMoveBy: moveBy,
                gameRegion1080PPosClick: click1080, GameRegion1080PPosClick: click1080,
                gameRegion1080PPosMove: move1080, GameRegion1080PPosMove: move1080
              });
              return GameCaptureRegion;
            }
        """)(
            _create_game_capture_region, _game_region_click, _game_region_move,
            _game_region_move_by, _game_region_1080p_click,
            _game_region_1080p_move,
        )
        g["GameCaptureRegion"] = game_capture_region_type

        g["GridScreenName"] = pm.eval("Object.freeze({"
            "Weapons:'Weapons',Artifacts:'Artifacts',"
            "CharacterDevelopmentItems:'CharacterDevelopmentItems',Food:'Food',"
            "Materials:'Materials',Gadget:'Gadget',Quest:'Quest',"
            "PreciousItems:'PreciousItems',Furnishings:'Furnishings',"
            "ArtifactSalvage:'ArtifactSalvage',Crafting:'Crafting',"
            "PartySetupCharacters:'PartySetupCharacters',"
            "ArtifactSetFilter:'ArtifactSetFilter'})")
        g["ItemIconRecognitionMode"] = pm.eval(
            "Object.freeze({GridIcon:'GridIcon',Item:'Item'})"
        )

        from .bv import BvImage, BvPage

        to_region_collection = pm.eval(
            "values => { const result = Array.from(values); "
            "Object.defineProperty(result, 'count', { get: () => result.length }); "
            "Object.defineProperty(result, 'Count', { get: () => result.length }); "
            "return result; }"
        )

        def _create_bv_page():
            return wrap(BvPage(
                ctx,
                to_collection=to_region_collection,
                check_cancel=self._check_cancel,
            ))

        g["BvPage"] = pm.eval(
            "factory => function BvPage() { return factory(); }"
        )(_create_bv_page)

        asset_aliases = {
            "UseRedeemCode": "redeem",
            "QuickSereniteaPot": "quick_serenitea",
            "AutoArtifactSalvage": "artifact_salvage",
            "AutoCook": "autocook",
            "AutoFishing": "autofishing",
            "AutoMusicGame": "automusic",
            "AutoOpenChest": "autoopenchest",
            "AutoPick": "autopick",
            "AutoSkip": "autoskip",
            "CharacterDevelopment": "character_development",
            "QuickClaimReward": "quick_claim",
            "QuickTeleport": "teleport",
        }
        builtin_template_root = Path(__file__).resolve().parents[2] / "assets" / "templates"

        def _resolve_bv_asset(value: str) -> Mat:
            name = str(value)
            if ":" not in name:
                raise ValueError("BvImage 素材名称必须为 Feature:file.png")
            feature, relative = name.split(":", 1)
            safe_relative = Path(relative.replace("\\", "/"))
            if safe_relative.is_absolute() or ".." in safe_relative.parts:
                raise PermissionError(f"BvImage 素材路径越界: {name}")
            roots = [
                rt.script_dir / "assets" / feature,
                builtin_template_root / asset_aliases.get(feature, feature.lower()),
            ]
            for root in roots:
                candidate = (root / safe_relative).resolve()
                if candidate.is_relative_to(root.resolve()) and candidate.is_file():
                    return Mat.from_file(str(candidate))
            raise FileNotFoundError(f"未找到 BvImage 素材: {name}")

        def _create_bv_image(template_asset, roi=None, threshold=0.8):
            return wrap(BvImage(
                str(template_asset), _resolve_bv_asset, roi, float(threshold)
            ))

        g["BvImage"] = pm.eval(
            "factory => function BvImage(asset, roi, threshold = 0.8) { "
            "return factory(asset, roi, threshold); }"
        )(_create_bv_image)
        def _create_scalar(*values):
            values = list(values)
            if len(values) == 1 and isinstance(values[0], (list, tuple)):
                values = list(values[0])
            values = (values[:4] + [0, 0, 0, 0])[:4]
            return wrap(Scalar(*values))

        g["OpenCvSharp"] = pm.eval(
            "scalarFactory => {"
            "function Vec3b() {}"
            "function Rect(x = 0, y = 0, width = 0, height = 0) {"
            "this.x=Number(x);this.y=Number(y);this.width=Number(width);"
            "this.height=Number(height);"
            "Object.defineProperties(this,{X:{get:()=>this.x,set:v=>this.x=Number(v)},"
            "Y:{get:()=>this.y,set:v=>this.y=Number(v)},"
            "Width:{get:()=>this.width,set:v=>this.width=Number(v)},"
            "Height:{get:()=>this.height,set:v=>this.height=Number(v)}}); }"
            "function Size(width = 0, height = 0) {"
            "this.width=Number(width);this.height=Number(height);"
            "Object.defineProperties(this,{Width:{get:()=>this.width,set:v=>this.width=Number(v)},"
            "Height:{get:()=>this.height,set:v=>this.height=Number(v)}}); }"
            "function Scalar(...values) { return scalarFactory(...values); }"
            "return { OpenCvSharp: { Rect, Size, Vec3b, Scalar }, Rect, Size, Vec3b, Scalar };"
            "}"
        )(_create_scalar)

        # genshin 助手。不要把整个 Python 宿主对象直接交给 SpiderMonkey：
        # PythonMonkey 在 ``await genshin.tp(...)`` 失败时会继续探测宿主对象
        # 的 then/属性，某些异常路径会在 Python↔JS 自动包装层之间递归，最终
        # 只留下 ``InternalError: too much recursion``。逐个绑定公开成员到
        # 一个普通 JS 对象，既保留 Python 方法的同步/异常语义，也让大小写
        # 不敏感 Proxy 只处理纯 JS 对象。
        from .genshin_api import GenshinApi

        genshin_api = GenshinApi(ctx, log, party_slots=self.party_slots)
        genshin_facade = pm.eval("({})")

        def _navigation_facade(navigation):
            # Do not expose the Python object itself.  PythonMonkey may probe
            # arbitrary host attributes while awaiting a failed call, which
            # is the source of the historical ``too much recursion`` error.
            # A plain JS object with individually bound methods has the same
            # ClearScript surface without crossing that proxy boundary.
            facade = pm.eval("({})")
            for name in sorted(
                value for value in dir(navigation) if not value.startswith("_")
            ):
                member = getattr(navigation, name)
                if callable(member):
                    facade[name] = member
            return self._case_proxy(facade)

        for member_name in sorted(name for name in dir(genshin_api) if not name.startswith("_")):
            member = getattr(genshin_api, member_name)
            if member_name == "lazyNavigationInstance":
                genshin_facade[member_name] = _navigation_facade(member)
            elif callable(member) or member is None or isinstance(
                member, (bool, int, float, str)
            ):
                genshin_facade[member_name] = member
        g["genshin"] = self._case_proxy(genshin_facade)

        from .combat_host import Avatar, CombatScenes

        def _create_combat_scenes(*_args):
            return wrap(CombatScenes(
                ctx, self.party_slots, log, to_collection=to_region_collection,
            ))

        def _create_avatar(scenes, name, index, name_rect=None, manual_skill_cd=-1):
            value = getattr(scenes, "__wrapped__", scenes)
            if not isinstance(value, CombatScenes):
                raise TypeError("Avatar 构造函数需要 CombatScenes")
            return wrap(Avatar(
                value, str(name), int(index), name_rect, float(manual_skill_cd)
            ))

        g["CombatScenes"] = pm.eval(
            "factory => function CombatScenes(...args) { return factory(...args); }"
        )(_create_combat_scenes)
        avatar_type = pm.eval(r"""
            (factory, parseCd, tpForRecover) => {
              function Avatar(scenes, name, index, nameRect, manualSkillCd = -1) {
                return factory(scenes, name, index, nameRect, manualSkillCd);
              }
              Avatar.parseActionSchedulerByCd = parseCd;
              Avatar.ParseActionSchedulerByCd = parseCd;
              Avatar.throwWhenDefeated = Avatar.ThrowWhenDefeated = () => {};
              Avatar.tpForRecover = Avatar.TpForRecover = tpForRecover;
              return Avatar;
            }
        """)(
            _create_avatar,
            Avatar.parse_action_scheduler_by_cd,
            lambda token=None, error=None: genshin_api.teleportToStatue(),
        )
        g["Avatar"] = avatar_type

        # ClearScript exposes Microsoft.Extensions.Task and HostFunctions.
        # Promise is the native SpiderMonkey equivalent for await/then, while
        # the small host object retains the safe construction/array helpers
        # scripts commonly use without exposing arbitrary Python types.
        pm.eval(r"""
            globalThis.Task = Promise;
            globalThis.host = Object.freeze({
              newObj(Type, ...args) {
                if (typeof Type !== 'function') throw new TypeError('host.newObj 需要构造函数');
                return new Type(...args);
              },
              newArr(_Type, length) {
                return Array.from({length: Math.max(0, Number(length) | 0)}, () => null);
              },
              newVar(_Type, value = null) { return {value}; },
              newVarOfArr(_Type, dimensions) {
                const depth = Math.max(1, Number(dimensions) | 0);
                const build = level => level >= depth ? null : [build(level + 1)];
                return {value: build(0)};
              },
              typeOf(value) { return typeof value; },
              isType(Type, value) { return typeof Type === 'function' && value instanceof Type; },
              cast(_Type, value) { return value; }
            });
        """)

        from ..tasks.character_development import CharacterDevelopmentTask
        expose(
            "characterDevelopmentTask",
            wrap(CharacterDevelopmentTask(ctx, log=log)),
            proxy=False,
        )

        # BetterGI's global one-key fight hotkey is represented as a host
        # object so community scripts can bind it to KeyMouseHook events.
        from ..combat.one_key_fight import HOLD_ON_MODE, OneKeyFightTask
        one_key_mode = str(
            self.settings.get("combatMacroHotkeyMode", HOLD_ON_MODE) or HOLD_ON_MODE
        )
        one_key_path = self.settings.get("avatarMacroPath")
        if one_key_path:
            candidate = Path(str(one_key_path)).expanduser()
            if not candidate.is_absolute():
                candidate = (self.script_dir / candidate).resolve()
            one_key_path = candidate
        enabled_value = self.settings.get("combatMacroEnabled", True)
        one_key_enabled = enabled_value if isinstance(enabled_value, bool) else (
            str(enabled_value).strip().casefold() not in {"0", "false", "no", "off"}
        )
        self._one_key_fight = OneKeyFightTask(
            ctx,
            party_slots=self.party_slots,
            macro_path=one_key_path,
            mode=one_key_mode,
            priority=int(self.settings.get("combatMacroPriority", 1) or 1),
            enabled=one_key_enabled,
            log=log,
        )
        expose("oneKeyFight", wrap(self._one_key_fight), proxy=False)

        from ..macro.hotkeys import HotkeyMacroHost
        self._hotkey_macros = HotkeyMacroHost(
            ctx,
            runaround_mouse_x_interval=float(
                self.settings.get("runaroundMouseXInterval", 500) or 1
            ),
            runaround_interval_ms=int(
                self.settings.get("runaroundInterval", 10) or 0
            ),
            enhance_wait_delay_ms=int(
                self.settings.get("enhanceWaitDelay", 0) or 0
            ),
            f_fire_interval_ms=int(
                self.settings.get("fFireInterval", 100) or 100
            ),
            space_fire_interval_ms=int(
                self.settings.get("spaceFireInterval", 100) or 100
            ),
            log=log,
        )
        expose("hotkeyMacros", wrap(self._hotkey_macros), proxy=False)

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
            def runFileFromUser(self, p):
                return pathing_exec.run(PathingTask.load(rt._resolve_pathing(p)))
            def isExists(self, p):
                try:
                    return rt._resolve_pathing(p).exists()
                except (OSError, ValueError, PermissionError):
                    return False
            def isFile(self, p):
                try:
                    return rt._resolve_pathing(p).is_file()
                except (OSError, ValueError, PermissionError):
                    return False
            def isFolder(self, p):
                try:
                    return rt._resolve_pathing(p).is_dir()
                except (OSError, ValueError, PermissionError):
                    return False
            def readPathSync(self, p="./"):
                try:
                    base = rt._resolve_pathing(p)
                    if not base.is_dir():
                        return []
                    return [
                        str(child.relative_to(rt.pathing_root))
                        for child in sorted(base.iterdir())
                    ]
                except (OSError, ValueError, PermissionError) as error:
                    log(f"[pathingScript] ReadPathSync 失败: {error}")
                    return []
            def readTextSync(self, p):
                try:
                    return rt._resolve_pathing(p).read_text(encoding="utf-8-sig")
                except (OSError, ValueError, PermissionError) as error:
                    log(f"[pathingScript] ReadTextSync 失败: {error}")
                    return ""
        expose("pathingScript", wrap(_Pathing()), proxy=False)

        # dispatcher / 任务模型。JS、WebUI、CLI 共用同一个实现，避免任务
        # 名称和参数在不同入口逐渐分叉。
        from ..tasks.dispatcher import TaskDispatcher
        task_dispatcher = TaskDispatcher(
            ctx,
            party_slots=self.party_slots,
            log=log,
            notification_service=self._notification_service,
            cancelled=lambda: rt.cancelled,
            strategy_roots=[rt.script_dir, *rt.strategy_roots],
            restrict_strategy_roots=True,
        )

        class _CTS:
            def __init__(self):
                self._cancelled = False
                self._callbacks = []
                self._timer = None
            @property
            def cancelled(self): return self._cancelled
            @property
            def isCancellationRequested(self): return self._cancelled
            @property
            def canBeCanceled(self): return True
            def cancel(self, _throw_on_first=False):
                if self._cancelled:
                    return
                self._cancelled = True
                for callback in self._callbacks:
                    callback()
                self._callbacks.clear()
            def cancelAsync(self): self.cancel()
            def cancelAfter(self, milliseconds):
                # PythonMonkey callbacks must stay on the JS thread. The timer
                # only flips the token; manual cancel still runs registrations.
                self._timer = threading.Timer(
                    float(milliseconds) / 1000,
                    lambda: setattr(self, "_cancelled", True),
                )
                self._timer.daemon = True
                self._timer.start()
            def tryReset(self):
                self._cancelled = False
                return True
            def register(self, callback, *_args):
                if self._cancelled:
                    callback()
                else:
                    self._callbacks.append(callback)
                return None
            unsafeRegister = register
            def throwIfCancellationRequested(self):
                if self._cancelled:
                    raise ScriptCancelled()
            def dispose(self):
                if self._timer is not None:
                    self._timer.cancel()
                self._callbacks.clear()
            @property
            def token(self): return self

        class _Dispatcher:
            def runTask(self, task, ct=None):
                return task_dispatcher.run_task(task, ct)

            def runAutoDomainTask(self, param, ct=None):
                return task_dispatcher.run_auto_domain_task(param, ct)

            def runOneDragonTask(self, param=None, ct=None):
                return task_dispatcher.run_one_dragon_task(param, ct)

            def runCheckRewardsTask(self, param=None, ct=None):
                return task_dispatcher.run_check_rewards_task(param, ct)
            def runWalkToFTask(self, param=None, ct=None):
                return task_dispatcher.run_walk_to_f_task(param, ct)
            def runScanPickTask(self, param=None, ct=None):
                return task_dispatcher.run_scan_pick_task(param, ct)
            def runLowerHeadThenWalkToTask(self, param=None, ct=None):
                return task_dispatcher.run_lower_head_then_walk_to_task(param, ct)
            def runBlessingOfTheWelkinMoonTask(self, param=None, ct=None):
                return task_dispatcher.run_blessing_of_the_welkin_moon_task(param, ct)
            def runClaimBattlePassRewardsTask(self, param=None, ct=None):
                return task_dispatcher.run_claim_battle_pass_rewards_task(param, ct)
            def runClaimEncounterPointsRewardsTask(self, param=None, ct=None):
                return task_dispatcher.run_claim_encounter_points_rewards_task(param, ct)
            def runClaimMailRewardsTask(self, param=None, ct=None):
                return task_dispatcher.run_claim_mail_rewards_task(param, ct)
            def runGoToAdventurersGuildTask(self, param=None, ct=None):
                return task_dispatcher.run_go_to_adventurers_guild_task(param, ct)
            def runGoToCraftingBenchTask(self, param=None, ct=None):
                return task_dispatcher.run_go_to_crafting_bench_task(param, ct)
            def runGoCraftResinTask(self, param=None, ct=None):
                return task_dispatcher.run_go_craft_resin_task(param, ct)
            def runCraftMaterialTask(self, param=None, ct=None):
                return task_dispatcher.run_craft_material_task(param, ct)
            def runSetTimeTask(self, param=None, ct=None):
                return task_dispatcher.run_set_time_task(param, ct)
            def runWonderlandCycleTask(self, param=None, ct=None):
                return task_dispatcher.run_wonderland_cycle_task(param, ct)
            def runReloginTask(self, param=None, ct=None):
                return task_dispatcher.run_relogin_task(param, ct)
            def runChooseTalkOptionTask(self, param=None, ct=None):
                return task_dispatcher.run_choose_talk_option_task(param, ct)
            def runLinneaMiningTask(self, param=None, ct=None):
                return task_dispatcher.run_linnea_mining_task(param, ct)

            def runAutoFightTask(self, param, ct=None):
                return task_dispatcher.run_auto_fight_task(param, ct)
            def runAutoWoodTask(self, param=None, ct=None):
                return task_dispatcher.run_auto_wood_task(param, ct)
            def runAutoCookTask(self, param=None, ct=None):
                return task_dispatcher.run_auto_cook_task(param, ct)
            def runAutoFishingTask(self, param=None, ct=None):
                return task_dispatcher.run_auto_fishing_task(param, ct)
            def runAutoOpenChestTask(self, param=None, ct=None):
                return task_dispatcher.run_auto_open_chest_task(param, ct)
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
            def runAutoBossTask(self, param=None, ct=None):
                return task_dispatcher.run_auto_boss_task(param, ct)
            def runAutoLeyLineTask(self, param=None, ct=None):
                return task_dispatcher.run_auto_leyline_task(param, ct)
            def runAutoLeyLineOutcropTask(self, param=None, ct=None):
                return task_dispatcher.run_auto_leyline_task(param, ct)
            def runQuickSereniteaPotTask(self, param=None, ct=None):
                return task_dispatcher.run_quick_serenitea_pot_task(param, ct)
            def runQuickClaimRewardTask(self, param=None, ct=None):
                return task_dispatcher.run_quick_claim_reward_task(param, ct)
            def runOneKeyExpeditionTask(self, param=None, ct=None):
                return task_dispatcher.run_one_key_expedition_task(param, ct)
            def runAutoTrackTask(self, param=None, ct=None):
                return task_dispatcher.run_auto_track_task(param, ct)
            def runQuickBuyTask(self, param=None, ct=None):
                return task_dispatcher.run_quick_buy_task(param, ct)
            def runUseRedemptionCodeTask(self, param=None, ct=None):
                return task_dispatcher.run_use_redemption_code_task(param, ct)
            def runAutoArtifactSalvageTask(self, param=None, ct=None):
                return task_dispatcher.run_auto_artifact_salvage_task(param, ct)
            def runCountInventoryItemTask(self, param=None, ct=None):
                return task_dispatcher.run_count_inventory_item_task(param, ct)
            def runGetGridIconsTask(self, param=None, ct=None):
                return task_dispatcher.run_get_grid_icons_task(param, ct)
            def runInventoryCountComparisonTask(self, param=None, ct=None):
                return task_dispatcher.run_inventory_count_comparison_task(param, ct)
            def runCharacterDevelopmentTask(self, param=None, ct=None):
                return task_dispatcher.run_character_development_task(param, ct)
            def runScriptGroupTask(self, param=None, ct=None):
                return task_dispatcher.run_script_group_task(param, ct)
            def runMusicPlayerTask(self, param=None, ct=None):
                return task_dispatcher.run_music_player_task(param, ct)
            def runShellTask(self, param=None, ct=None):
                return task_dispatcher.run_shell_task(param, ct)
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

            # ClearScript exposes the dispatcher with case-insensitive member
            # lookup.  PythonMonkey can proxy most host objects, but the
            # dispatcher is intentionally exposed as a callable host object
            # and PascalCase members used by community scripts are not always
            # resolved through that proxy. Keep both spellings explicit so
            # scripts copied from bettergi-scripts-list run unchanged.
            RunTask = runTask
            RunAutoDomainTask = runAutoDomainTask
            RunOneDragonTask = runOneDragonTask
            RunCheckRewardsTask = runCheckRewardsTask
            RunWalkToFTask = runWalkToFTask
            RunScanPickTask = runScanPickTask
            RunLowerHeadThenWalkToTask = runLowerHeadThenWalkToTask
            RunBlessingOfTheWelkinMoonTask = runBlessingOfTheWelkinMoonTask
            RunClaimBattlePassRewardsTask = runClaimBattlePassRewardsTask
            RunClaimEncounterPointsRewardsTask = runClaimEncounterPointsRewardsTask
            RunClaimMailRewardsTask = runClaimMailRewardsTask
            RunGoToAdventurersGuildTask = runGoToAdventurersGuildTask
            RunGoToCraftingBenchTask = runGoToCraftingBenchTask
            RunGoCraftResinTask = runGoCraftResinTask
            RunCraftMaterialTask = runCraftMaterialTask
            RunSetTimeTask = runSetTimeTask
            RunWonderlandCycleTask = runWonderlandCycleTask
            RunReloginTask = runReloginTask
            RunChooseTalkOptionTask = runChooseTalkOptionTask
            RunLinneaMiningTask = runLinneaMiningTask
            RunAutoFightTask = runAutoFightTask
            RunAutoWoodTask = runAutoWoodTask
            RunAutoCookTask = runAutoCookTask
            RunAutoFishingTask = runAutoFishingTask
            RunAutoOpenChestTask = runAutoOpenChestTask
            RunAutoEatTask = runAutoEatTask
            RunAutoMusicGameTask = runAutoMusicGameTask
            RunAutoAlbumTask = runAutoAlbumTask
            RunAutoAlbum = runAutoAlbum
            RunAutoGeniusInvokationTask = runAutoGeniusInvokationTask
            RunAutoStygianOnslaughtTask = runAutoStygianOnslaughtTask
            RunAutoBossTask = runAutoBossTask
            RunAutoLeyLineTask = runAutoLeyLineTask
            RunAutoLeyLineOutcropTask = runAutoLeyLineOutcropTask
            RunQuickSereniteaPotTask = runQuickSereniteaPotTask
            RunQuickClaimRewardTask = runQuickClaimRewardTask
            RunOneKeyExpeditionTask = runOneKeyExpeditionTask
            RunAutoTrackTask = runAutoTrackTask
            RunQuickBuyTask = runQuickBuyTask
            RunUseRedemptionCodeTask = runUseRedemptionCodeTask
            RunAutoArtifactSalvageTask = runAutoArtifactSalvageTask
            RunCountInventoryItemTask = runCountInventoryItemTask
            RunGetGridIconsTask = runGetGridIconsTask
            RunInventoryCountComparisonTask = runInventoryCountComparisonTask
            RunCharacterDevelopmentTask = runCharacterDevelopmentTask
            RunScriptGroupTask = runScriptGroupTask
            RunMusicPlayerTask = runMusicPlayerTask
            RunShellTask = runShellTask
            RunCombatScript = runCombatScript
            AddTimer = addTimer
            AddTrigger = addTrigger
            ClearAllTriggers = clearAllTriggers
            GetLinkedCancellationTokenSource = getLinkedCancellationTokenSource
            GetLinkedCancellationToken = getLinkedCancellationToken
        expose("dispatcher", wrap(_Dispatcher()), proxy=False)

        # 构造器类：脚本里 new RealtimeTimer("AutoPick") / new SoloTask("AutoFight")。
        # ClearScript 的自动大小写别名在 PythonMonkey 原生 JS 构造器上
        # 不会自动生成，因此显式保留 d.ts 中的 Name/Interval/Config。
        g["RealtimeTimer"] = pm.eval(r"""
            (function () {
              function RealtimeTimer(name, cfg) {
                this.name = name == null ? '' : String(name);
                this.interval = 50;
                this.config = cfg;
                Object.defineProperties(this, {
                  Name: { get: () => this.name, set: value => this.name = String(value) },
                  Interval: { get: () => this.interval, set: value => this.interval = Number(value) },
                  Config: { get: () => this.config, set: value => this.config = value }
                });
              }
              return RealtimeTimer;
            })()
        """)
        g["SoloTask"] = pm.eval(r"""
            (function () {
              function SoloTask(name, cfg) {
                this.name = name == null ? '' : String(name);
                this.config = cfg;
                Object.defineProperties(this, {
                  Name: { get: () => this.name, set: value => this.name = String(value) },
                  Config: { get: () => this.config, set: value => this.config = value }
                });
              }
              return SoloTask;
            })()
        """)
        g["CountInventoryItemParam"] = pm.eval(
            "(function(){ return function CountInventoryItemParam(){"
            "this.gridScreenName='Materials';this.itemName=null;this.itemNames=[];"
            "this.iconRecognitionMode='GridIcon';}; })()"
        )
        constructors = pm.eval(r"""
            (function () {
              function caseInsensitive(target) {
                return new Proxy(target, {
                  get(t, prop, recv) {
                    if (typeof prop !== 'string' || prop in t) return Reflect.get(t, prop, recv);
                    const key = Reflect.ownKeys(t).find(k => typeof k === 'string' && k.toLowerCase() === prop.toLowerCase());
                    return key === undefined ? undefined : Reflect.get(t, key, recv);
                  },
                  set(t, prop, value, recv) {
                    if (typeof prop !== 'string' || prop in t) return Reflect.set(t, prop, value, recv);
                    const key = Reflect.ownKeys(t).find(k => typeof k === 'string' && k.toLowerCase() === prop.toLowerCase());
                    return Reflect.set(t, key === undefined ? prop : key, value, recv);
                  }
                });
              }

              function FightFinishDetectConfig() {
                Object.assign(this, {
                  battleEndProgressBarColor: '', battleEndProgressBarColorTolerance: '',
                  fastCheckEnabled: false, fastCheckParams: '', checkAfterSwitchAvatar: false,
                  checkEndDelay: '0.4;钟离,1.4;', beforeDetectDelay: '0.4',
                  rotateFindEnemyEnabled: false, skipFightEndCheckWhenEnemyVisible: false,
                  blockCheckBeforeBattleSeconds: 0, paimonEndCheckEnabled: true,
                  paimonEndCheckDelay: 0.075
                });
                return caseInsensitive(this);
              }

              function AutoFightParam(strategyName) {
                Object.assign(this, {
                  combatStrategyPath: strategyName ? String(strategyName) : '', timeout: 120,
                  fightFinishDetectEnabled: false,
                  finishDetectConfig: new FightFinishDetectConfig(),
                  pickDropsAfterFightEnabled: false, pickDropsAfterFightSeconds: 15,
                  kazuhaPickupEnabled: true, kazuhaPartyName: '', actionSchedulerByCd: '',
                  onlyPickEliteDropsMode: '', battleThresholdForLoot: -1,
                  guardianAvatar: '', guardianCombatSkip: false, guardianAvatarHold: false,
                  checkBeforeBurst: false, isFirstCheck: true, rotaryFactor: 10,
                  burstEnabled: false, qinDoublePickUp: false
                });
                this.setCombatStrategyPath = value => {
                  this.combatStrategyPath = value ? String(value) : '';
                };
                this.setDefault = () => this;
                return caseInsensitive(this);
              }

              function AutoDomainParam(rounds, path) {
                Object.assign(this, {
                  domainRoundNum: Number(rounds || 0) || 9999,
                  combatStrategyPath: path ? String(path) : '', partyName: '',
                  domainName: '', sundaySelectedValue: '', autoArtifactSalvage: false,
                  maxArtifactStar: '4', specifyResinUse: false,
                  resinPriorityList: ['浓缩树脂', '原粹树脂'],
                  originalResinUseCount: 0, originalResin20UseCount: 0,
                  originalResin40UseCount: 0, condensedResinUseCount: 0,
                  transientResinUseCount: 0, fragileResinUseCount: 0,
                  rewardRecognitionEnabled: false
                });
                this.setDefault = () => this;
                this.setCombatStrategyPath = value => {
                  this.combatStrategyPath = value ? String(value) : '';
                  return this.combatStrategyPath;
                };
                this.setResinPriorityList = (...values) => {
                  this.resinPriorityList = values.map(String);
                };
                return caseInsensitive(this);
              }

              function AutoLeyLineOutcropParam(count, country, type) {
                Object.assign(this, {
                  count: Number(count || 0), country: country ? String(country) : '',
                  leyLineOutcropType: type ? String(type) : '',
                  isResinExhaustionMode: false, openModeCountMin: false,
                  useAdventurerHandbook: false, friendshipTeam: '', team: '',
                  timeout: 120, isGoToSynthesizer: false, useFragileResin: false,
                  useTransientResin: false, isNotification: false,
                  scanDropsAfterRewardEnabled: false, scanDropsAfterRewardSeconds: 12,
                  fightConfig: caseInsensitive({ strategyName: '', teamNames: '', timeout: 120,
                    fightFinishDetectEnabled: true, seekEnemyEnabled: false,
                    seekEnemyIntervalSeconds: 3, seekEnemyRotaryFactor: 6 })
                });
                this.setDefault = () => this;
                return caseInsensitive(this);
              }

              function AutoStygianOnslaughtParam(path) {
                Object.assign(this, {
                  routePath: '', bossNum: 1,
                  autoArtifactSalvage: false, specifyResinUse: false,
                  resinPriorityList: ['浓缩树脂', '原粹树脂'], originalResinUseCount: 0,
                  condensedResinUseCount: 0, transientResinUseCount: 0,
                  fragileResinUseCount: 0, fightTeamName: '',
                  combatScriptBagPath: path ? String(path) : '',
                  confirmQuickSalvage: false, confirmArtifactSalvage: false
                });
                this.setCombatStrategyPath = value => {
                  this.combatScriptBagPath = value ? String(value) : '';
                  return this.combatScriptBagPath;
                };
                this.setResinPriorityList = (...values) => {
                  this.resinPriorityList = values.map(String);
                };
                return caseInsensitive(this);
              }

              function AutoBossParam(path) {
                Object.assign(this, {
                  bossName: '', strategyName: '',
                  combatStrategyPath: path ? String(path) : '', teamName: '',
                  specifyRunCount: false, runCount: 1, useTransientResin: false,
                  useFragileResin: false, reviveRetryCount: 3,
                  returnToStatueAfterEachRound: false,
                  rewardRecognitionEnabled: false, timeout: 240
                });
                this.setDefault = () => this;
                this.setCombatStrategyPath = value => {
                  this.combatStrategyPath = value ? String(value) : '';
                };
                return caseInsensitive(this);
              }

              function AutoSkipConfig() {
                Object.assign(this, {
                  enabled: true,
                  quicklySkipConversationsEnabled: true,
                  afterChooseOptionSleepDelay: 0,
                  autoWaitDialogueOptionVoiceEnabled: false,
                  dialogueOptionVoiceMaxWaitSeconds: 30,
                  beforeClickConfirmDelay: 0,
                  autoGetDailyRewardsEnabled: true,
                  autoReExploreEnabled: true,
                  autoReExploreCharacter: '',
                  clickChatOption: '优先选择第一个选项',
                  customPriorityOptionsEnabled: false,
                  customPriorityOptions: '',
                  autoHangoutEventEnabled: false,
                  autoHangoutEndChoose: '',
                  autoHangoutChooseOptionSleepDelay: 0,
                  autoHangoutPressSkipEnabled: true,
                  runBackgroundEnabled: false,
                  bringGameToFrontAfterBackgroundDialogEnabled: false,
                  submitGoodsEnabled: true,
                  pictureInPictureEnabled: false,
                  pictureInPictureSourceType: 'CaptureLoop',
                  closePopupPagedEnabled: true,
                  skipBuiltInClickOptions: false
                });
                this.isClickFirstChatOption = () =>
                  this.clickChatOption === '优先选择第一个选项';
                this.isClickRandomChatOption = () =>
                  this.clickChatOption === '随机选择选项';
                this.isClickNoneChatOption = () =>
                  this.clickChatOption === '不选择选项';
                return caseInsensitive(this);
              }

              function CancellationTokenSource() {
                this.cancelled = false;
                this._callbacks = [];
                this.token = null;
                Object.defineProperties(this, {
                  isCancellationRequested: { get: () => this.cancelled },
                  canBeCanceled: { get: () => true }
                });
                this.cancel = () => {
                  if (this.cancelled) return;
                  this.cancelled = true;
                  for (const callback of this._callbacks.splice(0)) callback();
                };
                this.cancelAsync = async () => this.cancel();
                this.cancelAfter = milliseconds => setTimeout(() => this.cancel(), Number(milliseconds));
                this.tryReset = () => { this.cancelled = false; return true; };
                this.dispose = () => { this._callbacks.length = 0; };
                this.register = callback => {
                  if (this.cancelled) callback(); else this._callbacks.push(callback);
                  return { dispose() {} };
                };
                this.unsafeRegister = this.register;
                this.throwIfCancellationRequested = () => {
                  if (this.cancelled) throw new Error('OperationCanceledException');
                };
                const proxy = caseInsensitive(this);
                proxy.token = proxy;
                return proxy;
              }
              CancellationTokenSource.createLinkedTokenSource = (...tokens) => {
                const source = new CancellationTokenSource();
                for (const token of tokens) {
                  if (token && token.isCancellationRequested) source.cancel();
                  else if (token && typeof token.register === 'function') token.register(() => source.cancel());
                }
                return source;
              };
              CancellationTokenSource.CreateLinkedTokenSource =
                CancellationTokenSource.createLinkedTokenSource;
              const CancellationToken = CancellationTokenSource;
              CancellationToken.none = caseInsensitive({
                isCancellationRequested: false, canBeCanceled: false,
                register: () => ({ dispose() {} }),
                unsafeRegister: () => ({ dispose() {} }),
                throwIfCancellationRequested: () => {}
              });
              CancellationToken.None = CancellationToken.none;

              function PostMessage() {
                this.keyDown = globalThis.keyDown;
                this.keyUp = globalThis.keyUp;
                this.keyPress = globalThis.keyPress;
                this.click = globalThis.leftButtonClick;
                return caseInsensitive(this);
              }

              AutoFightParam.SwimmingEnabled = false;
              AutoFightParam.FightFinishDetectConfig = FightFinishDetectConfig;

              return {
                FightFinishDetectConfig, AutoFightParam, AutoDomainParam,
                AutoLeyLineOutcropParam, AutoStygianOnslaughtParam, AutoBossParam,
                AutoSkipConfig, CancellationTokenSource, CancellationToken, PostMessage
              };
            })()
        """)
        for name in (
            "FightFinishDetectConfig", "AutoFightParam", "AutoDomainParam",
            "AutoLeyLineOutcropParam", "AutoStygianOnslaughtParam", "AutoBossParam",
            "AutoSkipConfig", "CancellationTokenSource", "CancellationToken", "PostMessage",
        ):
            g[name] = constructors[name]
        pm.eval(r"""
            globalThis.AutoFightParam.SwimmingEnabled = false;
            globalThis.AutoFightParam.FightFinishDetectConfig =
              globalThis.FightFinishDetectConfig;
        """)
        pm.eval("""
            CancellationTokenSource.createLinkedTokenSource = (...tokens) => {
              const source = new CancellationTokenSource();
              for (const token of tokens) {
                if (token && token.isCancellationRequested) source.cancel();
                else if (token && typeof token.register === 'function') {
                  token.register(() => source.cancel());
                }
              }
              return source;
            };
            CancellationTokenSource.CreateLinkedTokenSource =
              CancellationTokenSource.createLinkedTokenSource;
            CancellationToken.none = {
              isCancellationRequested: false, canBeCanceled: false,
              register: () => ({ dispose() {} }),
              unsafeRegister: () => ({ dispose() {} }),
              throwIfCancellationRequested: () => {}
            };
            CancellationToken.None = CancellationToken.none;
        """)

    # ---- run ----

    def run(self, entry: str | None = None) -> Any:
        main = entry or self.manifest.get("main") or "main.js"
        main_path = self._resolve(main)
        from .js_modules import JsModuleLoader, extract_imports

        module_loader = JsModuleLoader(
            self.pm, self.script_dir, self.manifest, self._wrap,
        )
        import_code, source = extract_imports(
            main_path.read_text(encoding="utf-8-sig")
        )
        code = import_code + "\n" + _unwrap_async_iife(source)
        self.pm.eval("globalThis")["__bgi_require"] = (
            lambda value: module_loader.require(str(value), main_path)
        )
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
        finally:
            pointer = getattr(self, "_pointer", None)
            if pointer is not None:
                pointer.release_all()
                if getattr(self.ctx, "_script_pointer", None) is pointer:
                    delattr(self.ctx, "_script_pointer")
            one_key_fight = getattr(self, "_one_key_fight", None)
            if one_key_fight is not None:
                one_key_fight.stop()
            hotkey_macros = getattr(self, "_hotkey_macros", None)
            if hotkey_macros is not None:
                hotkey_macros.stop()
            key_mouse_hooks = getattr(self, "_key_mouse_hooks", None)
            if key_mouse_hooks is not None:
                key_mouse_hooks.close_all()
            html_mask_host = getattr(self, "_html_mask_host", None)
            if html_mask_host is not None:
                html_mask_host.closeAll()
            notification_service = getattr(self, "_notification_service", None)
            if notification_service is not None:
                notification_service.close()
