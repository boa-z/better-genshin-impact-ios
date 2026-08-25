"""Enter and leave the Thousand-Star Realm.

The desktop job is a small but important state machine: opening the realm is
not complete when the realm list appears.  A realm must be selected and
confirmed, the lobby must load, and the return-to-Teyvat confirmation must be
accepted as a second dialog.  Keeping those states explicit also prevents
AutoWood's cooldown refresh from reporting success after a single blind ESC.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from ..engine.context import GameContext
from ..engine.recognition import ImageRegion, Mat, RecognitionObject
from ..vision.game_ui import is_main_ui


ASSETS = Path(__file__).resolve().parents[2] / "assets" / "templates"
WONDERLAND_ASSETS = ASSETS / "wonderland"
COMMON_ASSETS = ASSETS / "stygian"


class WonderlandCycleTask:
    """Run the same enter/exit contract as BetterGI's common job."""

    def __init__(
        self,
        ctx: GameContext,
        *,
        timeout_s: float = 150.0,
        log: Callable[[str], None] = print,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.ctx = ctx
        self.timeout_s = max(15.0, float(timeout_s))
        self.log = log
        self.clock = clock
        self._recognitions: dict[str, RecognitionObject | None] = {}

    @staticmethod
    def _clean(value: object) -> str:
        return (
            str(value or "")
            .replace(" ", "")
            .replace("\u3000", "")
            .replace("\n", "")
            .replace("\r", "")
        )

    def _template_path(self, name: str) -> Path | None:
        candidates = [WONDERLAND_ASSETS / f"{name}.png"]
        if name == "btn_black_confirm":
            candidates.append(COMMON_ASSETS / "black_confirm.png")
        return next((path for path in candidates if path.is_file()), None)

    def _recognition(self, name: str) -> RecognitionObject | None:
        if name in self._recognitions:
            return self._recognitions[name]
        path = self._template_path(name)
        if path is None:
            self.log(f"[Wonderland] 缺少界面模板：{name}")
            self._recognitions[name] = None
            return None
        try:
            recognition = RecognitionObject.template_match(Mat.from_file(str(path)))
            recognition.threshold = 0.65
        except (OSError, ValueError, RuntimeError) as error:
            self.log(f"[Wonderland] 模板加载失败 {name}：{error}")
            recognition = None
        self._recognitions[name] = recognition
        return recognition

    def _find_template(self, frame, name: str):
        recognition = self._recognition(name)
        if recognition is None:
            return None
        return ImageRegion(self.ctx, frame).find(recognition)

    def _find_text(self, frame, *keywords: str, roi=(0, 0, 1920, 1080)):
        wanted = tuple(self._clean(value) for value in keywords if self._clean(value))
        if not wanted:
            return None
        hits = ImageRegion(self.ctx, frame).find_multi(
            RecognitionObject.ocr(*roi), limit=50,
        )
        return next(
            (
                hit for hit in hits
                if any(term in self._clean(getattr(hit, "text", "")) for term in wanted)
            ),
            None,
        )

    def _wait_template(
        self,
        name: str,
        deadline: float,
        *,
        action: Callable[[], None] | None = None,
        interval_ms: int = 300,
    ):
        while self.clock() < deadline:
            frame = self.ctx.capture_bgr()
            hit = self._find_template(frame, name)
            if hit is not None and hit.is_exist():
                return hit
            if action is not None:
                action()
            self.ctx.sleep(interval_ms)
        return None

    def _wait_main_ui(self, deadline: float) -> bool:
        while self.clock() < deadline:
            frame = self.ctx.capture_bgr()
            if is_main_ui(self.ctx, frame):
                return True
            self.ctx.sleep(500)
        return False

    def _click_and_wait_disappear(self, hit, name: str, deadline: float) -> bool:
        if hit is None or not hit.is_exist():
            return False
        hit.click()
        while self.clock() < deadline:
            frame = self.ctx.capture_bgr()
            current = self._find_template(frame, name)
            if current is None or not current.is_exist():
                return True
            self.ctx.sleep(250)
        return False

    def _open_realm_list(self, deadline: float) -> bool:
        # F6 is the desktop shortcut used by the upstream job.  If an older
        # profile does not expose it, the OCR fallback uses the normal menu.
        self.ctx.input.key_press("F6")
        self.ctx.sleep(500)
        if self._wait_template("wonderland_close", min(deadline, self.clock() + 10.0)):
            return True

        self.ctx.input.key_press("ESCAPE")
        self.ctx.sleep(800)
        frame = self.ctx.capture_bgr()
        option = self._find_text(
            frame,
            "千星奇域",
            "奇域",
            "Wonderland",
            "Imaginarium",
            roi=(700, 0, 1220, 1080),
        )
        if option is None:
            return False
        option.click()
        self.ctx.sleep(800)
        return self._wait_template("wonderland_close", min(deadline, self.clock() + 10.0)) is not None

    def _select_realm(self, deadline: float) -> bool:
        def select_action() -> None:
            # The fixed card position is the upstream fallback and, more
            # importantly, does not request a second screenshot while the
            # polling frame is being consumed.  The list's card layout is
            # anchored to the same 1080p reference space on iOS.
            self.ctx.input.click_ref(680, 310)

        confirm = self._wait_template(
            "btn_black_confirm",
            deadline,
            action=select_action,
            interval_ms=800,
        )
        if confirm is None:
            return False
        return self._click_and_wait_disappear(confirm, "btn_black_confirm", deadline)

    def _return_to_teyvat(self, deadline: float) -> bool:
        self.ctx.input.key_press("ESCAPE")
        self.ctx.sleep(800)
        back = self._wait_template("btn_back_teyvat", min(deadline, self.clock() + 20.0))
        if back is not None:
            if not self._click_and_wait_disappear(back, "btn_back_teyvat", deadline):
                return False
        else:
            frame = self.ctx.capture_bgr()
            back_text = self._find_text(
                frame,
                "返回提瓦特",
                "回到提瓦特",
                "Back to Teyvat",
                "Teyvat",
                roi=(700, 200, 1100, 700),
            )
            if back_text is None:
                return False
            back_text.click()
            self.ctx.sleep(500)

        confirm = self._wait_template("btn_black_confirm", deadline)
        if confirm is None:
            # Some clients use a translated text-only confirmation dialog.
            frame = self.ctx.capture_bgr()
            confirm_text = self._find_text(
                frame, "确认", "确定", "Confirm", roi=(750, 500, 950, 500)
            )
            if confirm_text is None:
                return False
            confirm_text.click()
            return True
        return self._click_and_wait_disappear(confirm, "btn_black_confirm", deadline)

    def run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        deadline = self.clock() + self.timeout_s
        if cancelled and cancelled():
            return False
        self.log("[Wonderland] 进入千星奇域")
        if not self._open_realm_list(deadline):
            self.log("[Wonderland] 未打开千星奇域界面")
            return False
        if cancelled and cancelled():
            return False
        if not self._select_realm(deadline):
            self.log("[Wonderland] 未能选择并确认奇域")
            return False
        if not self._wait_main_ui(min(deadline, self.clock() + 120.0)):
            self.log("[Wonderland] 进入奇域大厅超时")
            return False
        self.log("[Wonderland] 已进入奇域大厅，准备返回提瓦特")
        if cancelled and cancelled():
            return False
        if not self._return_to_teyvat(deadline):
            self.log("[Wonderland] 返回提瓦特确认失败")
            return False
        if not self._wait_main_ui(deadline):
            self.log("[Wonderland] 返回提瓦特后未恢复主界面")
            return False
        self.log("[Wonderland] 已返回提瓦特")
        return True


EnterAndExitWonderlandJob = WonderlandCycleTask
