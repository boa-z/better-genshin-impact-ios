"""BetterGI redemption-code task implemented with OCR and DeviceHub text input."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from ..engine.context import GameContext
from ..engine.genshin_api import GenshinApi
from ..engine.recognition import RecognitionObject


@dataclass(frozen=True)
class RedeemCode:
    code: str
    items: str | None = None


_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def strip_redeem_code_urls(value: Any) -> str:
    """Remove shared-page URLs before parsing pasted redemption codes.

    BetterGI's clipboard importer strips URLs first because announcement text
    commonly contains a gift-page link next to the actual code. Keep this
    normalization separate so explicit JS ``code`` values and pasted text use
    the same safe behavior without trying to treat a URL query parameter as a
    redeem code.
    """
    return _URL_RE.sub("", str(value or ""))


def normalize_redeem_codes(value: Any) -> tuple[RedeemCode, ...]:
    """Accept BetterGI string/object lists and newline/comma-separated text."""
    if value is None:
        return ()
    if isinstance(value, str):
        raw_items: Iterable[Any] = strip_redeem_code_urls(value).replace(",", "\n").splitlines()
    elif isinstance(value, Mapping):
        raw_items = (value,)
    else:
        try:
            raw_items = tuple(value)
        except TypeError as error:
            raise ValueError("兑换码必须是字符串、对象或数组") from error
    output: list[RedeemCode] = []
    seen: set[str] = set()
    for raw in raw_items:
        if isinstance(raw, Mapping):
            code = raw.get("code", raw.get("Code", ""))
            items = raw.get("items", raw.get("Items"))
        else:
            code, items = raw, None
        normalized = strip_redeem_code_urls(code).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(RedeemCode(normalized, str(items) if items else None))
    return tuple(output)


class UseRedemptionCodeTask:
    def __init__(
        self,
        ctx: GameContext,
        codes: Any,
        *,
        timeout_s: float = 120.0,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.codes = normalize_redeem_codes(codes)
        if not self.codes:
            raise ValueError("UseRedemptionCode 需要至少一个非空兑换码")
        self.timeout_s = max(20.0, float(timeout_s))
        self.log = log

    @staticmethod
    def _clean(text: str) -> str:
        return str(text).replace(" ", "").replace("\u3000", "")

    def _find_text(self, words: tuple[str, ...], roi=(0, 0, 1920, 1080)):
        region = self.ctx.capture_region()
        hits = region.find_multi(RecognitionObject.ocr(*roi), limit=60)
        for hit in hits:
            text = self._clean(hit.text)
            if any(self._clean(word) in text for word in words):
                return hit
        return None

    def _wait_text(self, words: tuple[str, ...], deadline: float,
                   roi=(0, 0, 1920, 1080)):
        while time.monotonic() < deadline:
            hit = self._find_text(words, roi)
            if hit is not None:
                return hit
            self.ctx.sleep(250)
        return None

    def _open_redeem_dialog(self, deadline: float) -> bool:
        self.ctx.input.key_press("ESCAPE")
        self.ctx.sleep(900)
        settings = self._wait_text(("设置", "Settings"), min(deadline, time.monotonic() + 3))
        if settings is not None:
            settings.click()
        else:
            self.ctx.input.click_ref(45, 825)
        self.ctx.sleep(700)
        account = self._wait_text(("账户", "Account"), min(deadline, time.monotonic() + 5),
                                  (0, 0, 700, 1080))
        if account is None:
            return False
        account.click()
        self.ctx.sleep(400)
        redeem = self._wait_text(("前往兑换", "Redeem Now", "兑换"),
                                 min(deadline, time.monotonic() + 5), (800, 0, 1120, 1080))
        if redeem is None:
            return False
        redeem.click()
        return self._wait_text(("兑换奖励", "兑换码", "Redemption Code"),
                               min(deadline, time.monotonic() + 5)) is not None

    def _use_one(self, item: RedeemCode, deadline: float) -> bool:
        field = self._wait_text(
            ("点击输入兑换码", "请输入兑换码", "兑换码", "Enter redemption code"),
            min(deadline, time.monotonic() + 4),
            (350, 250, 1220, 520),
        )
        if field is not None:
            field.click()
        else:
            self.ctx.input.click_ref(960, 510)
        self.ctx.sleep(150)
        self.ctx.device.paste_text(item.code)
        self.ctx.sleep(250)
        confirm = self._wait_text(("兑换", "Redeem"), min(deadline, time.monotonic() + 3),
                                  (900, 600, 1020, 480))
        if confirm is None:
            return False
        confirm.click()
        result = self._wait_text(
            ("兑换成功", "已被使用", "兑换码无效", "兑换码已过期", "成功", "失败"),
            min(deadline, time.monotonic() + 5),
        )
        success = result is not None and any(
            word in self._clean(result.text) for word in ("兑换成功", "成功")
        )
        if success:
            self.log(f"[UseRedemptionCode] {item.code} 兑换成功")
            ok = self._find_text(("确认", "确定", "OK"), (700, 550, 1220, 530))
            if ok is not None:
                ok.click()
            self.ctx.sleep(600)
            return True
        self.log(f"[UseRedemptionCode] {item.code} 兑换失败或已使用")
        clear = self._find_text(("清除", "Clear"), (700, 250, 1220, 600))
        if clear is not None:
            clear.click()
        return False

    def run(self, cancelled: Callable[[], bool] | None = None) -> dict[str, bool]:
        deadline = time.monotonic() + self.timeout_s
        api = GenshinApi(self.ctx, log=self.log)
        if not api.returnMainUi() or not self._open_redeem_dialog(deadline):
            raise RuntimeError("无法打开设置中的兑换码界面")
        results: dict[str, bool] = {}
        try:
            for item in self.codes:
                if time.monotonic() >= deadline or (cancelled and cancelled()):
                    break
                self.log(f"[UseRedemptionCode] 输入兑换码 {item.code}")
                results[item.code] = self._use_one(item, deadline)
            return results
        finally:
            api.returnMainUi()
