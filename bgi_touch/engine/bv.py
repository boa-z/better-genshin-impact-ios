"""Portable BetterGI BvPage/BvLocator JavaScript vision API."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable

from .recognition import ImageRegion, RecognitionObject, Region


@dataclass
class Rect:
    x: float = 0
    y: float = 0
    width: float = 0
    height: float = 0

    X = property(lambda self: self.x, lambda self, value: setattr(self, "x", float(value)))
    Y = property(lambda self: self.y, lambda self, value: setattr(self, "y", float(value)))
    Width = property(
        lambda self: self.width,
        lambda self, value: setattr(self, "width", float(value)),
    )
    Height = property(
        lambda self: self.height,
        lambda self, value: setattr(self, "height", float(value)),
    )


def _unwrap(value: Any) -> Any:
    return getattr(value, "__wrapped__", value)


def _rect(value: Any) -> tuple[float, float, float, float] | None:
    value = _unwrap(value)
    if value is None:
        return None
    if isinstance(value, Rect):
        parts = (value.x, value.y, value.width, value.height)
    elif isinstance(value, dict):
        lowered = {str(key).lower(): item for key, item in value.items()}
        parts = (
            lowered.get("x", 0), lowered.get("y", 0),
            lowered.get("width", 0), lowered.get("height", 0),
        )
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        parts = tuple(value)
    else:
        parts = tuple(
            getattr(value, name, getattr(value, name.capitalize(), 0))
            for name in ("x", "y", "width", "height")
        )
    result = tuple(float(part) for part in parts)
    return None if result == (0.0, 0.0, 0.0, 0.0) else result


class BvLocator:
    def __init__(
        self,
        ctx,
        recognition: RecognitionObject,
        *,
        to_collection: Callable[[list[Region]], Any] = lambda values: values,
        check_cancel: Callable[[], None] = lambda: None,
        text: str = "",
        any_texts: list[str] | None = None,
    ):
        recognition = _unwrap(recognition)
        if not isinstance(recognition, RecognitionObject):
            raise TypeError("BvLocator 需要 RecognitionObject")
        self.ctx = ctx
        self.recognition_object = copy.copy(recognition)
        self.to_collection = to_collection
        self.check_cancel = check_cancel
        self.text = str(text)
        self.any_texts = [str(value) for value in (any_texts or [])]
        self.retry_action: Callable[[Any], Any] | None = None
        self.timeout: int | None = None
        self.retry_interval: int | None = None

    recognitionObject = property(lambda self: self.recognition_object)
    RecognitionObject = property(lambda self: self.recognition_object)

    def _find_all(self) -> list[Region]:
        self.check_cancel()
        screen = ImageRegion(self.ctx, self.ctx.capture_bgr())
        ro = self.recognition_object
        if ro.recognition_type == "TemplateMatch":
            found = screen.find(ro)
            return [found] if found.is_exist() else []
        if ro.recognition_type not in ("Ocr", "OcrMatch"):
            raise NotImplementedError(
                f"BvLocator 不支持识别类型 {ro.recognition_type}"
            )
        values = screen.find_multi(ro, limit=100)
        if self.any_texts:
            return [
                value for value in values
                if any(text in value.text for text in self.any_texts)
            ]
        if self.text:
            return [value for value in values if self.text in value.text]
        return values

    def find_all(self):
        return self.to_collection(self._find_all())

    def is_exist(self) -> bool:
        return bool(self._find_all())

    def _attempts(self, timeout: int | None) -> tuple[int, int, int]:
        actual_timeout = int(timeout if timeout is not None else self.timeout or 10000)
        interval = int(self.retry_interval or 250)
        if actual_timeout <= 0 or interval <= 0:
            raise ValueError("BvLocator timeout/retryInterval 必须大于 0")
        return actual_timeout, interval, max(1, math.ceil(actual_timeout / interval))

    def _retry(self, *, disappear: bool, timeout: int | None = None) -> list[Region]:
        actual_timeout, interval, attempts = self._attempts(timeout)
        last: list[Region] = []
        for attempt in range(attempts):
            self.check_cancel()
            last = self._find_all()
            done = not last if disappear else bool(last)
            if done:
                return last
            if self.retry_action is not None:
                self.retry_action(self.to_collection(last))
            if attempt + 1 < attempts:
                self.ctx.sleep(interval)
        state = "消失" if disappear else "出现"
        raise TimeoutError(f"BvLocator 等待元素{state}超时（{actual_timeout}ms）")

    def wait_for(self, timeout: int | None = None):
        return self.to_collection(self._retry(disappear=False, timeout=timeout))

    def try_wait_for(self, timeout: int | None = None):
        try:
            return self.wait_for(timeout)
        except TimeoutError:
            return self.to_collection([])

    def wait_for_disappear(self, timeout: int | None = None) -> None:
        self._retry(disappear=True, timeout=timeout)

    def try_wait_for_disappear(self, timeout: int | None = None) -> None:
        try:
            self.wait_for_disappear(timeout)
        except TimeoutError:
            pass

    def click(self, timeout: int | None = None) -> Region:
        region = self._retry(disappear=False, timeout=timeout)[0]
        region.click()
        return region

    def double_click(self, timeout: int | None = None) -> Region:
        region = self._retry(disappear=False, timeout=timeout)[0]
        region.double_click()
        return region

    def click_until_disappears(self, timeout: int | None = None) -> Region:
        region = self._retry(disappear=False, timeout=timeout)[0]
        region.click()
        cloned = self.clone()
        cloned.retry_action = lambda values: values[0].click() if values else None
        cloned.wait_for_disappear(timeout)
        return region

    def with_roi(self, rect: Any) -> "BvLocator":
        self.recognition_object.roi = _rect(rect)
        return self

    def with_retry_action(self, action: Callable[[Any], Any] | None) -> "BvLocator":
        self.retry_action = action
        return self

    def with_timeout(self, timeout: int) -> "BvLocator":
        if int(timeout) <= 0:
            raise ValueError("timeout 必须大于 0")
        self.timeout = int(timeout)
        return self

    def with_retry_interval(self, interval: int) -> "BvLocator":
        if int(interval) <= 0:
            raise ValueError("retryInterval 必须大于 0")
        self.retry_interval = int(interval)
        return self

    def clone(self) -> "BvLocator":
        cloned = BvLocator(
            self.ctx,
            self.recognition_object,
            to_collection=self.to_collection,
            check_cancel=self.check_cancel,
            text=self.text,
            any_texts=self.any_texts,
        )
        cloned.retry_action = self.retry_action
        cloned.timeout = self.timeout
        cloned.retry_interval = self.retry_interval
        return cloned

    findAll = find_all
    FindAll = find_all
    isExist = is_exist
    IsExist = is_exist
    waitFor = wait_for
    WaitFor = wait_for
    tryWaitFor = try_wait_for
    TryWaitFor = try_wait_for
    waitForDisappear = wait_for_disappear
    WaitForDisappear = wait_for_disappear
    tryWaitForDisappear = try_wait_for_disappear
    TryWaitForDisappear = try_wait_for_disappear
    Click = click
    doubleClick = double_click
    DoubleClick = double_click
    clickUntilDisappears = click_until_disappears
    ClickUntilDisappears = click_until_disappears
    withRoi = with_roi
    WithRoi = with_roi
    withRetryAction = with_retry_action
    WithRetryAction = with_retry_action
    withTimeout = with_timeout
    WithTimeout = with_timeout
    withRetryInterval = with_retry_interval
    WithRetryInterval = with_retry_interval


class BvPage:
    def __init__(
        self,
        ctx,
        *,
        to_collection: Callable[[list[Region]], Any] = lambda values: values,
        check_cancel: Callable[[], None] = lambda: None,
    ):
        self.ctx = ctx
        self.to_collection = to_collection
        self.check_cancel = check_cancel
        self.default_timeout = 10000
        self.default_retry_interval = 1000

    defaultTimeout = property(
        lambda self: self.default_timeout,
        lambda self, value: setattr(self, "default_timeout", int(value)),
    )
    DefaultTimeout = defaultTimeout
    defaultRetryInterval = property(
        lambda self: self.default_retry_interval,
        lambda self, value: setattr(self, "default_retry_interval", int(value)),
    )
    DefaultRetryInterval = defaultRetryInterval

    def screenshot(self) -> ImageRegion:
        self.check_cancel()
        return ImageRegion(self.ctx, self.ctx.capture_bgr())

    def wait(self, milliseconds: int) -> "BvPage":
        self.check_cancel()
        self.ctx.sleep(max(0, int(milliseconds)))
        self.check_cancel()
        return self

    def locator(self, target: Any, rect: Any = None) -> BvLocator:
        target = _unwrap(target)
        if isinstance(target, str):
            roi = _rect(rect)
            ro = RecognitionObject.ocr(*(roi or (0, 0, 1920, 1080)))
            locator = BvLocator(
                self.ctx, ro, to_collection=self.to_collection,
                check_cancel=self.check_cancel, text=target,
            )
        elif isinstance(target, RecognitionObject):
            locator = BvLocator(
                self.ctx, target, to_collection=self.to_collection,
                check_cancel=self.check_cancel,
            )
        else:
            raise TypeError("BvPage.Locator 需要文字或 RecognitionObject")
        locator.timeout = self.default_timeout
        locator.retry_interval = self.default_retry_interval
        return locator

    def get_by_text(self, text: str = "", rect: Any = None) -> BvLocator:
        return self.locator(str(text), rect)

    def get_by_any_text(self, texts: Any, rect: Any = None) -> BvLocator:
        values = [str(value) for value in texts]
        if not values or any(not value.strip() for value in values):
            raise ValueError("候选文本不能为空")
        roi = _rect(rect)
        ro = RecognitionObject.ocr(*(roi or (0, 0, 1920, 1080)))
        locator = BvLocator(
            self.ctx, ro, to_collection=self.to_collection,
            check_cancel=self.check_cancel, any_texts=list(dict.fromkeys(values)),
        )
        locator.timeout = self.default_timeout
        locator.retry_interval = self.default_retry_interval
        return locator

    def ocr(self, rect: Any = None):
        return self.get_by_text("", rect).find_all()

    def click(self, x: float, y: float) -> None:
        self.ctx.input.click_ref(float(x), float(y))

    Screenshot = screenshot
    Wait = wait
    Locator = locator
    getByText = get_by_text
    GetByText = get_by_text
    getByAnyText = get_by_any_text
    GetByAnyText = get_by_any_text
    Ocr = ocr
    Click = click
