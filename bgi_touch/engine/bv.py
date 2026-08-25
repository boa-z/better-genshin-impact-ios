"""Portable BetterGI BvPage/BvLocator JavaScript vision API."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Any, Callable

from .recognition import ImageRegion, Mat, RecognitionObject, Region


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


class BvImage:
    """BetterGI image locator descriptor backed by a RecognitionObject."""

    def __init__(self, template_asset: str, asset_resolver: Callable[[str], Any],
                 roi: Any = None, threshold: float = 0.8):
        name = str(template_asset)
        resolved = _unwrap(asset_resolver(name))
        if isinstance(resolved, (str, bytes)):
            resolved = Mat.from_file(str(resolved))
        if not isinstance(resolved, Mat):
            raise TypeError("BvImage 素材解析器必须返回 Mat 或文件路径")
        self.recognition_object = RecognitionObject.template_match(resolved)
        self.recognition_object.name = name
        self.recognition_object.roi = _rect(roi)
        self.recognition_object.threshold = float(threshold)

    recognitionObject = property(lambda self: self.recognition_object)
    RecognitionObject = recognitionObject

    def to_recognition_object(self) -> RecognitionObject:
        return self.recognition_object

    toRecognitionObject = to_recognition_object
    ToRecognitionObject = to_recognition_object


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
        return self._find_all_in(screen)

    def _find_all_in(self, screen: ImageRegion) -> list[Region]:
        """Find this locator in an already captured frame.

        ``BvFlow`` shares one frame between all of its targets.  This keeps an
        ``Any``/``All`` check internally consistent when the game UI changes
        between two captures and avoids unnecessary screenshot requests.
        """
        self.check_cancel()
        ro = self.recognition_object
        if ro.recognition_type == "TemplateMatch":
            found = screen.find(ro)
            return [found] if found.is_exist() else []
        if ro.recognition_type not in (
            "Ocr",
            "OcrMatch",
            "ColorMatch",
            "ColorRangeAndOcr",
        ):
            raise NotImplementedError(
                f"BvLocator 不支持识别类型 {ro.recognition_type}"
            )
        # ImageRegion already owns the OpenCV/color/OCR implementation.  Do
        # not duplicate it here: BvLocator only adds collection conversion
        # and optional text filtering on top of the shared frame operation.
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


@dataclass
class _FlowStep:
    description: str
    operation: Callable[["_FlowContext"], Any] | None = None
    targets: tuple[BvLocator, ...] = ()
    condition: str = "once"
    timeout: int = 0
    retry_interval: int = 0


@dataclass
class _FlowContext:
    last_match: Region | None = None


class BvFlow:
    """Portable synchronous implementation of BetterGI's BvFlow chain."""

    ANY_APPEAR = "any_appear"
    ALL_DISAPPEAR = "all_disappear"

    def __init__(self, page: "BvPage", default_timeout: int,
                 default_retry_interval: int):
        self.page = page
        self.default_timeout = self._positive(default_timeout, "defaultTimeout")
        self.default_retry_interval = self._positive(
            default_retry_interval, "defaultRetryInterval",
        )
        self._steps: list[_FlowStep] = []
        self._started = False
        self._running = False

    @staticmethod
    def _positive(value: int, name: str) -> int:
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} 必须大于 0")
        return value

    def _ensure_mutable(self) -> None:
        if self._started:
            raise RuntimeError("BvFlow 已经开始执行，不能再添加步骤")

    def with_default_timeout(self, milliseconds: int) -> "BvFlow":
        self._ensure_mutable()
        self._ensure_defaults_can_change()
        self.default_timeout = self._positive(milliseconds, "timeout")
        return self

    def with_default_retry_interval(self, milliseconds: int) -> "BvFlow":
        self._ensure_mutable()
        self._ensure_defaults_can_change()
        self.default_retry_interval = self._positive(milliseconds, "retryInterval")
        return self

    def _ensure_defaults_can_change(self) -> None:
        if self._steps:
            raise RuntimeError("添加流程步骤后不能修改流程默认配置")

    def _add(self, step: _FlowStep) -> "BvFlow":
        self._ensure_mutable()
        self._steps.append(step)
        return self

    def _action(self, description: str, operation: Callable[[_FlowContext], Any]) -> "BvFlowAction":
        self._ensure_mutable()
        return BvFlowAction(self, description, operation)

    def _add_once(self, description: str, operation: Callable[[_FlowContext], Any]) -> "BvFlow":
        return self._add(_FlowStep(description, operation=operation))

    def _add_condition(
        self,
        description: str,
        operation: Callable[[_FlowContext], Any] | None,
        targets: list[BvLocator],
        condition: str,
        timeout: int | None = None,
        retry_interval: int | None = None,
    ) -> "BvFlow":
        actual_timeout = self._positive(
            timeout if timeout is not None else self.default_timeout,
            "timeout",
        )
        actual_interval = self._positive(
            retry_interval if retry_interval is not None else self.default_retry_interval,
            "retryInterval",
        )
        return self._add(_FlowStep(
            description,
            operation=operation,
            targets=tuple(target.clone() for target in targets),
            condition=condition,
            timeout=actual_timeout,
            retry_interval=actual_interval,
        ))

    @staticmethod
    def _require_locator(value: Any) -> BvLocator:
        value = _unwrap(value)
        if not isinstance(value, BvLocator):
            raise TypeError("BvFlow 目标必须是 BvLocator")
        return value

    @classmethod
    def _locator_list(cls, values: Any, name: str = "targets") -> list[BvLocator]:
        values = _unwrap(values)
        if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
            raise TypeError(f"{name} 必须是 BvLocator 数组")
        result = [cls._require_locator(value) for value in values]
        if not result:
            raise ValueError(f"{name} 不能为空")
        return result

    @staticmethod
    def _point(value: Region | None) -> tuple[float, float]:
        if value is None or value.is_empty():
            raise RuntimeError("没有可用的上一步识别位置，无法执行隐式坐标操作")
        return value.x + value.width / 2, value.y + value.height / 2

    def _click(self, context: _FlowContext, x=None, y=None) -> None:
        if x is None and y is None:
            x, y = self._point(context.last_match)
        elif x is None or y is None:
            raise TypeError("Click 需要同时提供 x 和 y")
        pointer = getattr(self.page.ctx, "_script_pointer", None)
        if pointer is not None:
            pointer.click_at_ref(float(x), float(y))
        else:
            self.page.ctx.input.click_ref(float(x), float(y))

    def _right_click(self, context: _FlowContext, x=None, y=None) -> None:
        if x is not None or y is not None:
            self._move_to(context, x, y)
        # BetterGI's right mouse button is sprint on the supported Genshin
        # profiles.  It must not be emulated by a left tap followed by Shift:
        # that would activate a menu item before sprinting.
        self.page.ctx.input.key_press("LSHIFT")

    def _middle_click(self, context: _FlowContext, x=None, y=None) -> None:
        if x is not None or y is not None:
            self._move_to(context, x, y)
        self.page.ctx.input.tap_button("elementalSight")

    def _move_to(self, context: _FlowContext, x=None, y=None) -> None:
        if x is None and y is None:
            x, y = self._point(context.last_match)
        elif x is None or y is None:
            raise TypeError("MoveTo 需要同时提供 x 和 y")
        pointer = getattr(self.page.ctx, "_script_pointer", None)
        if pointer is not None:
            pointer.move_to_ref(float(x), float(y))

    def _drag(self, _context: _FlowContext, x1, y1, x2, y2, duration=300) -> None:
        self.page.ctx.input.drag_ref(
            float(x1), float(y1), float(x2), float(y2),
            duration_ms=self._positive(duration, "duration"),
        )

    def _invoke(self, callback: Any, context: _FlowContext) -> None:
        callback = _unwrap(callback)
        if not callable(callback):
            raise TypeError("BvFlow.Do 需要函数")
        # The upstream delegate is parameterless. Keep the execution context
        # private so Python and ClearScript callbacks share one convention.
        callback()

    def key_press(self, key: str) -> "BvFlowAction":
        return self._action(
            f"KeyPress({key})",
            lambda _context: self.page.ctx.input.key_press(str(key)),
        )

    def click(self, x=None, y=None) -> "BvFlowAction":
        return self._action(
            "Click(previous)" if x is None and y is None else f"Click({x}, {y})",
            lambda context: self._click(context, x, y),
        )

    def right_click(self, x=None, y=None) -> "BvFlowAction":
        return self._action(
            "RightClick(previous)" if x is None and y is None else f"RightClick({x}, {y})",
            lambda context: self._right_click(context, x, y),
        )

    def middle_click(self, x=None, y=None) -> "BvFlowAction":
        return self._action(
            "MiddleClick(previous)" if x is None and y is None else f"MiddleClick({x}, {y})",
            lambda context: self._middle_click(context, x, y),
        )

    def move_to(self, x=None, y=None) -> "BvFlowAction":
        return self._action(
            "MoveTo(previous)" if x is None and y is None else f"MoveTo({x}, {y})",
            lambda context: self._move_to(context, x, y),
        )

    def drag(self, x1, y1, x2, y2, duration=300) -> "BvFlowAction":
        self._positive(duration, "duration")
        return self._action(
            f"Drag({x1}, {y1}, {x2}, {y2})",
            lambda _context: self._drag(_context, x1, y1, x2, y2, duration),
        )

    def drag_to(self, x, y, duration=300) -> "BvFlowAction":
        self._positive(duration, "duration")
        return self._action(
            f"DragTo({x}, {y})",
            lambda context: self._drag(
                context, *self._point(context.last_match), x, y, duration,
            ),
        )

    def drag_from(self, x, y, duration=300) -> "BvFlowAction":
        self._positive(duration, "duration")
        return self._action(
            f"DragFrom({x}, {y})",
            lambda context: self._drag(
                context, x, y, *self._point(context.last_match), duration,
            ),
        )

    def do(self, callback: Any) -> "BvFlowAction":
        return self._action("Do", lambda context: self._invoke(callback, context))

    def wait_until_text(self, text: str, rect: Any = None,
                        timeout: int | None = None,
                        retry_interval: int | None = None) -> "BvFlow":
        return self.wait_until(
            self.page.get_by_text(str(text), rect), timeout, retry_interval,
        )

    def wait_until_any_text(self, texts: Any, rect: Any = None,
                            timeout: int | None = None,
                            retry_interval: int | None = None) -> "BvFlow":
        return self.wait_until(
            self.page.get_by_any_text(texts, rect), timeout, retry_interval,
        )

    def wait_until(self, target: Any, timeout: int | None = None,
                   retry_interval: int | None = None) -> "BvFlow":
        return self._add_condition(
            "WaitUntil", None, [self._require_locator(target)],
            self.ANY_APPEAR, timeout, retry_interval,
        )

    def wait_until_any(self, targets: Any, timeout: int | None = None,
                       retry_interval: int | None = None) -> "BvFlow":
        return self._add_condition(
            "WaitUntilAny", None, self._locator_list(targets),
            self.ANY_APPEAR, timeout, retry_interval,
        )

    def wait_until_disappear(self, target: Any, timeout: int | None = None,
                             retry_interval: int | None = None) -> "BvFlow":
        return self._add_condition(
            "WaitUntilDisappear", None, [self._require_locator(target)],
            self.ALL_DISAPPEAR, timeout, retry_interval,
        )

    def wait_until_all_disappear(self, targets: Any, timeout: int | None = None,
                                 retry_interval: int | None = None) -> "BvFlow":
        return self._add_condition(
            "WaitUntilAllDisappear", None, self._locator_list(targets),
            self.ALL_DISAPPEAR, timeout, retry_interval,
        )

    def wait(self, milliseconds: int) -> "BvFlow":
        milliseconds = int(milliseconds)
        if milliseconds < 0:
            raise ValueError("milliseconds 不能小于 0")
        return self._add(_FlowStep(
            f"Wait({milliseconds})",
            operation=lambda _context: self.page.wait(milliseconds),
        ))

    def _find_targets(self, targets: tuple[BvLocator, ...], condition: str,
                      context: _FlowContext) -> bool:
        screen = ImageRegion(self.page.ctx, self.page.ctx.capture_bgr())
        for locator in targets:
            values = locator._find_all_in(screen)
            if condition == self.ANY_APPEAR and values:
                context.last_match = values[0]
                return True
            if condition == self.ALL_DISAPPEAR and values:
                return False
        if condition == self.ALL_DISAPPEAR:
            context.last_match = None
            return True
        return False

    def _run_condition(self, step: _FlowStep, context: _FlowContext) -> None:
        deadline = time.monotonic() + step.timeout / 1000
        while True:
            self.page.check_cancel()
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{step.description} 超时（{step.timeout}ms）")
            if self._find_targets(step.targets, step.condition, context):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{step.description} 超时（{step.timeout}ms）")
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{step.description} 超时（{step.timeout}ms）")
            self.page.wait(min(
                step.retry_interval, max(1, math.ceil(remaining * 1000)),
            ))

    def _run_action_condition(self, step: _FlowStep, context: _FlowContext) -> None:
        deadline = time.monotonic() + step.timeout / 1000
        while True:
            self.page.check_cancel()
            if step.operation is not None:
                step.operation(context)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{step.description} 超时（{step.timeout}ms）")
            self.page.wait(min(
                step.retry_interval, max(1, math.ceil(remaining * 1000)),
            ))
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{step.description} 超时（{step.timeout}ms）")
            if self._find_targets(step.targets, step.condition, context):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{step.description} 超时（{step.timeout}ms）")
                return

    def run(self) -> "BvPage":
        if self._running:
            raise RuntimeError("同一个 BvFlow 不能并发执行")
        self._started = True
        self._running = True
        context = _FlowContext()
        try:
            for step in self._steps:
                self.page.check_cancel()
                if step.condition == "once":
                    if step.operation is not None:
                        step.operation(context)
                elif step.condition == self.ANY_APPEAR or step.condition == self.ALL_DISAPPEAR:
                    if step.operation is None:
                        self._run_condition(step, context)
                    else:
                        self._run_action_condition(step, context)
                else:
                    raise RuntimeError(f"未知 BvFlow 条件: {step.condition}")
            return self.page
        finally:
            self._running = False

    WithDefaultTimeout = with_default_timeout
    WithDefaultRetryInterval = with_default_retry_interval
    KeyPress = key_press
    Click = click
    RightClick = right_click
    MiddleClick = middle_click
    MoveTo = move_to
    Drag = drag
    DragTo = drag_to
    DragFrom = drag_from
    Do = do
    WaitUntilText = wait_until_text
    WaitUntilAnyText = wait_until_any_text
    WaitUntil = wait_until
    WaitUntilAny = wait_until_any
    WaitUntilDisappear = wait_until_disappear
    WaitUntilAllDisappear = wait_until_all_disappear
    Wait = wait
    Run = run


class BvFlowAction:
    """An operation awaiting a terminal ``Until``/``Run`` condition."""

    def __init__(self, flow: BvFlow, description: str,
                 operation: Callable[[_FlowContext], Any]):
        self.flow = flow
        self.description = description
        self.operation = operation
        self.timeout: int | None = None
        self.retry_interval: int | None = None
        self.completed = False

    def _ensure_open(self) -> None:
        if self.completed:
            raise RuntimeError("当前动作已经设置完成条件，不能再次修改或添加")

    def with_timeout(self, milliseconds: int) -> "BvFlowAction":
        self._ensure_open()
        self.timeout = self.flow._positive(milliseconds, "timeout")
        return self

    def with_retry_interval(self, milliseconds: int) -> "BvFlowAction":
        self._ensure_open()
        self.retry_interval = self.flow._positive(milliseconds, "retryInterval")
        return self

    def _once(self) -> BvFlow:
        self._ensure_open()
        if self.timeout is not None or self.retry_interval is not None:
            raise RuntimeError(
                "一次性动作不支持 WithTimeout 或 WithRetryInterval，请使用 Until 系列方法设置重试条件"
            )
        self.completed = True
        return self.flow._add_once(self.description, self.operation)

    def _condition(self, targets: list[BvLocator], condition: str) -> BvFlow:
        self._ensure_open()
        self.completed = True
        return self.flow._add_condition(
            self.description, self.operation, targets, condition,
            self.timeout, self.retry_interval,
        )

    def do(self, callback: Any) -> "BvFlowAction":
        return self._once().do(callback)

    def key_press(self, key: str) -> "BvFlowAction":
        return self._once().key_press(key)

    def click(self, x=None, y=None) -> "BvFlowAction":
        return self._once().click(x, y)

    def right_click(self, x=None, y=None) -> "BvFlowAction":
        return self._once().right_click(x, y)

    def middle_click(self, x=None, y=None) -> "BvFlowAction":
        return self._once().middle_click(x, y)

    def move_to(self, x=None, y=None) -> "BvFlowAction":
        return self._once().move_to(x, y)

    def drag(self, x1, y1, x2, y2, duration=300) -> "BvFlowAction":
        return self._once().drag(x1, y1, x2, y2, duration)

    def drag_to(self, x, y, duration=300) -> "BvFlowAction":
        return self._once().drag_to(x, y, duration)

    def drag_from(self, x, y, duration=300) -> "BvFlowAction":
        return self._once().drag_from(x, y, duration)

    def wait(self, milliseconds: int) -> BvFlow:
        return self._once().wait(milliseconds)

    def until(self, target: Any) -> BvFlow:
        return self._condition([self.flow._require_locator(target)], BvFlow.ANY_APPEAR)

    def until_any(self, targets: Any) -> BvFlow:
        return self._condition(self.flow._locator_list(targets), BvFlow.ANY_APPEAR)

    def until_disappear(self, target: Any) -> BvFlow:
        return self._condition([self.flow._require_locator(target)], BvFlow.ALL_DISAPPEAR)

    def until_all_disappear(self, targets: Any) -> BvFlow:
        return self._condition(self.flow._locator_list(targets), BvFlow.ALL_DISAPPEAR)

    def until_text(self, text: str, rect: Any = None) -> BvFlow:
        return self.until(self.flow.page.get_by_text(str(text), rect))

    def until_any_text(self, texts: Any, rect: Any = None) -> BvFlow:
        return self.until(self.flow.page.get_by_any_text(texts, rect))

    def wait_until(self, target: Any, timeout: int | None = None,
                   retry_interval: int | None = None) -> BvFlow:
        self._ensure_open()
        self.flow._positive(timeout, "timeout") if timeout is not None else None
        self.flow._positive(retry_interval, "retryInterval") if retry_interval is not None else None
        return self._once().wait_until(target, timeout, retry_interval)

    def wait_until_any(self, targets: Any, timeout: int | None = None,
                       retry_interval: int | None = None) -> BvFlow:
        self._ensure_open()
        self.flow._positive(timeout, "timeout") if timeout is not None else None
        self.flow._positive(retry_interval, "retryInterval") if retry_interval is not None else None
        return self._once().wait_until_any(targets, timeout, retry_interval)

    def wait_until_disappear(self, target: Any, timeout: int | None = None,
                             retry_interval: int | None = None) -> BvFlow:
        self._ensure_open()
        self.flow._positive(timeout, "timeout") if timeout is not None else None
        self.flow._positive(retry_interval, "retryInterval") if retry_interval is not None else None
        return self._once().wait_until_disappear(target, timeout, retry_interval)

    def wait_until_all_disappear(self, targets: Any, timeout: int | None = None,
                                 retry_interval: int | None = None) -> BvFlow:
        self._ensure_open()
        self.flow._positive(timeout, "timeout") if timeout is not None else None
        self.flow._positive(retry_interval, "retryInterval") if retry_interval is not None else None
        return self._once().wait_until_all_disappear(targets, timeout, retry_interval)

    def wait_until_text(self, text: str, rect: Any = None,
                        timeout: int | None = None,
                        retry_interval: int | None = None) -> BvFlow:
        return self.wait_until(
            self.flow.page.get_by_text(str(text), rect), timeout, retry_interval,
        )

    def wait_until_any_text(self, texts: Any, rect: Any = None,
                            timeout: int | None = None,
                            retry_interval: int | None = None) -> BvFlow:
        return self.wait_until(
            self.flow.page.get_by_any_text(texts, rect), timeout, retry_interval,
        )

    def run(self) -> BvPage:
        return self._once().run()

    WithTimeout = with_timeout
    WithRetryInterval = with_retry_interval
    Do = do
    KeyPress = key_press
    Click = click
    RightClick = right_click
    MiddleClick = middle_click
    MoveTo = move_to
    Drag = drag
    DragTo = drag_to
    DragFrom = drag_from
    Wait = wait
    Until = until
    UntilAny = until_any
    UntilDisappear = until_disappear
    UntilAllDisappear = until_all_disappear
    UntilText = until_text
    UntilAnyText = until_any_text
    WaitUntil = wait_until
    WaitUntilAny = wait_until_any
    WaitUntilDisappear = wait_until_disappear
    WaitUntilAllDisappear = wait_until_all_disappear
    WaitUntilText = wait_until_text
    WaitUntilAnyText = wait_until_any_text
    Run = run


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

    def flow(self) -> BvFlow:
        self.check_cancel()
        return BvFlow(self, self.default_timeout, self.default_retry_interval)

    def locator(self, target: Any, rect: Any = None) -> BvLocator:
        target = _unwrap(target)
        if isinstance(target, str):
            roi = _rect(rect)
            ro = RecognitionObject.ocr(*(roi or (0, 0, 1920, 1080)))
            locator = BvLocator(
                self.ctx, ro, to_collection=self.to_collection,
                check_cancel=self.check_cancel, text=target,
            )
        elif isinstance(target, BvImage):
            locator = BvLocator(
                self.ctx, target.to_recognition_object(),
                to_collection=self.to_collection,
                check_cancel=self.check_cancel,
            )
        elif isinstance(target, RecognitionObject):
            locator = BvLocator(
                self.ctx, target, to_collection=self.to_collection,
                check_cancel=self.check_cancel,
            )
        else:
            raise TypeError("BvPage.Locator 需要文字、BvImage 或 RecognitionObject")
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
    Flow = flow
    Locator = locator
    getByText = get_by_text
    GetByText = get_by_text
    getByAnyText = get_by_any_text
    GetByAnyText = get_by_any_text
    Ocr = ocr
    Click = click
