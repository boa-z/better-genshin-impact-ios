from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest

from bgi_touch.engine.bv import BvLocator
from bgi_touch.engine.recognition import Mat, RecognitionObject


def _context():
    from bgi_touch.vision.coordinate import ScreenTransform

    input_simulator = SimpleNamespace(
        click_ref=Mock(), key_press=Mock(), tap_button=Mock(), drag_ref=Mock(),
    )
    ctx = SimpleNamespace(
        input=input_simulator,
        transform=ScreenTransform(1920, 1080),
        capture_bgr=Mock(return_value=np.zeros((1080, 1920, 3), dtype=np.uint8)),
        sleep=Mock(),
    )
    return ctx, input_simulator

class _SequenceLocator(BvLocator):
    """A BvLocator whose matches change without involving OpenCV/OCR."""

    def __init__(self, ctx, matches):
        super().__init__(
            ctx,
            RecognitionObject.template_match(
                Mat(np.zeros((2, 2, 3), dtype=np.uint8)),
            ),
        )
        self._matches = list(matches)
        self.calls = 0

    def _find_all_in(self, _screen):
        self.calls += 1
        if self._matches:
            return self._matches.pop(0)
        return []

    def clone(self):
        return self


def _locator(ctx, matches):
    """Return a real BvLocator instance with deterministic frame results."""
    sequence = _SequenceLocator(ctx, matches)
    assert isinstance(sequence, BvLocator)
    return sequence, sequence


def _region(ctx, x=100, y=200, width=40, height=20):
    from bgi_touch.engine.recognition import Region

    return Region(ctx, x, y, width, height)


def test_flow_waits_on_one_shared_frame_and_clicks_last_match_center():
    from bgi_touch.engine.bv import BvPage

    ctx, input_simulator = _context()
    target, state = _locator(ctx, [[_region(ctx)]])

    BvPage(ctx).Flow().WithDefaultTimeout(100).WithDefaultRetryInterval(1) \
        .WaitUntil(target).Click().Run()

    assert state.calls == 1
    ctx.capture_bgr.assert_called_once_with()
    input_simulator.click_ref.assert_called_once_with(120.0, 210.0)


@pytest.mark.parametrize("recognition_type", ["ColorMatch", "ColorRangeAndOcr"])
def test_bv_locator_accepts_upstream_color_recognition_types(recognition_type):
    ctx, _input_simulator = _context()
    ro = RecognitionObject()
    ro.RecognitionType = recognition_type
    locator = BvLocator(ctx, ro, text="目标")
    expected = _region(ctx, 30, 40, 20, 10, )
    other = _region(ctx, 80, 90, 20, 10)
    expected.text = "目标"
    other.text = "其他"
    screen = SimpleNamespace(
        find_multi=Mock(return_value=[expected, other]),
    )

    assert locator._find_all_in(screen) == [expected]
    screen.find_multi.assert_called_once_with(locator.recognition_object, limit=100)


def test_flow_action_until_repeats_action_after_retry_interval():
    from bgi_touch.engine.bv import BvPage

    ctx, input_simulator = _context()
    target, state = _locator(ctx, [[], [_region(ctx, 20, 30, 10, 10)]])

    BvPage(ctx).Flow().WithDefaultTimeout(1000).WithDefaultRetryInterval(1) \
        .KeyPress("E").WithTimeout(1000).WithRetryInterval(1).Until(target).Run()

    assert state.calls == 2
    assert input_simulator.key_press.call_args_list == [call("E"), call("E")]
    assert ctx.sleep.call_args_list == [call(1), call(1)]


def test_flow_any_and_all_disappear_only_capture_once_per_poll():
    from bgi_touch.engine.bv import BvPage

    ctx, _input_simulator = _context()
    first, first_state = _locator(ctx, [[]])
    second, second_state = _locator(ctx, [[_region(ctx, 50, 60, 10, 10)]])

    BvPage(ctx).Flow().WithDefaultTimeout(100).WithDefaultRetryInterval(1) \
        .WaitUntilAny([first, second]).Run()

    assert first_state.calls == 1
    assert second_state.calls == 1
    ctx.capture_bgr.assert_called_once_with()

    ctx.capture_bgr.reset_mock()
    disappearing, disappearing_state = _locator(
        ctx, [[_region(ctx)], []],
    )
    BvPage(ctx).Flow().WithDefaultTimeout(1000).WithDefaultRetryInterval(1) \
        .WaitUntilDisappear(disappearing).Run()

    assert disappearing_state.calls == 2
    assert ctx.capture_bgr.call_count == 2


def test_flow_action_supports_wait_until_disappear_and_do_is_parameterless():
    from bgi_touch.engine.bv import BvPage

    ctx, input_simulator = _context()
    target, state = _locator(ctx, [[_region(ctx)], []])
    callback = Mock()

    BvPage(ctx).Flow().Do(callback).WaitUntilDisappear(
        target, timeout=1000, retry_interval=1,
    ).Run()

    callback.assert_called_once_with()
    assert state.calls == 2
    assert input_simulator.click_ref.call_count == 0


def test_flow_mouse_buttons_follow_touch_semantics_without_spurious_left_tap():
    from bgi_touch.engine.bv import BvPage

    ctx, input_simulator = _context()
    page = BvPage(ctx)

    page.Flow().RightClick(100, 200).Run()
    page.Flow().MiddleClick(300, 400).Run()

    assert input_simulator.key_press.call_args_list == [call("LSHIFT")]
    input_simulator.tap_button.assert_called_once_with("elementalSight")
    input_simulator.click_ref.assert_not_called()


def test_js_bv_flow_chain_and_do_callback_are_exposed(tmp_path):
    pytest.importorskip("pythonmonkey")

    from bgi_touch.engine.js_runtime import JsScriptRuntime
    from bgi_touch.vision.coordinate import ScreenTransform

    (tmp_path / "main.js").write_text(
        """
const page = new BvPage();
let calls = 0;
page.Flow()
  .Click(11, 22)
  .KeyPress('E')
  .Do(() => { calls += 1; })
  .Wait(0)
  .Run();
return JSON.stringify({calls});
""",
        encoding="utf-8",
    )
    ctx, input_simulator = _context()
    ctx.device = SimpleNamespace(paste_text=Mock(), tap=Mock())
    ctx.transform = ScreenTransform(1920, 1080)

    result = JsScriptRuntime(ctx, tmp_path).run()

    assert result == '{"calls":1}'
    input_simulator.click_ref.assert_called_once_with(11.0, 22.0)
    input_simulator.key_press.assert_called_once_with("E")
