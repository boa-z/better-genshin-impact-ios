from types import SimpleNamespace
from unittest.mock import Mock, patch


class TriggerLoop:
    def __init__(self):
        self.pause = Mock(return_value=([], False, 1))
        self.resume = Mock()


def _context():
    loop = TriggerLoop()
    ctx = SimpleNamespace(
        triggers=loop,
        capture_region=Mock(return_value=SimpleNamespace()),
        sleep=Mock(),
        input=SimpleNamespace(
            click_ref=Mock(), key_press=Mock(), release_all=Mock(),
        ),
    )
    return ctx, loop


def test_quick_buy_isolates_screenshot_and_input_flow():
    from bgi_touch.tasks.quick_buy import QuickBuyTask

    ctx, loop = _context()
    task = QuickBuyTask.__new__(QuickBuyTask)
    task.ctx = ctx
    task.serenitea = False
    task.log = Mock()
    task._coin = None
    task._is_serenitea_shop = Mock(return_value=False)
    task._swipe_ref = Mock()

    assert task.run()
    loop.pause.assert_called_once_with()
    loop.resume.assert_called_once_with(([], False, 1))


def test_quick_claim_isolates_all_polling_frames():
    from bgi_touch.tasks.quick_claim import QuickClaimRewardTask

    ctx, loop = _context()
    task = QuickClaimRewardTask.__new__(QuickClaimRewardTask)
    task.ctx = ctx
    task.max_clicks = 1
    task.scroll_down = False
    task.max_scrolls = 0
    task.timeout_s = 2
    task.log = Mock()
    task._find_candidates = Mock(return_value=[])

    assert task.run() == 0
    task._find_candidates.assert_called_once_with(ctx.capture_region.return_value)
    loop.pause.assert_called_once_with()
    loop.resume.assert_called_once_with(([], False, 1))


def test_quick_serenitea_and_redeem_use_the_same_exclusive_scope():
    from bgi_touch.tasks.quick_serenitea import QuickSereniteaPotTask
    from bgi_touch.tasks.redeem_code import UseRedemptionCodeTask

    ctx, loop = _context()
    quick = QuickSereniteaPotTask.__new__(QuickSereniteaPotTask)
    quick.ctx = ctx
    quick._run_locked = Mock(return_value=True)
    assert quick.run()

    with patch("bgi_touch.tasks.redeem_code.GenshinApi") as api_type:
        api_type.return_value.returnMainUi.return_value = True
        redeem = UseRedemptionCodeTask.__new__(UseRedemptionCodeTask)
        redeem.ctx = ctx
        redeem.codes = (SimpleNamespace(code="ABC", items=None),)
        redeem.timeout_s = 20
        redeem.log = Mock()
        redeem._open_redeem_dialog = Mock(return_value=True)
        redeem._use_one = Mock(return_value=True)
        assert redeem.run() == {"ABC": True}

    assert loop.pause.call_count == 2
    assert loop.resume.call_count == 2
    assert [item.args for item in loop.resume.call_args_list] == [
        (([], False, 1),),
        (([], False, 1),),
    ]


def test_auto_stygian_isolates_event_menu_and_reward_flow():
    from bgi_touch.tasks.auto_stygian import AutoStygianOnslaughtTask

    ctx, loop = _context()
    task = AutoStygianOnslaughtTask.__new__(AutoStygianOnslaughtTask)
    task.ctx = ctx
    task._run_impl = Mock(return_value=True)

    assert task.run() is True
    task._run_impl.assert_called_once_with(None)
    loop.pause.assert_called_once_with()
    loop.resume.assert_called_once_with(([], False, 1))
