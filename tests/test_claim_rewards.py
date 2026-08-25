from types import SimpleNamespace
from unittest.mock import Mock, call, patch


def _ctx():
    return SimpleNamespace(
        input=SimpleNamespace(key_press=Mock(), click_ref=Mock()),
        sleep=Mock(),
    )


def test_claim_button_prefers_explicit_claim_all_and_ignores_claimed_status():
    from bgi_touch.tasks.claim_rewards import _ClaimRewardsTask

    claimed = SimpleNamespace(text="已领取")
    generic = SimpleNamespace(text="领取")
    explicit = SimpleNamespace(text="一键领取")

    assert _ClaimRewardsTask._find_claim_button(
        [claimed, generic, explicit]
    ) is explicit
    assert _ClaimRewardsTask._find_claim_button([claimed]) is None


def test_claim_battle_pass_runs_both_tabs_inside_trigger_exclusive_scope():
    from bgi_touch.tasks.claim_rewards import (
        ClaimBattlePassRewardsTask,
        ClaimPageResult,
    )

    ctx = _ctx()
    return_main_ui = Mock(side_effect=[True, True])
    events = []

    class Scope:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *_args):
            events.append("exit")

    task = ClaimBattlePassRewardsTask(
        ctx,
        return_main_ui=return_main_ui,
        log=Mock(),
    )
    task._wait_click_page = Mock(return_value=True)
    task._claim_current_page = Mock(side_effect=[
        ClaimPageResult(clicked=True, popup_closed=True),
        ClaimPageResult(clicked=False),
    ])

    with patch(
        "bgi_touch.tasks.claim_rewards.exclusive_realtime_triggers",
        return_value=Scope(),
    ) as isolated:
        result = task.run()

    assert result["ok"] is True
    assert result["opened"] is True
    assert result["claimed"] is True
    assert result["returned"] is True
    assert events == ["enter", "exit"]
    isolated.assert_called_once_with(ctx)
    assert ctx.input.key_press.call_args_list == [call("ESCAPE")]
    assert ctx.input.click_ref.call_args_list == [
        call(960, 45), call(858, 45),
    ]
    assert task._claim_current_page.call_count == 2


def test_claim_battle_pass_stops_after_manual_selection_and_returns_home():
    from bgi_touch.tasks.claim_rewards import (
        ClaimBattlePassRewardsTask,
        ClaimPageResult,
    )

    ctx = _ctx()
    return_main_ui = Mock(side_effect=[True, True])
    task = ClaimBattlePassRewardsTask(
        ctx,
        return_main_ui=return_main_ui,
        log=Mock(),
    )
    task._wait_click_page = Mock(return_value=True)
    task._claim_current_page = Mock(return_value=ClaimPageResult(
        clicked=True,
        manual_selection=True,
    ))

    result = task.run()

    assert result["claimed"] is True
    assert result["manualSelection"] is True
    # A selection dialog must not be followed by a click on the second tab.
    assert ctx.input.click_ref.call_args_list == [call(960, 45)]
    assert return_main_ui.call_count == 2


def test_claim_mail_uses_template_fallback_and_closes_page():
    from bgi_touch.tasks.claim_rewards import ClaimMailRewardsTask, ClaimPageResult

    ctx = _ctx()
    return_main_ui = Mock(side_effect=[True, True])
    task = ClaimMailRewardsTask(
        ctx,
        return_main_ui=return_main_ui,
        log=Mock(),
    )
    task._wait_click_page = Mock(return_value=True)
    task._claim_current_page = Mock(return_value=ClaimPageResult(clicked=True))

    result = task.run()

    assert result["ok"] is True
    assert result["claimed"] is True
    assert ctx.input.key_press.call_args_list == [
        call("ESCAPE"), call("ESCAPE"),
    ]
    task._wait_click_page.assert_called_once()
    assert task._wait_click_page.call_args.kwargs["template_name"] == "mail"


def test_claim_post_click_closes_primogem_popup_using_same_frame_probe():
    from bgi_touch.tasks.claim_rewards import ClaimMailRewardsTask

    ctx = _ctx()
    task = ClaimMailRewardsTask(ctx, return_main_ui=Mock(), log=Mock())
    click = SimpleNamespace(text="全部领取", click=Mock())
    region = object()
    task._scan = Mock(side_effect=[
        (region, [], [click]),
        (region, [SimpleNamespace(text="原石")], []),
    ])
    task._is_manual_selection = Mock(return_value=False)

    result = task._claim_current_page(10**12)

    assert result.clicked is True
    assert result.popup_closed is True
    click.click.assert_called_once_with()
    assert ctx.input.key_press.call_args_list == [call("ESCAPE")]
    assert task._scan.call_count == 2
