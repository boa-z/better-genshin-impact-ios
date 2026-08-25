from types import SimpleNamespace
from unittest.mock import Mock, call, patch


class _Hit:
    def __init__(self, text):
        self.text = text
        self.click = Mock()


class _Region:
    def __init__(self, handbook, claim, task):
        self.handbook = handbook
        self.claim = claim
        self.task = task

    def find_multi(self, recognition, *, limit):
        assert limit in (20, 40)
        if recognition is self.task._handbook_ocr:
            return self.handbook
        assert recognition is self.task._claim_ocr
        return self.claim


def _task(frames):
    from bgi_touch.tasks.claim_encounter_rewards import (
        ClaimEncounterPointsRewardsTask,
    )

    ctx = SimpleNamespace(
        input=SimpleNamespace(key_press=Mock()),
        capture_region=Mock(),
        sleep=Mock(),
    )
    return_main = Mock(return_value=True)
    task = ClaimEncounterPointsRewardsTask(
        ctx,
        timeout_s=4,
        return_main_ui=return_main,
        log=Mock(),
    )
    ctx.capture_region.side_effect = [
        _Region(handbook, claim, task)
        for handbook, claim in frames
    ]
    return task, ctx, return_main


def test_claim_encounter_rewards_selects_commission_and_claims_once():
    tab = _Hit("委托")
    claim = _Hit("领取")
    task, ctx, return_main = _task([
        ([tab], []),
        ([tab], [claim]),
    ])

    result = task.run()

    assert result == {
        "ok": True,
        "opened": True,
        "claimed": True,
        "alreadyClaimed": False,
    }
    ctx.input.key_press.assert_called_once_with("F1")
    tab.click.assert_called_once_with()
    claim.click.assert_called_once_with()
    return_main.assert_has_calls([call(), call()])


def test_claim_encounter_rewards_reports_already_claimed_without_clicking_claim():
    tab = _Hit("委托")
    status = _Hit("今日奖励已领取")
    task, ctx, return_main = _task([
        ([tab, status], []),
    ])

    result = task.run()

    assert result == {
        "ok": True,
        "opened": True,
        "claimed": False,
        "alreadyClaimed": True,
    }
    tab.click.assert_not_called()
    ctx.input.key_press.assert_called_once_with("F1")
    return_main.assert_has_calls([call(), call()])


def test_claim_encounter_rewards_does_not_click_status_text_in_claim_roi():
    status = _Hit("今日奖励已领取")
    task, ctx, _return_main = _task([
        ([_Hit("委托")], [status]),
    ])

    result = task.run()

    assert result["alreadyClaimed"] is True
    status.click.assert_not_called()
    ctx.input.key_press.assert_called_once_with("F1")


def test_claim_encounter_rewards_cancellation_does_not_open_handbook():
    task, ctx, return_main = _task([])

    result = task.run(cancelled=lambda: True)

    assert result == {
        "ok": False,
        "opened": False,
        "claimed": False,
        "alreadyClaimed": False,
        "cancelled": True,
    }
    ctx.input.key_press.assert_not_called()
    return_main.assert_not_called()


def test_claim_encounter_rewards_keeps_cleanup_inside_exclusive_scope():
    task, _ctx, _return_main = _task([
        ([_Hit("委托")], [_Hit("领取")]),
    ])
    events = []

    class Scope:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *_args):
            events.append("exit")

    with patch(
        "bgi_touch.tasks.claim_encounter_rewards.exclusive_realtime_triggers",
        return_value=Scope(),
    ) as isolated:
        result = task.run()

    assert result["ok"] is True
    isolated.assert_called_once_with(task.ctx)
    # return_main_ui is called once before F1 and once for cleanup; both happen
    # before the exclusive scope is released.
    assert events == ["enter", "exit"]


def test_dispatcher_passes_encounter_reward_timeout_to_genshin_api():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    api = Mock()
    api.claimEncounterPointsRewards.return_value = True
    dispatcher = TaskDispatcher(object())
    dispatcher._genshin_api = Mock(return_value=api)

    assert dispatcher.run_task({
        "name": "ClaimEncounterPointsRewardsTask",
        "config": {"timeoutSeconds": 7},
    }) is True
    api.claimEncounterPointsRewards.assert_called_once_with(7.0)
