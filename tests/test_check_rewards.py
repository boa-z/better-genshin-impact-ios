from types import SimpleNamespace
from unittest.mock import Mock, call


class _Hit:
    def __init__(self, text):
        self.text = text
        self.click = Mock()


class _Region:
    def __init__(self, hits):
        self.hits = hits

    def find_multi(self, _recognition, *, limit):
        assert limit == 80
        return self.hits


def _task(monkeypatch, frames, *, notifications=None):
    import bgi_touch.tasks.check_rewards as module

    monkeypatch.setattr(module.time, "monotonic", lambda: 0.0)
    from bgi_touch.tasks.check_rewards import CheckRewardsTask

    ctx = SimpleNamespace(
        capture_region=Mock(side_effect=[_Region(frame) for frame in frames]),
        input=SimpleNamespace(key_press=Mock()),
        sleep=Mock(),
    )
    return_main = Mock(return_value=True)
    task = CheckRewardsTask(
        ctx,
        timeout_s=4,
        notification_service=notifications,
        return_main_ui=return_main,
        log=Mock(),
    )
    return task, ctx, return_main


def test_check_rewards_opens_commission_page_and_notifies_claimed(monkeypatch):
    notifications = SimpleNamespace(notify=Mock())
    tab = _Hit("委托")
    task, ctx, return_main = _task(
        monkeypatch,
        [[tab], [_Hit("每日委托奖励")], [_Hit("今日奖励已领取")]],
        notifications=notifications,
    )

    result = task.run()

    assert result == {"checked": True, "claimed": True, "pageOpened": True}
    ctx.input.key_press.assert_called_once_with("F1")
    tab.click.assert_called_once_with()
    return_main.assert_has_calls([call(), call()])
    notifications.notify.assert_called_once_with(
        "daily.reward", "检查每日奖励：已领取", result="Success",
    )


def test_check_rewards_reports_unclaimed_without_failing_page_cleanup(monkeypatch):
    notifications = SimpleNamespace(notify=Mock())
    task, ctx, return_main = _task(
        monkeypatch,
        [[_Hit("每日委托奖励")]],
        notifications=notifications,
    )
    # Deadline 4s is reached before the status poll, so this test does not
    # need a fake OCR frame for the unclaimed state.
    import bgi_touch.tasks.check_rewards as module
    monotonic = iter((0.0, 0.0, 4.0))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic))

    result = task.run()

    assert result == {"checked": True, "claimed": False, "pageOpened": True}
    assert ctx.input.key_press.call_args_list == [call("F1")]
    return_main.assert_has_calls([call(), call()])
    notifications.notify.assert_called_once_with(
        "daily.reward", "检查到每日奖励未领取，请手动查看！", result="Fail",
    )


def test_check_rewards_cancellation_does_not_open_handbook(monkeypatch):
    task, ctx, return_main = _task(monkeypatch, [[]])

    result = task.run(cancelled=lambda: True)

    assert result == {"checked": False, "claimed": None, "cancelled": True}
    ctx.input.key_press.assert_not_called()
    return_main.assert_not_called()


def test_dispatcher_exposes_check_rewards_alias_and_notification_service(monkeypatch):
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    task = Mock()
    task.run.return_value = {"checked": True, "claimed": True}
    task_type = Mock(return_value=task)
    monkeypatch.setattr("bgi_touch.tasks.check_rewards.CheckRewardsTask", task_type)
    service = object()
    dispatcher = TaskDispatcher(object(), notification_service=service)

    assert dispatcher.run_task({
        "name": "CheckRewardsTask",
        "config": {"timeoutSeconds": 7},
    }) == {"checked": True, "claimed": True}
    assert task_type.call_args.kwargs["timeout_s"] == 7.0
    assert task_type.call_args.kwargs["notification_service"] is service
    task.run.assert_called_once()
