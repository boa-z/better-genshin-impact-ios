from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch


class Hit:
    def __init__(self, exists: bool):
        self._exists = exists
        self.click = Mock()

    def is_exist(self):
        return self._exists


class Capture:
    def __init__(self, hit):
        self.hit = hit

    def find(self, _recognition):
        return self.hit


def _context(hits):
    captures = [Capture(hit) for hit in hits]
    return SimpleNamespace(
        capture_region=Mock(side_effect=captures),
        sleep=Mock(),
        input=SimpleNamespace(key_press=Mock(), release_all=Mock()),
    )


def test_one_key_expedition_collects_retries_and_redispatches():
    from bgi_touch.tasks.expedition import OneKeyExpeditionTask

    collect = Hit(True)
    missing = Hit(False)
    redispatch = Hit(True)
    ctx = _context([collect, missing, redispatch])

    assert OneKeyExpeditionTask(ctx).run()
    collect.click.assert_called_once_with()
    missing.click.assert_not_called()
    redispatch.click.assert_called_once_with()
    ctx.input.key_press.assert_called_once_with("ESCAPE")
    ctx.input.release_all.assert_called_once_with()


def test_one_key_expedition_leaves_page_unchanged_when_nothing_completed():
    from bgi_touch.tasks.expedition import OneKeyExpeditionTask

    ctx = _context([Hit(False), Hit(False)])
    assert not OneKeyExpeditionTask(ctx).run()
    ctx.input.key_press.assert_not_called()
    ctx.input.release_all.assert_called_once_with()


def test_one_key_expedition_honors_cancellation_before_capture():
    from bgi_touch.tasks.expedition import OneKeyExpeditionTask

    ctx = _context([])
    assert not OneKeyExpeditionTask(ctx).run(cancelled=lambda: True)
    ctx.capture_region.assert_not_called()
    ctx.input.release_all.assert_called_once_with()


def test_dispatcher_maps_one_key_expedition_and_autoskip_config():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    ctx = Mock()
    with patch("bgi_touch.tasks.expedition.OneKeyExpeditionTask") as task:
        task.return_value.run.return_value = True
        dispatcher = TaskDispatcher(ctx)
        assert dispatcher.run_task({
            "name": "OneKeyExpedition",
            "config": {"redispatchRetries": 5, "closePage": False},
        })

    assert task.call_args.kwargs["redispatch_retries"] == 5
    assert task.call_args.kwargs["close_page"] is False

    TaskDispatcher(ctx).add_timer(SimpleNamespace(
        name="AutoSkip",
        config={"autoReExploreEnabled": "false"},
    ))
    assert ctx.enable_trigger.call_args.kwargs["auto_re_explore_enabled"] is False
