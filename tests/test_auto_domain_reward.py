from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch


class Hit:
    def __init__(self, exists: bool):
        self._exists = exists

    def is_exist(self) -> bool:
        return self._exists


def make_task(*states: bool):
    from bgi_touch.tasks.auto_domain import AutoDomainTask, REWARD_RESULT_EXIT

    regions = []
    for state in states:
        region = SimpleNamespace(find=Mock(return_value=Hit(state)))
        regions.append(region)
    ctx = SimpleNamespace(
        capture_region=Mock(side_effect=regions),
        sleep=Mock(),
    )
    task = object.__new__(AutoDomainTask)
    task.ctx = ctx
    task.log = Mock()
    return task, ctx, regions, REWARD_RESULT_EXIT


def test_reward_recognition_waits_for_stable_exit_control():
    task, ctx, regions, ready_template = make_task(False, False, True)

    assert task._wait_for_reward_result_ready() is True

    assert ctx.capture_region.call_count == 3
    assert ctx.sleep.call_args_list == [((300,),), ((300,),)]
    for region in regions:
        region.find.assert_called_once_with(ready_template)


def test_reward_recognition_ready_wait_honors_cancellation_without_capture():
    task, ctx, _regions, _ready_template = make_task()

    assert task._wait_for_reward_result_ready(lambda: True) is False

    ctx.capture_region.assert_not_called()
    ctx.sleep.assert_not_called()


def test_reward_recognition_ready_wait_times_out_after_upstream_retry_budget():
    task, ctx, _regions, _ready_template = make_task(*([False] * 20))

    assert task._wait_for_reward_result_ready() is False

    assert ctx.capture_region.call_count == 20
    assert ctx.sleep.call_count == 20


def test_auto_domain_run_pauses_and_restores_realtime_triggers():
    from bgi_touch.tasks.auto_domain import AutoDomainTask

    loop = Mock()
    ctx = SimpleNamespace(triggers=loop)
    task = object.__new__(AutoDomainTask)
    task.ctx = ctx
    task._run_impl = Mock(return_value={"摩拉": 1200})

    scope = MagicMock()
    scope.__enter__.return_value = None
    scope.__exit__.return_value = False
    with patch(
        "bgi_touch.tasks.auto_domain.exclusive_realtime_triggers",
        return_value=scope,
    ) as _scope:
        # The context-manager mock above verifies ownership at the task edge;
        # the lower-level helper contract is covered by common-job tests.
        result = task.run()

    assert result == {"摩拉": 1200}
    task._run_impl.assert_called_once_with(None)
    _scope.assert_called_once_with(ctx)
    scope.__enter__.assert_called_once_with()
    scope.__exit__.assert_called_once()
