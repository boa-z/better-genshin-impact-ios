from types import SimpleNamespace
from unittest.mock import Mock, patch


def test_wonderland_cycle_requires_all_four_state_transitions():
    from bgi_touch.tasks.wonderland import WonderlandCycleTask

    task = WonderlandCycleTask(SimpleNamespace(), timeout_s=20, log=Mock())
    task._open_realm_list = Mock(return_value=True)
    task._select_realm = Mock(return_value=True)
    task._wait_main_ui = Mock(side_effect=[True, True])
    task._return_to_teyvat = Mock(return_value=True)

    assert task.run()
    task._open_realm_list.assert_called_once()
    task._select_realm.assert_called_once()
    task._return_to_teyvat.assert_called_once()
    assert task._wait_main_ui.call_count == 2


def test_wonderland_cycle_stops_before_input_when_cancelled():
    from bgi_touch.tasks.wonderland import WonderlandCycleTask

    task = WonderlandCycleTask(SimpleNamespace(), timeout_s=20, log=Mock())
    task._open_realm_list = Mock()

    assert not task.run(cancelled=lambda: True)
    task._open_realm_list.assert_not_called()


def test_dispatcher_passes_cancellation_callback_to_wonderland_job():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    dispatcher = TaskDispatcher(SimpleNamespace(), log=Mock())
    with patch("bgi_touch.tasks.wonderland.WonderlandCycleTask") as task_type:
        task_type.return_value.run.return_value = True
        assert dispatcher.run_wonderland_cycle_task(ct=lambda: False)

    task_type.return_value.run.assert_called_once()
    assert callable(task_type.return_value.run.call_args.kwargs["cancelled"])

