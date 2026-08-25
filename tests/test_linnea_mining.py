from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np

from bgi_touch.pathing.linnea_mining import (
    LinneaMiningTask,
    MineralCluster,
    cluster_minerals,
    parse_linnea_mining_params,
)
from bgi_touch.vision.yolo import Detection


def _detection(x, y, width=40, height=40, confidence=0.9, class_id=0):
    return Detection(x, y, width, height, confidence, class_id)


def test_linnea_params_match_upstream_forms_and_round_floor():
    assert parse_linnea_mining_params("") == (1, 1)
    assert parse_linnea_mining_params("3,10") == (3, 10)
    assert parse_linnea_mining_params("mines=4,rounds=2") == (4, 4)
    assert parse_linnea_mining_params("mines=0,rounds=1000") == (1, 999)


def test_mineral_cluster_rejects_area_outliers_and_prefers_right_near_target():
    cluster = MineralCluster(_detection(100, 100), prefer_right=True)
    assert cluster.try_add(_detection(145, 100))
    assert not cluster.try_add(_detection(300, 100, width=200, height=200))
    assert cluster.target_x == 165
    assert cluster.target_y == 120


def test_cluster_minerals_uses_scale_aware_distance():
    detections = [_detection(100, 100), _detection(480, 100)]

    assert len(cluster_minerals(detections, width_scale=1.0)) == 2
    scaled = [_detection(200, 100, 80, 80), _detection(900, 100, 80, 80)]
    assert len(cluster_minerals(scaled, width_scale=2.0)) == 1


def test_linnea_task_scans_centered_ore_and_restores_input_state():
    input_simulator = SimpleNamespace(
        key_press=Mock(),
        button_down=Mock(),
        button_up=Mock(),
        move_camera_by=Mock(),
        attack=Mock(),
    )
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    predictor = SimpleNamespace(
        predict=Mock(return_value=[_detection(940, 520, 40, 40)])
    )
    ctx = SimpleNamespace(input=input_simulator, sleep=Mock(), capture_bgr=Mock(return_value=frame))

    task = LinneaMiningTask(ctx, predictor=predictor, log=Mock())

    assert task.run()
    assert input_simulator.key_press.call_args_list == [call("R"), call("R")]
    assert input_simulator.button_down.call_args_list[0] == call("elementalSight")
    assert input_simulator.button_up.call_args_list[-1] == call("elementalSight")
    input_simulator.attack.assert_called_once_with()
    predictor.predict.assert_called_once()


def test_pathing_linnea_action_switches_to_linnea_and_passes_parameters():
    from bgi_touch.pathing.actions import PathingActionRunner
    from bgi_touch.pathing.model import Waypoint

    ctx = SimpleNamespace(
        input=SimpleNamespace(key_press=Mock()),
        sleep=Mock(),
        party_slots={"Linnea": 2},
    )
    runner = PathingActionRunner(ctx, log=Mock())
    runner.combat = Mock()

    with patch("bgi_touch.pathing.linnea_mining.LinneaMiningTask") as task_type:
        task_type.return_value.run.return_value = True
        assert runner.run(Waypoint(1, 0, 0, "target", "walk", "linnea_mining", "2,4"))

    runner.combat.switch_to.assert_called_once_with("Linnea")
    task_type.assert_called_once()
    assert task_type.call_args.kwargs["mine_count"] == 2
    assert task_type.call_args.kwargs["scan_rounds"] == 4


def test_dispatcher_linnea_entrypoint_accepts_route_and_object_parameters():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    ctx = SimpleNamespace(party_slots={"Linnea": 2})
    with patch("bgi_touch.tasks.dispatcher.CombatExecutor") as executor_type, \
         patch("bgi_touch.pathing.linnea_mining.LinneaMiningTask") as task_type:
        task_type.return_value.run.return_value = True
        dispatcher = TaskDispatcher(ctx, party_slots={"Linnea": 2}, log=Mock())

        assert dispatcher.run_task({
            "name": "LinneaMining",
            "config": {"actionParams": "mines=2,rounds=4"},
        })
        assert dispatcher.run_linnea_mining_task({
            "mineCount": 3,
            "scanRounds": 2,
        })

    assert executor_type.for_context.call_count == 2
    assert executor_type.for_context.return_value.switch_to.call_args_list == [
        call("莉奈娅"),
        call("莉奈娅"),
    ]
    assert task_type.call_args_list[0].kwargs["mine_count"] == 2
    assert task_type.call_args_list[0].kwargs["scan_rounds"] == 4
    assert task_type.call_args_list[1].kwargs["mine_count"] == 3
    assert task_type.call_args_list[1].kwargs["scan_rounds"] == 3
