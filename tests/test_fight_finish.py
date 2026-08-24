from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


def test_fight_finish_config_parses_latest_bettergi_options():
    from bgi_touch.combat.finish import FightFinishConfig

    config = FightFinishConfig.from_mapping({
        "fastCheckEnabled": True,
        "fastCheckParams": "3;白术;钟离;",
        "checkAfterSwitchAvatar": True,
        "checkEndDelay": "0.5;钟离,1.2;",
        "beforeDetectDelay": "0.7",
        "skipFightEndCheckWhenEnemyVisible": True,
        "blockCheckBeforeBattleSeconds": 2.5,
        "paimonEndCheckEnabled": True,
        "paimonEndCheckDelay": 0.02,
    })

    assert config.fast_check_enabled is True
    assert config.check_interval_s == 3
    assert config.check_names == frozenset({"白术", "钟离"})
    assert config.check_after_switch_avatar is True
    assert config.delay_ms == 500
    assert config.character_delay_ms == {"钟离": 1200}
    assert config.detect_delay_ms == 700
    assert config.skip_when_enemy_visible is True
    assert config.block_after_start_s == 2.5
    # Upstream clamps the probe delay to 50..400 ms.
    assert config.paimon_check_delay_ms == 50


def test_fight_finish_block_window_prevents_device_probe():
    from bgi_touch.combat.finish import FightFinishConfig, FightFinishDetector

    now = [10.0]
    ctx = SimpleNamespace(input=SimpleNamespace(key_press=Mock()))
    detector = FightFinishDetector(
        ctx,
        FightFinishConfig(block_after_start_s=3),
        clock=lambda: now[0],
    )
    detector.start_battle()
    now[0] = 12.0

    assert not detector.check()
    ctx.input.key_press.assert_not_called()
    assert detector.last_check_at == 12.0


def test_visible_enemy_skips_party_probe_but_forced_check_is_bounded(monkeypatch):
    from bgi_touch.combat import finish

    ctx = SimpleNamespace(input=SimpleNamespace(key_press=Mock()))
    detector = finish.FightFinishDetector(
        ctx,
        finish.FightFinishConfig(
            skip_when_enemy_visible=True,
            max_enemy_skip_checks=2,
        ),
    )
    monkeypatch.setattr(finish, "enemies_nearby", lambda _ctx: True)
    monkeypatch.setattr(detector, "_capture_after", lambda *_args: "main")
    monkeypatch.setattr(finish, "is_main_ui", lambda _ctx, frame: frame == "main")
    ctx.sleep = Mock()
    ctx.capture_bgr = Mock(return_value="main")
    ctx.device = SimpleNamespace(last_frame_version=1)

    assert not detector.check()
    assert not detector.check()
    ctx.input.key_press.assert_not_called()
    # The third call forces the definitive party-screen probe.
    assert not detector.check()
    ctx.input.key_press.assert_called_once_with("L")


def test_party_screen_probe_distinguishes_active_combat_and_finished_battle(monkeypatch):
    from bgi_touch.combat import finish

    monkeypatch.setattr(finish, "is_main_ui", lambda _ctx, frame: frame == "main")
    monkeypatch.setattr(finish, "is_party_setup_open", lambda _ctx, frame: frame == "party")

    def context(frame):
        return SimpleNamespace(
            input=SimpleNamespace(key_press=Mock()),
            device=SimpleNamespace(last_frame_version=4),
            sleep=Mock(),
            capture_bgr_after_frame=Mock(return_value=frame),
            capture_bgr=Mock(return_value=frame),
        )

    active = context("main")
    assert not finish.FightFinishDetector(active).check()
    active.input.key_press.assert_called_once_with("L")

    ended = context("party")
    assert finish.FightFinishDetector(ended).check()
    assert ended.input.key_press.call_args_list[0].args == ("L",)
    assert ended.input.key_press.call_args_list[1].args == ("X",)


def test_party_setup_signature_requires_yellow_bar_and_white_tile():
    from bgi_touch.combat.finish import is_party_setup_open
    from bgi_touch.vision.coordinate import ScreenTransform

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[49:52, 789:792] = (20, 230, 240)  # yellow in BGR
    frame[49:52, 767:770] = (250, 250, 250)
    ctx = SimpleNamespace(transform=ScreenTransform(1920, 1080))

    assert is_party_setup_open(ctx, frame)
    frame[49:52, 767:770] = 0
    assert not is_party_setup_open(ctx, frame)


def test_auto_fight_fast_check_after_switch_skips_character_actions(tmp_path: Path):
    from bgi_touch.tasks.auto_fight import AutoFightTask

    strategy = tmp_path / "fight.txt"
    strategy.write_text("钟离 e, attack(1)", encoding="utf-8")
    ctx = SimpleNamespace(input=SimpleNamespace(release_all=Mock()))
    executor = Mock()
    detector = Mock()
    detector.config = SimpleNamespace(check_after_switch_avatar=True)
    detector.should_fast_check.return_value = True
    detector.check.return_value = True

    with patch("bgi_touch.tasks.auto_fight.CombatExecutor.for_context", return_value=executor), \
            patch("bgi_touch.tasks.auto_fight.FightFinishDetector", return_value=detector):
        task = AutoFightTask(ctx, str(strategy), timeout_s=10)
        assert task.run()

    executor.switch_to.assert_called_once_with("钟离")
    executor.exec.assert_not_called()
    detector.check.assert_called_once()
    ctx.input.release_all.assert_called_once_with()


def test_auto_fight_dispatcher_maps_finish_detection_config():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    with patch("bgi_touch.tasks.auto_fight.AutoFightTask") as task:
        task.return_value.run.return_value = True
        assert TaskDispatcher(object()).run_auto_fight_task({
            "fightFinishDetectEnabled": True,
            "finishDetectConfig": {
                "fastCheckEnabled": True,
                "blockCheckBeforeBattleSeconds": 4,
            },
        })

    assert task.call_args.kwargs["fight_finish_detect_enabled"] is True
    config = task.call_args.kwargs["finish_detect_config"]
    assert config.fast_check_enabled is True
    assert config.block_after_start_s == 4
