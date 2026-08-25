import json
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

import pytest


def _api_with_task(task):
    from bgi_touch.engine.genshin_api import GenshinApi

    api = GenshinApi.__new__(GenshinApi)
    api.ctx = SimpleNamespace()
    api.log = Mock()
    api._tp_for = Mock(return_value=task)
    api._positioner_for = Mock()
    return api


def test_genshin_tp_forwards_force_to_touch_task():
    task = MagicMock()
    task.tp.return_value = True
    api = _api_with_task(task)

    assert api.tp(4328, 3960, "Teyvat", True)

    task.tp.assert_called_once_with(4328.0, 3960.0, force=True)
    api._positioner_for.return_value.set_prior.assert_called_once_with(4328.0, 3960.0)


def test_genshin_tp_supports_the_three_argument_force_overload():
    task = MagicMock()
    task.tp.return_value = True
    api = _api_with_task(task)

    assert api.tp(4328, 3960, True)

    api._tp_for.assert_called_once_with(None)
    task.tp.assert_called_once_with(4328.0, 3960.0, force=True)
    api._positioner_for.return_value.set_prior.assert_called_once_with(4328.0, 3960.0)


def test_genshin_tp_decodes_string_force_values():
    task = MagicMock()
    task.tp.return_value = True
    api = _api_with_task(task)

    assert api.tp(4328, 3960, "Teyvat", "false")

    task.tp.assert_called_once_with(4328.0, 3960.0, force=False)


def test_genshin_tp_accepts_string_coordinate_overload():
    task = MagicMock()
    task.tp.return_value = True
    api = _api_with_task(task)

    assert api.tp("4328.25", "3960.75")

    task.tp.assert_called_once_with(4328.25, 3960.75, force=False)


def test_genshin_tp_invalid_string_coordinate_matches_try_parse_zero():
    task = MagicMock()
    task.tp.return_value = True
    api = _api_with_task(task)

    assert api.tp("not-a-number", "")

    task.tp.assert_called_once_with(0.0, 0.0, force=False)


def test_genshin_tp_does_not_retry_when_point_panel_is_not_opened():
    from bgi_touch.pathing.tp import TeleportPanelNotOpenedError

    task = MagicMock()
    task.tp.side_effect = TeleportPanelNotOpenedError(
        "传送失败：点击传送点后未出现交互面板，可能是传送点未激活"
    )
    api = _api_with_task(task)

    with pytest.raises(TeleportPanelNotOpenedError):
        api.tp(4328, 3960, "Teyvat")

    task.tp.assert_called_once_with(4328.0, 3960.0, force=False)
    api.log.assert_not_called()


def test_teleport_candidate_text_is_short_and_normalized():
    from bgi_touch.pathing.tp import TpTask

    assert TpTask._is_anchor_entry_text("傳送錨點 >")
    assert TpTask._is_anchor_entry_text("七天神像")
    assert not TpTask._is_anchor_entry_text("传送锚点·优兰尼娅湖")


def test_genshin_uid_matches_bettergi_numeric_return_contract():
    from types import SimpleNamespace
    from unittest.mock import patch

    from bgi_touch.engine.genshin_api import GenshinApi

    api = GenshinApi.__new__(GenshinApi)
    api.ctx = SimpleNamespace(capture_bgr=lambda: object())
    ocr = SimpleNamespace(recognize=lambda _frame: [
        SimpleNamespace(text="UID: 123456789"),
    ])
    with patch("bgi_touch.engine.genshin_api.get_ocr", return_value=ocr):
        assert api.uid() == 123456789


def test_choose_talk_option_skips_non_orange_duplicate_and_clicks_orange():
    from bgi_touch.engine.genshin_api import GenshinApi

    non_orange = SimpleNamespace(text="领取奖励", click=Mock())
    orange = SimpleNamespace(text="领取奖励", click=Mock())
    region = SimpleNamespace(find_multi=Mock(return_value=[non_orange, orange]))
    api = GenshinApi.__new__(GenshinApi)
    api.ctx = SimpleNamespace(
        capture_region=Mock(return_value=region),
        input=SimpleNamespace(click_ref=Mock()),
        sleep=Mock(),
    )
    api._talk_option_region = Mock(side_effect=[object(), object()])

    with patch.object(
        GenshinApi, "_is_orange_option", side_effect=[False, True]
    ) as is_orange:
        assert api.chooseTalkOption("领取奖励", skip_times=1, is_orange=True)

    non_orange.click.assert_not_called()
    orange.click.assert_called_once_with()
    assert is_orange.call_count == 2
    api.ctx.input.click_ref.assert_not_called()


def test_choose_talk_option_decodes_string_is_orange_flag():
    from bgi_touch.engine.genshin_api import GenshinApi

    first = SimpleNamespace(text="领取奖励", click=Mock())
    region = SimpleNamespace(find_multi=Mock(return_value=[first]))
    api = GenshinApi.__new__(GenshinApi)
    api.ctx = SimpleNamespace(
        capture_region=Mock(return_value=region),
        input=SimpleNamespace(click_ref=Mock()),
        sleep=Mock(),
    )
    api._talk_option_region = Mock()

    assert api.chooseTalkOption("领取奖励", skip_times=1, is_orange="false")

    first.click.assert_called_once_with()
    api._talk_option_region.assert_not_called()


def test_choose_talk_option_does_not_advance_non_orange_matching_option():
    from bgi_touch.engine.genshin_api import GenshinApi

    non_orange = SimpleNamespace(text="每日", click=Mock())
    region = SimpleNamespace(find_multi=Mock(return_value=[non_orange]))
    api = GenshinApi.__new__(GenshinApi)
    api.ctx = SimpleNamespace(
        capture_region=Mock(return_value=region),
        input=SimpleNamespace(click_ref=Mock()),
        sleep=Mock(),
    )
    api._talk_option_region = Mock(return_value=object())

    with patch.object(GenshinApi, "_is_orange_option", return_value=False):
        assert not api.chooseTalkOption("每日", skip_times=5, is_orange=True)

    non_orange.click.assert_not_called()
    api.ctx.input.click_ref.assert_not_called()
    api.ctx.sleep.assert_not_called()


def test_genshin_menu_ocr_reuses_active_trigger_frame():
    from bgi_touch.engine.genshin_api import GenshinApi

    cached = object()
    direct_capture = Mock()
    loop = SimpleNamespace(active=True, interval=0.7)
    api = GenshinApi.__new__(GenshinApi)
    api.ctx = SimpleNamespace(
        _trigger_loop=loop,
        cached_frame=Mock(return_value=(cached, 0.2)),
        capture_region=direct_capture,
    )

    with patch("bgi_touch.engine.recognition.ImageRegion") as region_type:
        region_type.return_value = object()
        assert api._text_capture_region() is region_type.return_value

    direct_capture.assert_not_called()
    region_type.assert_called_once_with(api.ctx, cached)


def test_genshin_menu_ocr_uses_fresh_capture_without_active_trigger():
    from bgi_touch.engine.genshin_api import GenshinApi

    region = object()
    direct_capture = Mock(return_value=region)
    api = GenshinApi.__new__(GenshinApi)
    api.ctx = SimpleNamespace(capture_region=direct_capture)

    assert api._text_capture_region() is region
    direct_capture.assert_called_once_with()


def test_is_orange_option_matches_bettergi_hsv_ratio_threshold():
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from bgi_touch.engine.genshin_api import GenshinApi

    orange = np.zeros((10, 10, 3), dtype=np.uint8)
    orange[:2, :] = (0, 165, 255)  # BGR, HSV hue ~= 19
    boundary = np.zeros_like(orange)
    boundary[:1, :] = (0, 165, 255)
    blue = np.full_like(orange, (255, 0, 0))

    assert GenshinApi._is_orange_option(orange)
    assert not GenshinApi._is_orange_option(boundary)
    assert not GenshinApi._is_orange_option(blue)


def test_click_chat_exit_until_main_ui_reuses_frame_and_clicks_exit():
    from bgi_touch.engine.genshin_api import GenshinApi

    exit_hit = SimpleNamespace(is_exist=Mock(return_value=True), click=Mock())
    frame_one, frame_two = object(), object()
    api = GenshinApi.__new__(GenshinApi)
    api.ctx = SimpleNamespace(
        capture_bgr=Mock(side_effect=[frame_one, frame_two]),
        sleep=Mock(),
        input=SimpleNamespace(key_press=Mock()),
    )
    api.log = Mock()
    api._is_main_ui = Mock(side_effect=[False, True])
    api._find_chat_exit = Mock(return_value=exit_hit)
    api._is_talk_ui_frame = Mock()

    assert api.clickChatExitUntilMainUi(2)

    exit_hit.click.assert_called_once_with()
    api._find_chat_exit.assert_called_once_with(frame_one)
    api._is_talk_ui_frame.assert_not_called()
    api.ctx.input.key_press.assert_not_called()
    assert api.ctx.sleep.call_args_list == [call(200), call(500)]


def test_click_chat_exit_advances_text_only_talk_step():
    from bgi_touch.engine.genshin_api import GenshinApi

    empty_hit = SimpleNamespace(is_exist=Mock(return_value=False))
    frame_one, frame_two = object(), object()
    api = GenshinApi.__new__(GenshinApi)
    api.ctx = SimpleNamespace(
        capture_bgr=Mock(side_effect=[frame_one, frame_two]),
        sleep=Mock(),
        input=SimpleNamespace(key_press=Mock()),
    )
    api.log = Mock()
    api._is_main_ui = Mock(side_effect=[False, True])
    api._find_chat_exit = Mock(return_value=empty_hit)
    api._is_talk_ui_frame = Mock(return_value=True)

    assert api.clickChatExitUntilMainUi(2)

    api._find_chat_exit.assert_called_once_with(frame_one)
    api._is_talk_ui_frame.assert_called_once_with(frame_one)
    api.ctx.input.key_press.assert_called_once_with("SPACE")
    api.ctx.sleep.assert_called_once_with(500)


def test_return_main_ui_uses_escape_and_recognized_exit_door():
    from bgi_touch.engine.genshin_api import GenshinApi

    exit_hit = SimpleNamespace(is_exist=Mock(return_value=True), click=Mock())
    frame_initial, frame_after_escape, frame_after_door = object(), object(), object()
    api = GenshinApi.__new__(GenshinApi)
    api.ctx = SimpleNamespace(
        capture_bgr=Mock(side_effect=[
            frame_initial, frame_after_escape, frame_after_door,
        ]),
        sleep=Mock(),
        input=SimpleNamespace(key_press=Mock()),
    )
    api.log = Mock()
    api._is_main_ui = Mock(side_effect=[False, True])
    api._find_exit_door = Mock(return_value=exit_hit)

    assert api.returnMainUi(max_tries=1)

    api.ctx.input.key_press.assert_called_once_with("ESCAPE")
    api.ctx.sleep.assert_has_calls([call(900), call(5000)])
    exit_hit.click.assert_called_once_with()
    api._find_exit_door.assert_called_once_with(frame_after_escape)


def test_return_main_ui_keyboard_fallback_is_confirmed():
    from bgi_touch.engine.genshin_api import GenshinApi

    frames = [object(), object(), object()]
    api = GenshinApi.__new__(GenshinApi)
    api.ctx = SimpleNamespace(
        capture_bgr=Mock(side_effect=frames),
        sleep=Mock(),
        input=SimpleNamespace(key_press=Mock()),
    )
    api.log = Mock()
    api._is_main_ui = Mock(return_value=False)
    api._find_exit_door = Mock(
        return_value=SimpleNamespace(is_exist=Mock(return_value=False)),
    )

    assert not api.returnMainUi(max_tries=1)

    assert api.ctx.input.key_press.call_args_list == [
        call("ESCAPE"), call("ENTER"), call("ESCAPE"),
    ]
    api.log.assert_called_once()


def test_genshin_exposes_native_capture_metrics_while_using_reference_coordinates():
    from bgi_touch.engine.genshin_api import GenshinApi
    from bgi_touch.vision.coordinate import ScreenTransform

    api = GenshinApi.__new__(GenshinApi)
    api.ctx = SimpleNamespace(transform=ScreenTransform(2816, 1296))

    assert api.width == 2816
    assert api.height == 1296
    assert api.scaleTo1080PRatio == pytest.approx(1.2)


def test_genshin_metrics_fall_back_to_reference_space_without_a_context_transform():
    from bgi_touch.engine.genshin_api import GenshinApi

    api = GenshinApi.__new__(GenshinApi)
    api.ctx = SimpleNamespace()

    assert (api.width, api.height) == (1920, 1080)
    assert api.scaleTo1080PRatio == 1.0


def test_genshin_big_map_position_returns_none_when_view_is_not_located():
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    from bgi_touch.engine.genshin_api import GenshinApi

    api = GenshinApi.__new__(GenshinApi)
    api.ctx = SimpleNamespace(capture_bgr=Mock(return_value=object()))
    api._big_locators = {}
    api._big_locator = None
    locator = SimpleNamespace(locate_view=Mock(return_value=None))
    with patch("bgi_touch.pathing.tp.BigMapLocator", return_value=locator):
        assert api.getPositionFromBigMap() is None


def test_genshin_exposes_lazy_navigation_instance_with_pixel_contract():
    from bgi_touch.engine.genshin_api import GenshinApi, NavigationInstanceApi

    api = GenshinApi.__new__(GenshinApi)
    api._navigation_instance = None
    first = api.lazyNavigationInstance
    second = api.lazyNavigationInstance

    assert isinstance(first, NavigationInstanceApi)
    assert first is second


def test_navigation_instance_routes_matching_to_positioner_pixel_coordinates():
    from bgi_touch.engine.genshin_api import GenshinApi

    positioner = SimpleNamespace(
        get_position_pixel=Mock(return_value=(123.5, 456.25)),
    )
    api = GenshinApi.__new__(GenshinApi)
    api.ctx = SimpleNamespace(capture_bgr=Mock())
    api._positioners = {}
    api._positioner_for = Mock(return_value=positioner)
    api.log = Mock()

    point = api.lazyNavigationInstance.getPosition(
        SimpleNamespace(bgr=object()), "Teyvat", "SIFT"
    )

    assert (point.x, point.y) == (123.5, 456.25)
    api._positioner_for.assert_called_once_with("Teyvat")
    positioner.get_position_pixel.assert_called_once()
    api.ctx.capture_bgr.assert_not_called()


def test_genshin_matching_method_single_argument_is_not_used_as_map_name():
    from bgi_touch.engine.genshin_api import GenshinApi

    api = GenshinApi.__new__(GenshinApi)
    api.log = Mock()
    api.getPositionFromMap = Mock(return_value="position")

    assert api.getPositionFromMapWithMatchingMethod("SIFT") == "position"
    api.getPositionFromMap.assert_called_once_with("Teyvat", 900)


def test_genshin_matching_method_full_overload_preserves_map_and_cache():
    from bgi_touch.engine.genshin_api import GenshinApi

    api = GenshinApi.__new__(GenshinApi)
    api.log = Mock()
    api.getPositionFromMap = Mock(return_value="position")

    assert api.getPositionFromMapWithMatchingMethod("层岩巨渊", "SIFT", 1200) == "position"
    api.getPositionFromMap.assert_called_once_with("层岩巨渊", 1200)


def test_genshin_local_position_overload_uses_pixel_match_not_stable_cache():
    from bgi_touch.engine.genshin_api import GenshinApi

    frame = object()
    positioner = SimpleNamespace(
        set_prior=Mock(),
        get_position_pixel=Mock(return_value=(100.0, 200.0)),
        locator=SimpleNamespace(
            config=SimpleNamespace(
                image_to_world=lambda x, y: (321.5, 654.25),
            ),
        ),
    )
    api = GenshinApi.__new__(GenshinApi)
    api.ctx = SimpleNamespace(capture_bgr=Mock(return_value=frame))
    api._positioner_for = Mock(return_value=positioner)

    point = api.getPositionFromMap("Teyvat", "12.5", "24.5")

    assert (point.x, point.y) == (321.5, 654.25)
    positioner.set_prior.assert_called_once_with(12.5, 24.5)
    positioner.get_position_pixel.assert_called_once_with(frame)


def test_genshin_position_cache_overload_accepts_numeric_string():
    from bgi_touch.engine.genshin_api import GenshinApi

    frame = object()
    positioner = SimpleNamespace(
        get_position_stable=Mock(return_value=(1.0, 2.0)),
    )
    api = GenshinApi.__new__(GenshinApi)
    api.ctx = SimpleNamespace(capture_bgr=Mock(return_value=frame))
    api._positioner_for = Mock(return_value=positioner)

    point = api.getPositionFromMap("Teyvat", "1200")

    assert (point.x, point.y) == (1.0, 2.0)
    positioner.get_position_stable.assert_called_once_with(
        frame, cache_time_ms=1200,
    )


def test_clear_party_cache_discards_team_mapping_without_resetting_map_position():
    from bgi_touch.engine.genshin_api import GenshinApi

    shared = {"旧角色": 1}
    api = GenshinApi.__new__(GenshinApi)
    api.ctx = SimpleNamespace(party_slots=shared)
    api._party_slots = shared
    positioner = SimpleNamespace(reset=Mock())
    api._positioners = {"Teyvat": positioner}

    api.clearPartyCache()

    assert shared == {}
    positioner.reset.assert_not_called()


def test_genshin_map_helpers_forward_force_country():
    task = MagicMock()
    api = _api_with_task(task)

    api.moveMapTo(1, 2, forceCountry="Mondstadt")
    api.clickMapPoint(3, 4, "璃月")
    api.moveIndependentMapTo(5, 6, "层岩巨渊", "璃月")

    task.move_map_to.assert_called_once_with(1.0, 2.0, force_country="Mondstadt")
    task.click_map_point.assert_called_once_with(3.0, 4.0, force_country="璃月")
    task.move_independent_map_to.assert_called_once_with(
        5.0, 6.0, "层岩巨渊", force_country="璃月"
    )


def test_country_coordinate_helpers_match_upstream_centers():
    from bgi_touch.pathing.map_locator import (
        nearest_teyvat_country,
        resolve_country_name,
    )

    assert resolve_country_name("Mondstadt") == "蒙德"
    assert resolve_country_name("Nod-Krai") == "挪德卡莱"
    assert nearest_teyvat_country(-876, 2278) == "蒙德"
    assert nearest_teyvat_country(4515, 3631) == "枫丹"


def test_area_switch_normalizes_force_country_alias():
    import bgi_touch.pathing.tp as module

    hit = SimpleNamespace(text="蒙德", y=720, click=Mock())
    region = SimpleNamespace(find_multi=Mock(return_value=[hit]))
    ctx = SimpleNamespace(
        input=SimpleNamespace(click_ref=Mock()),
        sleep=Mock(),
        capture_region=Mock(return_value=region),
    )
    task = module.TpTask.__new__(module.TpTask)
    task.ctx = ctx
    task.log = Mock()
    task.map_name = "Teyvat"

    assert task._switch_area("Mondstadt", timeout_s=0.05)

    hit.click.assert_called_once_with()
    assert task._selected_area == "蒙德"
    assert any("蒙德" in call.args[0] for call in task.log.call_args_list)


def test_js_runtime_awaits_genshin_tp_without_host_object_recursion(tmp_path, monkeypatch):
    """The public genshin facade must not expose a recursive Python host proxy."""
    pytest.importorskip("pythonmonkey")

    from bgi_touch.engine.genshin_api import GenshinApi
    from bgi_touch.engine.js_runtime import JsScriptRuntime
    from bgi_touch.vision.coordinate import ScreenTransform

    calls = []

    def fail_tp(self, x, y, map_name=None, force=False):
        calls.append((x, y, map_name, force))
        raise RuntimeError("传送失败：未能完成锚点确认（迭代/超时耗尽）")

    monkeypatch.setattr(GenshinApi, "tp", fail_tp)
    (tmp_path / "main.js").write_text(
        """
(async function () {
  await genshin.tp(4328, 3960);
})();
""",
        encoding="utf-8",
    )
    input_simulator = SimpleNamespace(
        key_down=Mock(), key_up=Mock(), key_press=Mock(), click_ref=Mock(),
        move_camera_by=Mock(), attack=Mock(), attack_down=Mock(),
        attack_up=Mock(), button_down=Mock(), button_up=Mock(),
        release_all=Mock(), tap_button=Mock(),
    )
    ctx = SimpleNamespace(
        input=input_simulator,
        device=SimpleNamespace(paste_text=Mock(), tap=Mock()),
        transform=ScreenTransform(2778, 1284),
        sleep=lambda _ms: None,
    )

    with pytest.raises(RuntimeError, match="未能完成锚点确认"):
        JsScriptRuntime(ctx, tmp_path, log=lambda _message: None).run()

    assert calls == [(4328.0, 3960.0, None, False)]


def test_js_runtime_exposes_lazy_navigation_as_plain_facade(tmp_path):
    """NavigationInstance must be callable without exposing a Python host object."""
    pytest.importorskip("pythonmonkey")

    from bgi_touch.engine.js_runtime import JsScriptRuntime
    from bgi_touch.vision.coordinate import ScreenTransform

    (tmp_path / "main.js").write_text(
        """
return JSON.stringify({
  lower: typeof genshin.lazyNavigationInstance.getPositionStableByCache,
  upper: typeof genshin.LazyNavigationInstance.GetPositionStableByCache,
  sameObject: genshin.lazyNavigationInstance === genshin.LazyNavigationInstance
});
""",
        encoding="utf-8",
    )
    input_simulator = SimpleNamespace(
        key_down=Mock(), key_up=Mock(), key_press=Mock(), click_ref=Mock(),
        move_camera_by=Mock(), attack=Mock(), attack_down=Mock(),
        attack_up=Mock(), button_down=Mock(), button_up=Mock(),
        release_all=Mock(), tap_button=Mock(),
    )
    ctx = SimpleNamespace(
        input=input_simulator,
        device=SimpleNamespace(paste_text=Mock(), tap=Mock()),
        transform=ScreenTransform(2778, 1284),
        sleep=lambda _ms: None,
    )

    result = json.loads(JsScriptRuntime(ctx, tmp_path).run())

    assert result == {
        "lower": "function",
        "upper": "function",
        "sameObject": True,
    }
