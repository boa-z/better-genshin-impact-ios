from types import SimpleNamespace

import numpy as np


def _hit(text, y):
    return SimpleNamespace(text=text, y=y)


def test_party_hud_maps_rows_when_active_slot_changes():
    from bgi_touch.engine.party_hud import map_party_hits

    existing = {"钟离": 1, "夜兰": 2, "枫原万叶": 3, "纳西妲": 4}
    rows = (215, 350, 465)

    active_first = map_party_hits(
        [_hit("夜 兰", 215), _hit("万叶", 350), _hit("纳西妲", 465)],
        active_slot=1,
        row_centers=rows,
        existing=existing,
    )
    active_second = map_party_hits(
        [_hit("钟离", 215), _hit("枫原万叶", 350), _hit("纳西妲", 465)],
        active_slot=2,
        row_centers=rows,
        existing=existing,
    )

    assert active_first == existing
    assert active_second == existing


def test_party_hud_normalizes_aliases_and_ignores_unknown_text():
    from bgi_touch.engine.party_hud import canonical_avatar_name, map_party_hits

    assert canonical_avatar_name(" Kamisato Ayaka ") == "神里绫华"
    assert canonical_avatar_name("神里 凌华") == "神里绫华"
    assert canonical_avatar_name("优兰尼娅湖") is None

    result = map_party_hits(
        [_hit("Kamisato Ayaka", 215), _hit("不是角色", 350)],
        active_slot=1,
        row_centers=(215, 350, 465),
    )
    assert result == {"神里绫华": 2}


def test_party_hud_deduplicates_ocr_rows_and_preserves_active_slot():
    from bgi_touch.engine.party_hud import map_party_hits

    existing = {"钟离": 1, "夜兰": 2, "枫原万叶": 3, "纳西妲": 4}
    result = map_party_hits(
        [
            _hit("夜兰", 214),
            _hit("夜兰", 348),
            _hit("钟离", 465),
        ],
        active_slot=1,
        row_centers=(215, 350, 465),
        existing=existing,
    )

    # The closer first row wins; a repeated active label cannot displace slot 1.
    assert result == {
        "钟离": 1,
        "夜兰": 2,
        "枫原万叶": 3,
        "纳西妲": 4,
    }


def test_recognize_party_slots_uses_layout_row_centers():
    from bgi_touch.engine.party_hud import PARTY_NAME_ROI, recognize_party_slots
    from bgi_touch.vision.coordinate import ScreenTransform

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    region = SimpleNamespace(
        find_multi=lambda ro, limit: (
            assert_roi(ro, limit)
            or [_hit("夜兰", 220), _hit("枫原万叶", 350), _hit("纳西妲", 460)]
        )
    )
    ctx = SimpleNamespace(
        input=SimpleNamespace(_active_slot=1),
        layout=SimpleNamespace(buttons={
            "partyRow1": (0.96, 220 / 1080),
            "partyRow2": (0.96, 350 / 1080),
            "partyRow3": (0.96, 460 / 1080),
        }),
        transform=ScreenTransform(1920, 1080),
    )

    result = recognize_party_slots(
        ctx,
        region,
        existing={"钟离": 1},
        log=lambda _message: None,
    )

    assert result == {"钟离": 1, "夜兰": 2, "枫原万叶": 3, "纳西妲": 4}


def assert_roi(ro, limit):
    assert ro.roi == (1560, 120, 360, 480)
    assert limit == 8
    return False
