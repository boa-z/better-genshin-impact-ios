import numpy as np


def test_minimap_preprocessor_keeps_bettergi_center_crop_and_masks_icons():
    from bgi_touch.pathing.minimap import (
        prepare_minimap_source,
        preprocess_minimap_for_matching,
    )

    image = np.full((212, 212, 3), (80, 120, 170), dtype=np.uint8)
    image[102:110, 102:110] = (100, 100, 100)

    prepared = prepare_minimap_source(image)
    assert prepared is not None
    source, icon_mask = prepared
    assert source.shape == (156, 156, 3)
    assert icon_mask.shape == (156, 156)
    assert tuple(source[0, 0]) == (80, 120, 170)
    assert icon_mask[78, 78] == 255

    query = preprocess_minimap_for_matching(image, angle=45)
    assert query is not None
    gray, valid_mask = query
    assert gray.shape == (156, 156)
    assert valid_mask.shape == (156, 156)
    assert gray.dtype == np.uint8
    assert valid_mask.dtype == np.uint8
    assert valid_mask[0, 0] == 0
    assert valid_mask[78, 78] == 0
    assert valid_mask[70, 70] == 255


def test_map_locator_uses_preprocessed_mask_for_sift():
    from bgi_touch.pathing.map_locator import MapLocator

    seen = {}

    class Sift:
        def detectAndCompute(self, image, mask):
            seen["image"] = image
            seen["mask"] = mask
            return [], None

    locator = object.__new__(MapLocator)
    locator._sift = Sift()
    image = np.zeros((156, 156), dtype=np.uint8)
    valid_mask = np.zeros_like(image)
    valid_mask[40:100, 40:100] = 255

    desc, points = locator._extract(image, valid_mask)

    assert desc is None
    assert points.shape == (0, 2)
    assert np.array_equal(seen["image"], image)
    assert np.array_equal(seen["mask"], valid_mask)


def test_positioner_falls_back_to_legacy_sift_query(monkeypatch):
    from bgi_touch.pathing.positioner import MinimapPositioner

    image = np.zeros((156, 156), dtype=np.uint8)
    mask = np.full_like(image, 255)
    calls = []

    class Locator:
        def locate_world(self, *query):
            calls.append(query)
            return None if len(calls) == 1 else (12.0, 34.0)

    positioner = object.__new__(MinimapPositioner)
    positioner.locator = Locator()
    monkeypatch.setattr(
        positioner,
        "_preprocessed_query",
        lambda _frame: (image, mask),
    )
    monkeypatch.setattr(positioner, "crop_minimap", lambda _frame: image)

    assert positioner.get_position(object()) == (12.0, 34.0)
    assert len(calls) == 2
    assert len(calls[0]) == 2
    assert len(calls[1]) == 1
