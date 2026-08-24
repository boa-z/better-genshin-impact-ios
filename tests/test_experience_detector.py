from types import SimpleNamespace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _context(width=1920, height=1080):
    from bgi_touch.vision.coordinate import ScreenTransform

    return SimpleNamespace(transform=ScreenTransform(width, height))


def _frame_with_template(x=500, y=300):
    from bgi_touch.engine.recognition import Mat

    template = Mat.from_file(str(
        ROOT / "assets" / "templates" / "autofight" / "experience_57.png"
    ))
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    height, width = template.bgr.shape[:2]
    frame[y:y + height, x:x + width] = template.bgr
    return frame, x, y, template


def test_experience_detector_requires_template_and_color_confirmation():
    from bgi_touch.combat.experience import ExperienceDetector, ExperienceDetectorConfig
    from bgi_touch.engine.recognition import RecognitionObject

    frame, x, y, template = _frame_with_template()
    recognition = RecognitionObject.template_match(template)
    recognition.name = "experience_57"
    detector = ExperienceDetector(
        _context(),
        config=ExperienceDetectorConfig(interval_s=0, template_threshold=0.99),
        templates=[recognition],
    )
    detector.start()

    # BetterGI's secondary pixel is 147 reference pixels left of the hit.
    frame[y - 1:y + 2, x - 147 - 1:x - 147 + 2] = (220, 220, 180)
    assert detector.observe(frame)
    assert detector.has_detected_experience
    assert detector.stop() is True


def test_experience_detector_rejects_template_without_exp_pixel():
    from bgi_touch.combat.experience import ExperienceDetector, ExperienceDetectorConfig
    from bgi_touch.engine.recognition import RecognitionObject

    frame, _x, _y, template = _frame_with_template()
    recognition = RecognitionObject.template_match(template)
    detector = ExperienceDetector(
        _context(),
        config=ExperienceDetectorConfig(interval_s=0, template_threshold=0.99),
        templates=[recognition],
    )
    detector.start()

    assert not detector.observe(frame)
    assert not detector.has_detected_experience


def test_experience_pixel_offset_scales_with_device_height():
    from bgi_touch.combat.experience import validate_experience_pixel

    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
    # A 1920x1080 reference offset is scaled by 1440 / 1080.
    check_x = 1000 - round(147 * (1440 / 1080))
    frame[600:603, check_x - 1:check_x + 2] = (230, 230, 180)
    assert validate_experience_pixel(_context(2560, 1440), frame, 1000, 600)


def test_experience_detector_from_mapping_clamps_invalid_values():
    from bgi_touch.combat.experience import ExperienceDetectorConfig

    config = ExperienceDetectorConfig.from_mapping({
        "enabled": False,
        "intervalSeconds": -1,
        "templateThreshold": 2,
        "sampleRadius": -4,
        "colorMinBgr": [300, -2, "bad"],
    })
    assert config.enabled is False
    assert config.interval_s == 0
    assert config.template_threshold == 1
    assert config.sample_radius == 0
    assert config.color_min_bgr == (200, 200, 150)
