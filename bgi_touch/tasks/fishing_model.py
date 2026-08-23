"""Fish YOLO labels, bait policy, and BetterGI's HutaoFisher rod model."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from ..vision.yolo import Detection, YoloPredictor


FISH_LABELS = (
    "angler", "axe_marlin", "axehead", "butterflyfish", "crystal_eye",
    "err_rod", "heartfeather_bass", "koi", "koi_head", "large_medaka",
    "magma_rapidfish", "maintenance_mek", "mauler_shark", "medaka",
    "phony_unihornfish", "pufferfish", "rapidfish", "ray", "rod",
    "secret_source", "stickleback", "sunfish", "unihornfish",
)


@dataclass(frozen=True)
class FishKind:
    label: str
    bait: str
    chinese_name: str
    net_index: int


_KINDS = (
    FishKind("medaka", "果酿饵", "花鳉", 0),
    FishKind("large_medaka", "果酿饵", "大花鳉", 1),
    FishKind("stickleback", "赤糜饵", "棘鱼", 2),
    FishKind("koi", "飞蝇假饵", "假龙", 3),
    FishKind("koi_head", "飞蝇假饵", "假龙头", 3),
    FishKind("butterflyfish", "蠕虫假饵", "蝶鱼", 4),
    FishKind("pufferfish", "飞蝇假饵", "炮鲀", 5),
    FishKind("ray", "飞蝇假饵", "鳐", 6),
    FishKind("angler", "甘露饵", "角鲀", 7),
    FishKind("axe_marlin", "甘露饵", "斧枪鱼", 8),
    FishKind("heartfeather_bass", "酸桔饵", "心羽鲈", 9),
    FishKind("maintenance_mek", "维护机关频闪诱饵", "维护机关", 10),
    FishKind("unihornfish", "澄晶果粒饵", "独角鱼", 10),
    FishKind("sunfish", "澄晶果粒饵", "翻车鲀", 7),
    FishKind("rapidfish", "澄晶果粒饵", "斗士急流鱼", 9),
    FishKind("phony_unihornfish", "温火饵", "燃素独角鱼", 10),
    FishKind("magma_rapidfish", "温火饵", "炽岩斗士急流鱼", 9),
    FishKind("secret_source", "温火饵", "秘源机关・巡戒使", 9),
    FishKind("mauler_shark", "清白饵", "凶凶鲨", 9),
    FishKind("crystal_eye", "清白饵", "明眼鱼", 9),
    FishKind("axehead", "槲梭饵", "巨斧鱼", 9),
)
FISH_KINDS = {kind.label: kind for kind in _KINDS}


@dataclass(frozen=True)
class FishTarget:
    kind: FishKind
    rect: tuple[float, float, float, float]
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        x, y, width, height = self.rect
        return x + width / 2, y + height / 2


@dataclass(frozen=True)
class FishPond:
    fishes: tuple[FishTarget, ...]
    rod: tuple[float, float, float, float] | None = None

    @property
    def bounds(self) -> tuple[float, float, float, float] | None:
        if not self.fishes:
            return None
        left = min(fish.rect[0] for fish in self.fishes)
        top = min(fish.rect[1] for fish in self.fishes)
        right = max(fish.rect[0] + fish.rect[2] for fish in self.fishes)
        bottom = max(fish.rect[1] + fish.rect[3] for fish in self.fishes)
        return left, top, right - left, bottom - top


def fishpond_from_detections(
    detections: list[Detection], *, include_target: bool = False
) -> FishPond:
    fishes = []
    rod = None
    for detection in detections:
        if detection.confidence < 0.4 or not 0 <= detection.class_id < len(FISH_LABELS):
            continue
        label = FISH_LABELS[detection.class_id]
        rect = (detection.x, detection.y, detection.width, detection.height)
        if label in {"rod", "err_rod"}:
            rod = rect
            continue
        kind = FISH_KINDS.get(label)
        if kind is not None and not (include_target and label == "koi"):
            fishes.append(FishTarget(kind, rect, detection.confidence))
    fishes.sort(key=lambda fish: -fish.confidence)
    return FishPond(tuple(fishes), rod)


def choose_bait(fishes: tuple[FishTarget, ...], ignored: set[str] | None = None) -> str | None:
    ignored = ignored or set()
    counts = Counter(fish.kind.bait for fish in fishes if fish.kind.bait not in ignored)
    return counts.most_common(1)[0][0] if counts else None


class FishPredictor:
    def __init__(self):
        self.model = YoloPredictor("bgi_fish")

    def predict(self, bgr, *, include_target: bool = False) -> FishPond:
        return fishpond_from_detections(
            self.model.predict(bgr, conf_threshold=0.4),
            include_target=include_target,
        )


_DZ = (1.0307939, 1.5887239, 1.4377865, 0.8548809, 1.8640924, -0.1687729,
       1.8621461, 0.7167622, 1.7071064, 1.8727832, 0.5531539)
_H = (0.5840698, 0.8029298, 0.6090596, -0.1390072, 0.7214464, -0.6076725,
      0.3286690, -0.2991239, 0.6072225, 0.7662407, -0.3689651)
_WEIGHT = (
    (0.7779633, -1.7124480, 2.7366412), (-0.0381155, -1.6536976, 3.5904298),
    (0.1947731, -0.0445049, 0.8416666), (-0.0331017, -1.3641578, 1.2834741),
    (1.0268835, -1.6553984, 2.9930501), (0.0108103, -0.8515291, 1.0032536),
    (-0.0746362, -0.9677668, 0.7450780), (0.7382144, -9.5275803, 2.6134675),
    (-0.3597502, -1.7422760, 1.4354013), (-0.0578425, -2.0274212, 1.7173727),
    (-0.1225260, -1.0630554, 1.2958838),
)
_BIAS = (
    (3.1733532, 9.3601589, -11.0612173), (6.4961057, 11.2683334, -13.7752209),
    (2.3662698, 2.4709859, -2.5402584), (2.4701204, 8.5112562, -7.6070199),
    (0.9597272, 8.9189463, -11.9037018), (2.1239815, 5.8446727, -5.7748013),
    (2.1403685, 5.5432696, -4.0048418), (-9.0128260, 28.4402637, -24.2205143),
    (5.2072763, 8.6428488, -9.2946615), (4.9253063, 11.4634714, -9.4336052),
    (5.2460732, 7.7711511, -7.5998945),
)
_OFFSET = (0.8, 0.4, 0.35, 0.35, 0.6, 0.3, 0.3, 0.8, 0.8, 0.8, 0.8)


def rod_state(
    rod: tuple[float, float, float, float],
    fish: FishTarget,
    screen_width: float,
    screen_height: float,
) -> int:
    """Return 0=cast, 1=too near, 2=too far, or -1 for invalid geometry."""
    rx, ry, rw, rh = rod
    fx, fy, fw, fh = fish.rect
    rx1, rx2 = rx / screen_width * 1024, (rx + rw) / screen_width * 1024
    ry1, ry2 = ry / screen_height * 576, (ry + rh) / screen_height * 576
    fx1, fx2 = fx / screen_width * 1024, (fx + fw) / screen_width * 1024
    fy1, fy2 = fy / screen_height * 576, (fy + fh) / screen_height * 576
    alpha = 1734.34 / 2.5
    index = fish.kind.net_index
    try:
        a = (rx2 - rx1) / 2 / alpha
        b = (ry2 - ry1) / 2 / alpha
        h = (fy2 - fy1) / 2 / alpha
        if a < b:
            b = math.sqrt(a * b)
            a = b + 1e-6
        v0 = (288 - (ry1 + ry2) / 2) / alpha
        u = (fx1 + fx2 - rx1 - rx2) / 2 / alpha
        v = (288 - (fy1 + fy2) / 2) / alpha
        y0 = math.sqrt(a ** 4 - b * b + a * a * (1 - b * b + v0 * v0)) / (a * a)
        z0 = b / (a * a)
        t = a * a * (y0 * b + v0) / (a * a - b * b)
        v -= h * _H[index]
        x = u * (z0 + _DZ[index]) * math.sqrt(1 + t * t) / (t - v)
        y = (z0 + _DZ[index]) * (1 + t * v) / (t - v)
        distance = math.hypot(x, y - y0)
        logits = [
            _WEIGHT[index][state] * distance + _BIAS[index][state]
            for state in range(3)
        ]
        maximum = max(logits)
        exp = [math.exp(value - maximum) for value in logits]
        probability = [value / sum(exp) for value in exp]
        probability[0] -= _OFFSET[index]
        return max(range(3), key=probability.__getitem__)
    except (ValueError, ZeroDivisionError, OverflowError):
        return -1
