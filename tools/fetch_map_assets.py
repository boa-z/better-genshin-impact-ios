#!/usr/bin/env python3
"""下载 BetterGI 官方地图、模型与任务识别资产。

用法：.venv/bin/python tools/fetch_map_assets.py [--models] [--tcg] [--auto-boss]
资产较大（~180MB 下载，解出 ~120MB），已在 .gitignore 中排除。

默认校验 BetterGI.Assets.Map 1.0.21 的提瓦特底图尺寸；已有旧版资产时会
自动重新解出。使用 --refresh 可在上游资产更新但尺寸未变时强制刷新。
"""

from __future__ import annotations

import argparse
import shutil
import struct
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEST = PROJECT_ROOT / "assets" / "map"
NUPKG_URL = "https://www.nuget.org/api/v2/package/BetterGI.Assets.Map/{version}"
PKG_PREFIX = "contentFiles/any/any/Assets/Map/"
DEFAULT_MAP_VERSION = "1.0.21"

# The dimensions are deliberately checked from the image header rather than
# by decoding the complete map.  Teyvat_0_256.png is the cheap, authoritative
# layout sentinel: 22 columns × 256 and 19 rows × 256 in BetterGI 1.0.21.
# Keep unknown package versions usable; when the upstream map expands again,
# adding one entry here makes stale local assets self-healing instead of being
# silently accepted.
EXPECTED_MAP_IMAGE_SIZES: dict[str, dict[str, tuple[int, int]]] = {
    "1.0.21": {
        "Teyvat/Teyvat_0_256.png": (5632, 4864),
    },
}

# 需要的文件（相对 Assets/Map/）：曲面层 SIFT 特征 + 各尺度整图
WANTED = [
    "Teyvat/Teyvat_0_2048_SIFT.kp.bin",
    "Teyvat/Teyvat_0_2048_SIFT.mat.png",
    "Teyvat/Teyvat_0_256_SIFT.kp.bin",
    "Teyvat/Teyvat_0_256_SIFT.mat.png",
    "Teyvat/Teyvat_0_256.png",
    "Teyvat/MapBack_gray.png",
    "TheChasm/TheChasm_0_1024.png",
    "Enkanomiya/Enkanomiya_0_1024.png",
    "SeaOfBygoneEras/SeaOfBygoneEras_0_1024.png",
    "SeaOfBygoneEras/SeaOfBygoneEras_-1_1024.webp",
    "SeaOfBygoneEras/SeaOfBygoneEras_-2_1024.webp",
    "AncientSacredMountain/AncientSacredMountain_0_1024.png",
    "AncientSacredMountain/AncientSacredMountain_-1_1024.webp",
    "TempleOfSpace/TempleOfSpace_0_1024.png",
    "MoonCanon/MoonCanon_0_1024_SIFT.kp.bin",
    "MoonCanon/MoonCanon_0_1024_SIFT.mat.png",
]

FEATURE_SOURCES = [
    relative for relative in WANTED
    if relative.lower().endswith((".png", ".webp"))
    and "Teyvat/" not in relative
    and "_SIFT." not in relative
]

KP_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("size", "<f4"), ("angle", "<f4"),
    ("response", "<f4"), ("octave", "<i4"), ("class_id", "<i4"),
])


def image_size(path: str | Path) -> tuple[int, int]:
    """Read an image's dimensions without decoding large map bitmaps."""

    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(32)
    if header.startswith(b"\x89PNG\r\n\x1a\n") and header[12:16] == b"IHDR":
        return tuple(int(value) for value in struct.unpack(">II", header[16:24]))
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        chunk = header[12:16]
        if chunk == b"VP8X" and len(header) >= 30:
            width = 1 + int.from_bytes(header[24:27], "little")
            height = 1 + int.from_bytes(header[27:30], "little")
            return width, height
        if chunk == b"VP8 " and len(header) >= 30 and header[23:26] == b"\x9d\x01\x2a":
            width = int.from_bytes(header[26:28], "little") & 0x3FFF
            height = int.from_bytes(header[28:30], "little") & 0x3FFF
            return width, height
        if chunk == b"VP8L" and len(header) >= 25 and header[20] == 0x2F:
            b1, b2, b3, b4 = header[21:25]
            width = 1 + b1 + ((b2 & 0x3F) << 8)
            height = 1 + (b2 >> 6) + (b3 << 2) + ((b4 & 0x0F) << 10)
            return width, height
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"无法读取地图图像: {path}")
    return int(image.shape[1]), int(image.shape[0])


def expected_map_image_sizes(version: str) -> dict[str, tuple[int, int]]:
    """Return known layout sentinels for a map package version."""

    return EXPECTED_MAP_IMAGE_SIZES.get(str(version).strip(), {})


def map_asset_is_usable(relative: str, path: Path, version: str) -> bool:
    """Check existence and, for known sentinels, the expected dimensions."""

    if not path.is_file() or path.stat().st_size <= 0:
        return False
    expected = expected_map_image_sizes(version).get(relative)
    if expected is None:
        return True
    try:
        return image_size(path) == expected
    except (OSError, ValueError, RuntimeError):
        return False


def maps_ready(version: str) -> bool:
    return all(
        map_asset_is_usable(relative, DEST / relative, version)
        for relative in WANTED
    )


def download_file(url: str, destination: Path) -> None:
    """Download through a sibling temporary file and atomically install it."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    urllib.request.urlretrieve(url, partial)
    partial.replace(destination)


def build_map_features(force: bool = False) -> None:
    """Generate BetterGI-compatible SIFT stores for independent maps."""
    sift = cv2.SIFT_create()
    for relative in FEATURE_SOURCES:
        source = DEST / relative
        stem = source.with_suffix("")
        keypoint_path = stem.with_name(stem.name + "_SIFT.kp.bin")
        descriptor_path = stem.with_name(stem.name + "_SIFT.mat.png")
        if not force and keypoint_path.is_file() and descriptor_path.is_file():
            continue
        image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"无法读取地图图像: {source}")
        print(f"生成 SIFT：{relative} …")
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is None or not keypoints:
            raise RuntimeError(f"地图未提取到 SIFT 特征: {source}")
        raw = np.empty(len(keypoints), dtype=KP_DTYPE)
        for index, keypoint in enumerate(keypoints):
            raw[index] = (
                keypoint.pt[0], keypoint.pt[1], keypoint.size, keypoint.angle,
                keypoint.response, keypoint.octave, keypoint.class_id,
            )
        keypoint_temp = keypoint_path.with_suffix(".bin.part")
        descriptor_temp = descriptor_path.with_name(
            descriptor_path.name.removesuffix(".png") + ".part.png"
        )
        raw.tofile(keypoint_temp)
        descriptor_image = np.clip(descriptors, 0, 255).astype(np.uint8)
        if not cv2.imwrite(str(descriptor_temp), descriptor_image):
            keypoint_temp.unlink(missing_ok=True)
            raise OSError(f"无法写入地图描述子: {descriptor_temp}")
        keypoint_temp.replace(keypoint_path)
        descriptor_temp.replace(descriptor_path)
        print(f"  {len(keypoints)} 个关键点")

TCG_REPOSITORY = "https://raw.githubusercontent.com/babalae/better-genshin-impact"
TCG_ASSET_ROOT = "BetterGenshinImpact/GameTask/AutoGeniusInvokation/Assets"
TCG_DICE = [
    f"1920x1080/dice/{phase}_{element}.png"
    for phase in ("roll", "action")
    for element in (
        "anemo", "cryo", "dendro", "electro",
        "geo", "hydro", "omni", "pyro",
    )
]
TCG_OTHER = [
    f"1920x1080/other/{name}.png"
    for name in (
        "元素调和", "元素骰子不足", "冻结", "出战角色", "回合结束",
        "回合结算阶段", "对方行动中", "满能量", "确定", "空能量",
        "角色死亡", "角色状态_冻结", "角色状态_冻结2", "角色状态_水泡",
        "角色血量上方", "角色被打败", "退出挑战",
    )
]
TCG_WANTED = [*TCG_DICE, *TCG_OTHER, "tcg_character_card.json"]
TCG_EXTRA = {
    "combat_avatar.json": "BetterGenshinImpact/GameTask/AutoFight/Assets/combat_avatar.json",
}
QUICK_BUY_SOURCE = (
    "BetterGenshinImpact/GameTask/QuickBuy/Assets/1920x1080/SereniteaPotCoin.png"
)
AUTO_BOSS_ASSET_ROOT = "BetterGenshinImpact/GameTask/AutoBoss/Assets/1920x1080"
AUTO_BOSS_WANTED = [
    "original_resin_top_icon.png",
    "box.png",
    "open_resin_supplement_pane_button.png",
    "transient_resin_in_supplement_pane.png",
    "fragile_resin_in_supplement_pane.png",
    "increase_resin_usage_quantity_button.png",
]


def fetch_tcg_assets(ref: str) -> None:
    destination = PROJECT_ROOT / "assets" / "tcg"
    sources = {
        relative: f"{TCG_ASSET_ROOT}/{relative}" for relative in TCG_WANTED
    } | TCG_EXTRA
    missing = [
        relative for relative in sources if not (destination / relative).is_file()
    ]
    if not missing:
        print(f"七圣召唤资产已就绪：{destination}")
        return
    for relative in missing:
        out = destination / relative
        out.parent.mkdir(parents=True, exist_ok=True)
        source_path = quote(sources[relative], safe="/")
        url = f"{TCG_REPOSITORY}/{quote(ref, safe='')}/{source_path}"
        download_file(url, out)
        print(f"下载 TCG/{relative}")
    print(f"七圣召唤资产完成：{destination}")


def fetch_quick_buy_asset(ref: str) -> None:
    destination = PROJECT_ROOT / "assets" / "quickbuy" / "SereniteaPotCoin.png"
    if destination.is_file():
        print(f"快速购买资产已就绪：{destination.parent}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{TCG_REPOSITORY}/{quote(ref, safe='')}/{quote(QUICK_BUY_SOURCE, safe='/')}"
    download_file(url, destination)
    print(f"快速购买资产完成：{destination.parent}")


def fetch_auto_boss_assets(ref: str) -> None:
    destination = PROJECT_ROOT / "assets" / "autoboss"
    missing = [name for name in AUTO_BOSS_WANTED if not (destination / name).is_file()]
    if not missing:
        print(f"自动首领资产已就绪：{destination}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    for name in missing:
        out = destination / name
        source = f"{AUTO_BOSS_ASSET_ROOT}/{name}"
        url = f"{TCG_REPOSITORY}/{quote(ref, safe='')}/{quote(source, safe='/')}"
        download_file(url, out)
        print(f"下载 AutoBoss/{name}")
    print(f"自动首领资产完成：{destination}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=DEFAULT_MAP_VERSION)
    ap.add_argument("--nupkg", help="已下载的 .nupkg 路径（跳过下载）")
    ap.add_argument("--models", action="store_true", help="同时下载 YOLO 模型资产（BetterGI.Assets.Model）")
    ap.add_argument("--tcg", action="store_true", help="同时下载七圣召唤模板与角色卡配置")
    ap.add_argument(
        "--bettergi-ref", "--tcg-ref",
        dest="bettergi_ref",
        default="c3c22507c1e9ae95b8673ab3046f5ad4806c3b72",
        help="BetterGI Git ref（用于七圣召唤和快捷任务资产）",
    )
    ap.add_argument("--quick-buy", action="store_true", help="同时下载快速购买识别模板")
    ap.add_argument("--auto-boss", action="store_true", help="同时下载自动首领识别模板")
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="强制从地图包刷新全部地图资产（上游内容变更但尺寸未变时使用）",
    )
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    ready = maps_ready(args.version)
    refresh_maps = bool(args.refresh or not ready)
    if ready and not args.refresh:
        print(f"地图资产已就绪：{DEST}")

    if refresh_maps:
        if args.nupkg:
            pkg = Path(args.nupkg)
        else:
            url = NUPKG_URL.format(version=args.version)
            pkg = Path(tempfile.gettempdir()) / f"bettergi-map-{args.version}.nupkg"
            if not pkg.exists():
                print(f"下载 {url} …")
                download_file(url, pkg)
            print(f"包就绪：{pkg}（{pkg.stat().st_size / 1e6:.0f} MB）")

        with zipfile.ZipFile(pkg) as z:
            # Stage the complete requested set before replacing any current
            # file.  A cancelled/failed download therefore leaves the old
            # usable map intact instead of producing a half-new asset set.
            with tempfile.TemporaryDirectory(
                prefix="bgi-map-assets-", dir=str(DEST.parent)
            ) as staging_root:
                staging = Path(staging_root)
                staged: list[tuple[Path, Path, str]] = []
                for w in WANTED:
                    member = PKG_PREFIX + w
                    try:
                        z.getinfo(member)
                    except KeyError as error:
                        raise FileNotFoundError(
                            f"地图包 {args.version} 缺少资产: {member}"
                        ) from error
                    staged_path = staging / w
                    staged_path.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(member) as src, staged_path.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    if not map_asset_is_usable(w, staged_path, args.version):
                        expected = expected_map_image_sizes(args.version).get(w)
                        detail = f"，期望尺寸 {expected}" if expected else ""
                        raise ValueError(f"地图资产无效: {w}{detail}")
                    staged.append((staged_path, DEST / w, w))
                for staged_path, out, relative in staged:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    staged_path.replace(out)
                    print(f"解出 {relative}")
        print(f"完成：{DEST}")

    build_map_features(force=refresh_maps)

    if args.models:
        mdl_dest = PROJECT_ROOT / "assets" / "models"
        mdl_dest.mkdir(parents=True, exist_ok=True)
        wanted = ["Domain/bgi_tree.onnx", "Fish/bgi_fish.onnx", "Mine/bgi_mine.onnx"]
        missing_yolo = [w for w in wanted if not (mdl_dest / Path(w).name).exists()]
        if missing_yolo:
            pkg2 = Path(tempfile.gettempdir()) / "bettergi-model-1.0.29.nupkg"
            if not pkg2.exists():
                print("下载模型包（~160MB）…")
                download_file(
                    NUPKG_URL.format(version="1.0.29").replace("Assets.Map", "Assets.Model"),
                    pkg2,
                )
            with zipfile.ZipFile(pkg2) as z:
                for w in missing_yolo:
                    out = mdl_dest / Path(w).name
                    with z.open("contentFiles/any/any/Assets/Model/" + w) as src, open(out, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    print(f"解出 {w}")
        item_base = (
            "https://raw.githubusercontent.com/babalae/bettergi-libraries/main/"
            "BetterGI.Assets.Model/Assets/Model/ItemV2/"
        )
        for name in ("item.onnx", "item.csv"):
            out = mdl_dest / name
            if out.exists():
                continue
            print(f"下载 ItemV2/{name} …")
            download_file(item_base + name, out)
        print(f"模型完成：{mdl_dest}")

    if args.tcg:
        fetch_tcg_assets(args.bettergi_ref)
    if args.quick_buy:
        fetch_quick_buy_asset(args.bettergi_ref)
    if args.auto_boss:
        fetch_auto_boss_assets(args.bettergi_ref)


if __name__ == "__main__":
    main()
