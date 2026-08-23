#!/usr/bin/env python3
"""下载 BetterGI 官方地图、模型与任务识别资产。

用法：.venv/bin/python tools/fetch_map_assets.py [--models] [--tcg] [--auto-boss]
资产较大（~180MB 下载，解出 ~120MB），已在 .gitignore 中排除。
"""

from __future__ import annotations

import argparse
import shutil
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


def build_map_features() -> None:
    """Generate BetterGI-compatible SIFT stores for independent maps."""
    sift = cv2.SIFT_create()
    for relative in FEATURE_SOURCES:
        source = DEST / relative
        stem = source.with_suffix("")
        keypoint_path = stem.with_name(stem.name + "_SIFT.kp.bin")
        descriptor_path = stem.with_name(stem.name + "_SIFT.mat.png")
        if keypoint_path.is_file() and descriptor_path.is_file():
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
        partial = out.with_suffix(out.suffix + ".part")
        urllib.request.urlretrieve(url, partial)
        partial.replace(out)
        print(f"下载 TCG/{relative}")
    print(f"七圣召唤资产完成：{destination}")


def fetch_quick_buy_asset(ref: str) -> None:
    destination = PROJECT_ROOT / "assets" / "quickbuy" / "SereniteaPotCoin.png"
    if destination.is_file():
        print(f"快速购买资产已就绪：{destination.parent}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{TCG_REPOSITORY}/{quote(ref, safe='')}/{quote(QUICK_BUY_SOURCE, safe='/')}"
    partial = destination.with_suffix(".png.part")
    urllib.request.urlretrieve(url, partial)
    partial.replace(destination)
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
        partial = out.with_suffix(out.suffix + ".part")
        urllib.request.urlretrieve(url, partial)
        partial.replace(out)
        print(f"下载 AutoBoss/{name}")
    print(f"自动首领资产完成：{destination}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1.0.21")
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
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    maps_ready = all((DEST / w).exists() for w in WANTED)
    if maps_ready:
        print(f"地图资产已就绪：{DEST}")

    if not maps_ready:
        if args.nupkg:
            pkg = Path(args.nupkg)
        else:
            url = NUPKG_URL.format(version=args.version)
            pkg = Path(tempfile.gettempdir()) / f"bettergi-map-{args.version}.nupkg"
            if not pkg.exists():
                print(f"下载 {url} …")
                urllib.request.urlretrieve(url, pkg)
            print(f"包就绪：{pkg}（{pkg.stat().st_size / 1e6:.0f} MB）")

        with zipfile.ZipFile(pkg) as z:
            for w in WANTED:
                out = DEST / w
                if out.exists():
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(PKG_PREFIX + w) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                print(f"解出 {w}")
        print(f"完成：{DEST}")

    build_map_features()

    if args.models:
        mdl_dest = PROJECT_ROOT / "assets" / "models"
        mdl_dest.mkdir(parents=True, exist_ok=True)
        wanted = ["Domain/bgi_tree.onnx", "Fish/bgi_fish.onnx", "Mine/bgi_mine.onnx"]
        missing_yolo = [w for w in wanted if not (mdl_dest / Path(w).name).exists()]
        if missing_yolo:
            pkg2 = Path(tempfile.gettempdir()) / "bettergi-model-1.0.29.nupkg"
            if not pkg2.exists():
                print("下载模型包（~160MB）…")
                urllib.request.urlretrieve(NUPKG_URL.format(version="1.0.29").replace("Assets.Map", "Assets.Model"), pkg2)
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
            urllib.request.urlretrieve(item_base + name, out)
        print(f"模型完成：{mdl_dest}")

    if args.tcg:
        fetch_tcg_assets(args.bettergi_ref)
    if args.quick_buy:
        fetch_quick_buy_asset(args.bettergi_ref)
    if args.auto_boss:
        fetch_auto_boss_assets(args.bettergi_ref)


if __name__ == "__main__":
    main()
