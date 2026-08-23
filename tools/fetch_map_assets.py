#!/usr/bin/env python3
"""下载 BetterGI 官方地图、模型与七圣召唤识别资产。

用法：.venv/bin/python tools/fetch_map_assets.py [--models] [--tcg]
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
]

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1.0.19")
    ap.add_argument("--nupkg", help="已下载的 .nupkg 路径（跳过下载）")
    ap.add_argument("--models", action="store_true", help="同时下载 YOLO 模型资产（BetterGI.Assets.Model）")
    ap.add_argument("--tcg", action="store_true", help="同时下载七圣召唤模板与角色卡配置")
    ap.add_argument(
        "--tcg-ref",
        default="c3c22507c1e9ae95b8673ab3046f5ad4806c3b72",
        help="BetterGI Git ref（默认是本移植版本审计对应提交）",
    )
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
        fetch_tcg_assets(args.tcg_ref)


if __name__ == "__main__":
    main()
