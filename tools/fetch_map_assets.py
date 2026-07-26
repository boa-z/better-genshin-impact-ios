#!/usr/bin/env python3
"""下载 BetterGI 官方地图资产（NuGet 包）并解出本项目所需文件到 assets/map/。

用法：.venv/bin/python tools/fetch_map_assets.py [--version 1.0.19]
资产较大（~180MB 下载，解出 ~120MB），已在 .gitignore 中排除。
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1.0.19")
    ap.add_argument("--nupkg", help="已下载的 .nupkg 路径（跳过下载）")
    ap.add_argument("--models", action="store_true", help="同时下载 YOLO 模型资产（BetterGI.Assets.Model）")
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
        pkg2 = Path(tempfile.gettempdir()) / "bettergi-model-1.0.29.nupkg"
        if not pkg2.exists():
            print("下载模型包（~160MB）…")
            urllib.request.urlretrieve(NUPKG_URL.format(version="1.0.29").replace("Assets.Map", "Assets.Model"), pkg2)
        with zipfile.ZipFile(pkg2) as z:
            for w in wanted:
                out = mdl_dest / Path(w).name
                if out.exists():
                    continue
                with z.open("contentFiles/any/any/Assets/Model/" + w) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                print(f"解出 {w}")
        print(f"模型完成：{mdl_dest}")


if __name__ == "__main__":
    main()
