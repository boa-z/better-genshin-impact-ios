import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest


TOOL = Path(__file__).parents[1] / "tools" / "fetch_map_assets.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("fetch_map_assets_test", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def png_header(width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\x0dIHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x00\x00\x00\x00"
    )


def test_maps_ready_rejects_pre_expansion_teyvat_asset(tmp_path, monkeypatch):
    module = load_tool()
    monkeypatch.setattr(module, "DEST", tmp_path)

    sentinel = "Teyvat/Teyvat_0_256.png"
    for relative in module.WANTED:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            png_header(5632, 3840) if relative == sentinel else b"asset"
        )

    assert not module.maps_ready("1.0.21")
    (tmp_path / sentinel).write_bytes(png_header(5632, 4864))
    assert module.maps_ready("1.0.21")


def test_refresh_replaces_complete_map_set_and_rebuilds_features(tmp_path, monkeypatch):
    module = load_tool()
    destination = tmp_path / "map"
    package = tmp_path / "map.nupkg"
    monkeypatch.setattr(module, "DEST", destination)

    with zipfile.ZipFile(package, "w") as archive:
        for relative in module.WANTED:
            payload = (
                png_header(5632, 4864)
                if relative == "Teyvat/Teyvat_0_256.png"
                else b"new-asset"
            )
            archive.writestr(module.PKG_PREFIX + relative, payload)

    for relative in module.WANTED:
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"old-asset")

    rebuilt = []
    monkeypatch.setattr(module, "build_map_features", lambda force=False: rebuilt.append(force))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fetch_map_assets.py",
            "--version",
            "1.0.21",
            "--nupkg",
            str(package),
            "--refresh",
        ],
    )

    module.main()

    assert rebuilt == [True]
    assert (destination / "Teyvat/Teyvat_0_256.png").read_bytes() == png_header(5632, 4864)
    assert (destination / "Teyvat/Teyvat_0_2048_SIFT.kp.bin").read_bytes() == b"new-asset"


def test_failed_map_package_leaves_existing_assets_untouched(tmp_path, monkeypatch):
    module = load_tool()
    destination = tmp_path / "map"
    package = tmp_path / "incomplete.nupkg"
    monkeypatch.setattr(module, "DEST", destination)

    for relative in module.WANTED:
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"old-asset")
    sentinel = destination / "Teyvat/Teyvat_0_256.png"
    sentinel.write_bytes(png_header(5632, 3840))

    with zipfile.ZipFile(package, "w") as archive:
        for relative in module.WANTED[:-1]:
            payload = (
                png_header(5632, 4864)
                if relative == "Teyvat/Teyvat_0_256.png"
                else b"new-asset"
            )
            archive.writestr(module.PKG_PREFIX + relative, payload)

    monkeypatch.setattr(module, "build_map_features", lambda force=False: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch_map_assets.py", "--nupkg", str(package)],
    )

    with pytest.raises(FileNotFoundError, match="缺少资产"):
        module.main()

    assert sentinel.read_bytes() == png_header(5632, 3840)
