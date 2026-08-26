from pathlib import Path


def test_converter_reports_unknown_fixed_host_members_with_source_line(tmp_path: Path):
    from bgi_touch.converter.convert import scan_js_compat

    (tmp_path / "main.js").write_text(
        "await genshin.returnMainUi();\n"
        "await genshin.notMigratedYet();\n"
        "dispatcher.notARealTask();\n"
        "settings.notAHostApi = true;\n",
        encoding="utf-8",
    )

    findings = scan_js_compat(tmp_path)

    assert any(
        "main.js:2" in item and "genshin.notMigratedYet" in item
        for item in findings["unsupported"]
    )
    assert any("dispatcher.notARealTask" in item for item in findings["unsupported"])
    assert not any("returnMainUi" in item for item in findings["unsupported"])
    assert not any("settings.notAHostApi" in item for item in findings["unsupported"])


def test_converter_distinguishes_known_unsupported_mat_apis(tmp_path: Path):
    from bgi_touch.converter.convert import scan_js_compat

    (tmp_path / "main.js").write_text(
        "const mat = Mat.fromNativePointer(ptr);\n",
        encoding="utf-8",
    )

    findings = scan_js_compat(tmp_path)

    assert any(
        "Mat.fromNativePointer" in item and "原生 OpenCV 指针" in item
        for item in findings["unsupported"]
    )


def test_converter_catalogue_covers_every_registered_host_member():
    from bgi_touch.converter.convert import SUPPORTED, _HOST_MEMBER_SURFACE

    supported = set(SUPPORTED)
    missing = {
        f"{root}.{member}"
        for root, members in _HOST_MEMBER_SURFACE.items()
        for member in members
        if f"{root}.{member}" not in supported
    }

    assert not missing
