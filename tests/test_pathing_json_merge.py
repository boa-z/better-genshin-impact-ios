import json


def _route():
    return {
        "info": {"name": "route", "mapName": "Teyvat", "tags": ["base"]},
        "config": {"realtimeTriggers": {"AutoPick": True}},
        "positions": [{"id": 1, "x": 0, "y": 0}],
    }


def test_pathing_load_merges_json5_global_and_named_covers(tmp_path):
    from bgi_touch.pathing.model import PathingTask

    route_path = tmp_path / "route.json"
    route_path.write_text(json.dumps(_route()), encoding="utf-8")
    (tmp_path / "control.json5").write_text(
        """
        {
          // Shared route defaults may contain comments and trailing commas.
          "global_cover": {
            "info": {
              "description": "global",
              "tags": ["global"],
            },
            "config": {
              "realtimeTriggers": {"AutoSkip": true},
            },
            "_arr_add": ["positions"],
            "positions": [{"id": 2, "x": 1, "y": 1}],
          },
          "json_list": [
            {
              "name": "route",
              "cover": {
                "config": {"realtimeTriggers": {"AutoPick": false}},
                "_arr_add": ["positions"],
                "positions": [{"id": 3, "x": 2, "y": 2}],
              },
            },
          ],
        }
        """,
        encoding="utf-8",
    )

    task = PathingTask.load(route_path)

    assert task.info["description"] == "global"
    assert task.info["tags"] == ["global"]
    assert task.realtime_triggers == {"AutoPick": False, "AutoSkip": True}
    assert [point.id for point in task.positions] == [1, 2, 3]


def test_pathing_load_resolves_control_ref_and_object_cover(tmp_path):
    from bgi_touch.pathing.model import PathingTask

    route_path = tmp_path / "route.json"
    route_path.write_text(json.dumps(_route()), encoding="utf-8")
    (tmp_path / "base-control.json5").write_text(
        json.dumps({
            "global_cover": {
                "_obj_cover": ["info"],
                "info": {"name": "replaced", "map_name": "Enkanomiya"},
            },
        }),
        encoding="utf-8",
    )
    (tmp_path / "control.json5").write_text(
        '{"ref": "base-control.json5"}',
        encoding="utf-8",
    )

    task = PathingTask.load(route_path)

    assert task.name == "replaced"
    assert task.map_name == "Enkanomiya"
    assert len(task.positions) == 1


def test_pathing_load_ignores_broken_optional_control_file(tmp_path):
    from bgi_touch.pathing.model import PathingTask

    route_path = tmp_path / "route.json"
    route_path.write_text(json.dumps(_route()), encoding="utf-8")
    (tmp_path / "control.json5").write_text("{ broken", encoding="utf-8")

    task = PathingTask.load(route_path)

    assert task.name == "route"
    assert len(task.positions) == 1
