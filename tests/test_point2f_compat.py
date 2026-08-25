import json
from types import SimpleNamespace

import pytest


def test_point2f_geometry_and_value_conversions():
    from bgi_touch.engine.recognition import Point2f

    point = Point2f(3.5, -4.25)
    assert (point.X, point.Y) == (3.5, -4.25)
    point.X = 5
    point.Y = 6
    assert (point.x, point.y) == (5.0, 6.0)
    assert point.ToPoint() == {"x": 5, "y": 6, "X": 5, "Y": 6}
    assert point.ToVec2f() == {
        "item0": 5.0, "item1": 6.0, "Item0": 5.0, "Item1": 6.0,
    }
    assert tuple(point.Plus()) == (5.0, 6.0)
    assert tuple(point.Negate()) == (-5.0, -6.0)
    assert tuple(point.Add({"X": 1, "Y": 2})) == (6.0, 8.0)
    assert tuple(point.Subtract([1, 2])) == (4.0, 4.0)
    assert tuple(point.Multiply(2)) == (10.0, 12.0)
    assert point.DistanceTo(Point2f(2, 2)) == pytest.approx(5.0)
    assert point.DotProduct(Point2f(1, 2)) == pytest.approx(17.0)
    assert point.CrossProduct(Point2f(1, 2)) == pytest.approx(4.0)
    assert point.Deconstruct() == [5.0, 6.0]
    assert tuple(Point2f.FromPoint({"x": 7, "y": 8})) == (7.0, 8.0)
    assert tuple(Point2f.FromVec2f({"Item0": 9, "Item1": 10})) == (9.0, 10.0)
    assert Point2f.Distance(Point2f(), Point2f(3, 4)) == pytest.approx(5.0)
    assert Point2f.DotProduct(Point2f(3, 4), Point2f(2, 1)) == pytest.approx(10.0)
    assert Point2f.CrossProduct(Point2f(3, 4), Point2f(2, 1)) == pytest.approx(-5.0)


def test_point2f_js_host_exposes_instance_and_static_aliases(tmp_path):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime
    from bgi_touch.vision.coordinate import ScreenTransform

    (tmp_path / "main.js").write_text(
        """
const a = new Point2f(3, 4);
const b = new Point2f(1, 2);
const fromPoint = Point2f.FromPoint({X: 7, Y: 8});
const fromVec = Point2f.fromVec2f({Item0: 9, Item1: 10});
return JSON.stringify({
  coords: [a.X, a.Y],
  add: [a.Add(b).x, a.Add(b).y],
  distance: Point2f.Distance(a, b),
  dot: a.DotProduct(b),
  staticDot: Point2f.dotProduct(a, b),
  cross: a.CrossProduct(b),
  staticCross: Point2f.CrossProduct(a, b),
  point: [a.ToPoint().X, a.ToPoint().Y],
  vec: [a.ToVec2f().Item0, a.ToVec2f().Item1],
  deconstructed: a.Deconstruct(),
  fromPoint: [fromPoint.x, fromPoint.y],
  fromVec: [fromVec.x, fromVec.y]
});
""",
        encoding="utf-8",
    )
    ctx = SimpleNamespace(
        transform=ScreenTransform(1920, 1080),
        input=SimpleNamespace(),
        device=SimpleNamespace(),
        sleep=lambda _ms: None,
    )
    result = json.loads(JsScriptRuntime(ctx, tmp_path, log=lambda _msg: None).run())
    assert result == {
        "coords": [3, 4],
        "add": [4, 6],
        "distance": pytest.approx(2.8284271247461903),
        "dot": 11,
        "staticDot": 11,
        "cross": 2,
        "staticCross": 2,
        "point": [3, 4],
        "vec": [3, 4],
        "deconstructed": [3, 4],
        "fromPoint": [7, 8],
        "fromVec": [9, 10],
    }
