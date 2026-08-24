"""Load BetterGI ``Recognition.json`` assets on the portable runtime.

The desktop client stores small recognition objects as JSON and resolves their
regions against the current capture size.  Keeping that format available on
iOS lets migrated tasks consume upstream assets without duplicating every
template definition in Python.
"""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2

from .recognition import Mat, RecognitionObject, SearchOptions


class RecognitionJsonError(ValueError):
    """Raised when a Recognition.json file cannot be mapped safely."""


def _number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RecognitionJsonError(f"{field} 必须是数字") from error
    if not math.isfinite(result):
        raise RecognitionJsonError(f"{field} 必须是有限数字")
    return result


def _rect(value: Any, field: str) -> tuple[float, float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        values = value
    elif isinstance(value, Mapping):
        lowered = {str(key).casefold(): item for key, item in value.items()}
        values = (
            lowered.get("x", 0), lowered.get("y", 0),
            lowered.get("width", 0), lowered.get("height", 0),
        )
    else:
        values = (
            getattr(value, "x", getattr(value, "X", 0)),
            getattr(value, "y", getattr(value, "Y", 0)),
            getattr(value, "width", getattr(value, "Width", 0)),
            getattr(value, "height", getattr(value, "Height", 0)),
        )
    return tuple(_number(item, field) for item in values)


def _parse_color(value: Any) -> tuple[int, int, int]:
    if isinstance(value, str):
        text = value.strip().lstrip("#")
        if len(text) == 6:
            try:
                red, green, blue = (
                    int(text[index:index + 2], 16) for index in (0, 2, 4)
                )
            except ValueError as error:
                raise RecognitionJsonError("maskColor 不是有效的 HTML 颜色") from error
            return blue, green, red
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return int(value[2]), int(value[1]), int(value[0])
    if isinstance(value, Mapping):
        lowered = {str(key).casefold(): item for key, item in value.items()}
        return (
            int(lowered.get("b", lowered.get("blue", 0))),
            int(lowered.get("g", lowered.get("green", 255))),
            int(lowered.get("r", lowered.get("red", 0))),
        )
    raise RecognitionJsonError("maskColor 必须是 #RRGGBB、数组或颜色对象")


def _template_mode(value: Any) -> int:
    if value is None or str(value).strip() in {"", "Color"}:
        return cv2.IMREAD_COLOR
    normalized = str(value).casefold().replace("_", "")
    modes = {
        "color": cv2.IMREAD_COLOR,
        "grayscale": cv2.IMREAD_GRAYSCALE,
        "gray": cv2.IMREAD_GRAYSCALE,
        "unchanged": cv2.IMREAD_UNCHANGED,
    }
    if normalized not in modes:
        raise RecognitionJsonError(f"不支持的 templateMode: {value}")
    return modes[normalized]


def _match_mode(value: Any) -> int:
    if value is None:
        return cv2.TM_CCOEFF_NORMED
    normalized = re.sub(r"[^a-z0-9]", "", str(value).casefold())
    modes = {
        "sqdiff": cv2.TM_SQDIFF,
        "sqdiffnormed": cv2.TM_SQDIFF_NORMED,
        "ccorr": cv2.TM_CCORR,
        "ccorrnormed": cv2.TM_CCORR_NORMED,
        "ccoeff": cv2.TM_CCOEFF,
        "ccoeffnormed": cv2.TM_CCOEFF_NORMED,
    }
    if normalized not in modes:
        raise RecognitionJsonError(f"不支持的 templateMatchMode: {value}")
    return modes[normalized]


class _ExpressionEvaluator:
    """Small side-effect-free subset of the NCalc expressions used by BGI."""

    _binary = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.Div: lambda left, right: left / right,
        ast.FloorDiv: lambda left, right: left // right,
        ast.Mod: lambda left, right: left % right,
        ast.Pow: lambda left, right: left ** right,
    }

    def __init__(self, config: Mapping[str, Any], width: int, height: int,
                 extra_parameters: Mapping[str, Any] | None = None):
        self.config = config
        self.width = int(width)
        self.height = int(height)
        self.variables = dict(config.get("vars") or {})
        self.regions = dict(config.get("regions") or {})
        self.values: dict[str, float] = {}
        self.resolving: set[str] = set()
        self.parameters = {
            "cw": self.width,
            "ch": self.height,
            "cx": 0,
            "cy": 0,
            "s": self.width / 1920 if self.width < 1920 else 1,
        }
        if extra_parameters:
            self.parameters.update(extra_parameters)

    def evaluate_rect(self, expression: Any, field: str) -> tuple[int, int, int, int]:
        if not isinstance(expression, str):
            return tuple(round(value) for value in _rect(expression, field))
        resolved = expression.strip()
        seen: set[str] = set()
        while resolved.startswith("@"):
            alias = resolved[1:]
            if alias in seen or alias not in self.regions:
                raise RecognitionJsonError(f"未找到区域别名或存在循环引用: {alias}")
            seen.add(alias)
            resolved = str(self.regions[alias]).strip()
        value = self.evaluate(resolved, field)
        if not isinstance(value, tuple) or len(value) != 4:
            raise RecognitionJsonError(f"{field} 未返回 Rect")
        return tuple(round(_number(item, field)) for item in value)

    def evaluate(self, expression: str, field: str) -> Any:
        try:
            tree = ast.parse(expression, mode="eval").body
        except SyntaxError as error:
            raise RecognitionJsonError(f"{field} 表达式无效: {expression}") from error
        try:
            return self._visit(tree, field)
        except RecognitionJsonError:
            raise
        except (ArithmeticError, TypeError, ValueError) as error:
            raise RecognitionJsonError(f"{field} 表达式计算失败: {expression}") from error

    def _visit(self, node: ast.AST, field: str) -> Any:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self.parameters:
                return self.parameters[node.id]
            if node.id in self.variables:
                return self._variable(node.id, field)
            raise RecognitionJsonError(f"{field} 使用了未定义变量: {node.id}")
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            value = self._visit(node.operand, field)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in self._binary:
            return self._binary[type(node.op)](
                self._visit(node.left, field), self._visit(node.right, field),
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            args = [self._visit(argument, field) for argument in node.args]
            return self._call(node.func.id, args, field)
        raise RecognitionJsonError(f"{field} 包含不支持的表达式节点")

    def _variable(self, name: str, field: str) -> float:
        if name in self.values:
            return self.values[name]
        if name in self.resolving:
            raise RecognitionJsonError(f"变量 {name} 存在循环引用")
        self.resolving.add(name)
        try:
            value = self.evaluate(str(self.variables[name]), f"vars.{name}")
            result = _number(value, f"vars.{name}")
            self.values[name] = result
            return result
        finally:
            self.resolving.remove(name)

    def _call(self, name: str, args: list[Any], field: str) -> Any:
        if name == "rect" and len(args) == 4:
            return tuple(round(_number(value, field)) for value in args)
        if name in {
            "cutLeft", "cutRight", "cutTop", "cutBottom",
        } and len(args) == 1:
            ratio = _number(args[0], field)
            if name == "cutLeft":
                return (0, 0, round(self.width * ratio), self.height)
            if name == "cutRight":
                size = round(self.width * ratio)
                return (self.width - size, 0, size, self.height)
            if name == "cutTop":
                return (0, 0, self.width, round(self.height * ratio))
            size = round(self.height * ratio)
            return (0, self.height - size, self.width, size)
        cuts = {
            "cutLeftTop": (True, True),
            "cutRightTop": (False, True),
            "cutLeftBottom": (True, False),
            "cutRightBottom": (False, False),
        }
        if name in cuts and len(args) == 2:
            horizontal_left, vertical_top = cuts[name]
            width = round(self.width * _number(args[0], field))
            height = round(self.height * _number(args[1], field))
            x = 0 if horizontal_left else self.width - width
            y = 0 if vertical_top else self.height - height
            return x, y, width, height
        raise RecognitionJsonError(f"{field} 调用了不支持的函数或参数数量: {name}")


def _resolve_alias(value: Any, aliases: Mapping[str, Any], field: str) -> str:
    if value is None:
        raise RecognitionJsonError(f"{field} 不能为空")
    resolved = str(value).strip()
    seen: set[str] = set()
    while resolved.startswith("@"):
        alias = resolved[1:]
        if alias in seen or alias not in aliases:
            raise RecognitionJsonError(f"{field} 未找到别名或存在循环引用: {alias}")
        seen.add(alias)
        resolved = str(aliases[alias]).strip()
    return resolved


def _load_template(path: Path, mode: int) -> Mat:
    if not path.is_file():
        raise FileNotFoundError(f"未找到识别模板: {path}")
    image = cv2.imread(str(path), mode)
    if image is None:
        raise RecognitionJsonError(f"无法读取识别模板: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return Mat(image)


def load_recognition_object(
    config: str | Path | Mapping[str, Any],
    object_name: str,
    *,
    capture_width: int = 1920,
    capture_height: int = 1080,
    template_root: str | Path | None = None,
    template_loader: Callable[[str, int], Mat | str | Path] | None = None,
    extra_parameters: Mapping[str, Any] | None = None,
) -> RecognitionObject:
    """Load one object from a BetterGI Recognition.json document."""
    config_path: Path | None = None
    if isinstance(config, Mapping):
        document = dict(config)
    else:
        config_path = Path(config).expanduser().resolve()
        document = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(document, Mapping):
        raise RecognitionJsonError("Recognition.json 顶层必须是对象")
    objects = document.get("objects")
    if not isinstance(objects, Mapping) or object_name not in objects:
        raise KeyError(f"未找到名称为 {object_name} 的 RecognitionObject")
    raw = objects[object_name]
    if not isinstance(raw, Mapping):
        raise RecognitionJsonError(f"objects.{object_name} 必须是对象")

    evaluator = _ExpressionEvaluator(document, int(capture_width), int(capture_height), extra_parameters)
    recognition_type = str(raw.get("type") or "").strip()
    if not recognition_type:
        raise RecognitionJsonError(f"objects.{object_name}.type 不能为空")
    ro = RecognitionObject()
    ro.RecognitionType = recognition_type
    ro.Name = str(raw.get("name") or object_name)
    if raw.get("roi") is not None:
        ro.RegionOfInterest = evaluator.evaluate_rect(raw["roi"], f"objects.{object_name}.roi")
    if raw.get("threshold") is not None:
        ro.Threshold = _number(raw["threshold"], f"objects.{object_name}.threshold")
    if raw.get("use3Channels") is not None:
        ro.Use3Channels = bool(raw["use3Channels"])
    if raw.get("templateMatchMode") is not None:
        ro.TemplateMatchMode = _match_mode(raw["templateMatchMode"])
    if raw.get("useMask") is not None:
        ro.UseMask = bool(raw["useMask"])
    if raw.get("maskColor") is not None:
        ro.MaskColor = _parse_color(raw["maskColor"])
    if raw.get("maxMatchCount") is not None:
        ro.MaxMatchCount = int(_number(raw["maxMatchCount"], f"objects.{object_name}.maxMatchCount"))
    for key, target in (
        ("allContains", ro.AllContainMatchText),
        ("oneContains", ro.OneContainMatchText),
        ("regex", ro.RegexMatchText),
    ):
        values = raw.get(key) or []
        if not isinstance(values, list):
            raise RecognitionJsonError(f"objects.{object_name}.{key} 必须是数组")
        target.extend(str(value) for value in values)
    if raw.get("text") is not None:
        ro.Text = str(raw["text"])

    reference = raw.get("reference")
    if reference is not None:
        if not isinstance(reference, Mapping):
            raise RecognitionJsonError(f"objects.{object_name}.reference 必须是对象")
        size = reference.get("size")
        if not isinstance(size, list) or len(size) != 2:
            raise RecognitionJsonError(f"objects.{object_name}.reference.size 必须是 [width, height]")
        ro.ReferenceImageSize = tuple(
            _number(value, f"objects.{object_name}.reference.size") for value in size
        )
        if reference.get("bbox") is not None:
            ro.ReferenceBoundingBox = evaluator.evaluate_rect(
                reference["bbox"], f"objects.{object_name}.reference.bbox",
            )

    search = raw.get("search")
    if search is not None:
        if not isinstance(search, Mapping):
            raise RecognitionJsonError(f"objects.{object_name}.search 必须是对象")
        options = SearchOptions()
        if search.get("anchor") is not None:
            options.AnchorMode = str(search["anchor"])
        if search.get("box") is not None:
            options.ReferenceSearchBox = evaluator.evaluate_rect(
                search["box"], f"objects.{object_name}.search.box",
            )
        if search.get("expand") is not None:
            expand = search["expand"]
            if not isinstance(expand, list) or len(expand) != 2:
                raise RecognitionJsonError(f"objects.{object_name}.search.expand 必须是 [width, height]")
            options.ExpandSize = tuple(
                _number(value, f"objects.{object_name}.search.expand") for value in expand
            )
        if search.get("expandPercent") is not None:
            values = search["expandPercent"]
            if not isinstance(values, list) or len(values) not in {1, 2, 4}:
                raise RecognitionJsonError(
                    f"objects.{object_name}.search.expandPercent 必须包含 1、2 或 4 个数字"
                )
            numbers = [
                _number(value, f"objects.{object_name}.search.expandPercent")
                for value in values
            ]
            if any(value < 0 for value in numbers):
                raise RecognitionJsonError(
                    f"objects.{object_name}.search.expandPercent 不能为负数"
                )
            options.ExpandPercent = numbers
        ro.SearchOptions = options

    if recognition_type == "TemplateMatch":
        template_name = _resolve_alias(raw.get("template"), document.get("templates") or {}, f"objects.{object_name}.template")
        mode = _template_mode(raw.get("templateMode"))
        if template_loader is not None:
            loaded = template_loader(template_name, mode)
            template = loaded if isinstance(loaded, Mat) else _load_template(Path(loaded), mode)
        else:
            roots = []
            if template_root is not None:
                base = Path(template_root).expanduser()
                roots.extend((
                    base / f"{capture_width}x{capture_height}",
                    base / "1920x1080",
                    base,
                ))
            if config_path is not None:
                roots.extend((
                    config_path.parent / f"{capture_width}x{capture_height}",
                    config_path.parent / "1920x1080",
                    config_path.parent,
                ))
            candidates = [root / template_name for root in roots]
            template_path = next((path for path in candidates if path.is_file()), None)
            if template_path is None:
                raise FileNotFoundError(
                    f"未找到 {object_name} 的模板 {template_name}，搜索路径: {candidates}"
                )
            template = _load_template(template_path, mode)
        ro.TemplateImageMat = template
        ro.InitTemplate()
    return ro


class RecognitionAssetStore:
    """Cached task-level Recognition.json loader matching BetterGI's API shape."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self._cache: dict[tuple[str, str, int, int], RecognitionObject] = {}

    def _config_path(self, task_name: str) -> Path:
        candidates = (
            self.root / task_name / "Assets" / "Recognition.json",
            self.root / task_name / "Recognition.json",
            self.root / "templates" / task_name.casefold() / "Recognition.json",
        )
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(f"未找到 {task_name} 的 Recognition.json: {candidates}")

    def get(self, task_name: str, object_name: str,
            capture_width: int = 1920, capture_height: int = 1080) -> RecognitionObject:
        key = (str(task_name), str(object_name), int(capture_width), int(capture_height))
        if key not in self._cache:
            config_path = self._config_path(key[0])
            self._cache[key] = load_recognition_object(
                config_path, key[1], capture_width=key[2], capture_height=key[3],
                template_root=config_path.parent,
            )
        return self._cache[key]

    Get = get

    def clear(self, task_name: str, object_name: str) -> None:
        task_name, object_name = str(task_name), str(object_name)
        self._cache = {
            key: value for key, value in self._cache.items()
            if key[:2] != (task_name, object_name)
        }

    Clear = clear

    def clear_task(self, task_name: str) -> None:
        task_name = str(task_name)
        self._cache = {
            key: value for key, value in self._cache.items()
            if key[0] != task_name
        }

    ClearTask = clear_task

    def clear_all(self) -> None:
        self._cache.clear()

    ClearAll = clear_all
