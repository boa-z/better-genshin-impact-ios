"""BetterGI JSON combat strategy models and condition evaluation.

The upstream JSON strategy is a priority program, not a serialized TXT
strategy: one condition is selected per decision cycle and execution history
is visible to later conditions.  This module intentionally has no DeviceHub
dependency so strategy files can be parsed and validated offline.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


FUNCTION_NAMES = frozenset({
    "last-exec", "q-ready", "e-ready", "e-cd", "low-hp",
    "battle-time", "in-party", "onfield", "t", "since", "count",
    "min", "max", "last-check",
})


def _key(value: Any, name: str, default: Any = None) -> Any:
    if not isinstance(value, Mapping):
        return default
    wanted = name.replace("_", "").casefold()
    for candidate, result in value.items():
        if str(candidate).replace("_", "").casefold() == wanted:
            return result
    return default


def _text(value: Any) -> str:
    return "" if value is None else str(value)


@dataclass(frozen=True)
class JsonCondition:
    expression: str = ""


@dataclass(frozen=True)
class JsonMorePriority:
    expression: str = ""
    priority: int = 0


@dataclass(frozen=True)
class JsonAction:
    name: str = ""
    character: str = ""
    action: str = ""
    condition: JsonCondition = field(default_factory=JsonCondition)
    index: int = 0
    ensure_cast: bool = False
    more_priorities: tuple[JsonMorePriority, ...] = ()


@dataclass(frozen=True)
class JsonInfo:
    name: str = ""
    author: str = ""
    config: Mapping[str, Any] = field(default_factory=dict)
    pre_actions: tuple[str, ...] = ()


@dataclass(frozen=True)
class JsonCombatStrategy:
    info: JsonInfo
    actions: tuple[JsonAction, ...]

    @property
    def character_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(a.character for a in self.actions if a.character))


@dataclass(frozen=True)
class PrioritizedAction:
    action: JsonAction
    expression: str
    priority: int
    order: int


class StrategyFormatError(ValueError):
    pass


def _integer(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise StrategyFormatError(f"{field_name} 必须是整数") from error


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on"}
    return bool(value)


def parse_json_strategy(source: str | bytes | Mapping[str, Any]) -> JsonCombatStrategy:
    """Parse PascalCase/camelCase BetterGI JSON strategy data."""
    if isinstance(source, Mapping):
        raw: Any = source
    else:
        try:
            raw = json.loads(source)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as error:
            raise StrategyFormatError(f"JSON 战斗策略格式错误：{error}") from error
    if not isinstance(raw, Mapping):
        raise StrategyFormatError("JSON 战斗策略根节点必须是对象")

    info_raw = _key(raw, "Info")
    if not isinstance(info_raw, Mapping):
        raise StrategyFormatError("JSON 战斗策略缺少 Info 节点")
    pre_raw = _key(info_raw, "PreActions", [])
    if pre_raw is None:
        pre_raw = []
    if not isinstance(pre_raw, list):
        raise StrategyFormatError("Info.PreActions 必须是数组")
    config = _key(info_raw, "Config", {})
    info = JsonInfo(
        name=_text(_key(info_raw, "Name", "")),
        author=_text(_key(info_raw, "Author", "")),
        config=config if isinstance(config, Mapping) else {},
        pre_actions=tuple(_text(item) for item in pre_raw if _text(item).strip()),
    )

    actions_raw = _key(raw, "Actions")
    if not isinstance(actions_raw, list) or not actions_raw:
        raise StrategyFormatError("JSON 战斗策略中未定义任何动作")
    actions: list[JsonAction] = []
    for position, item in enumerate(actions_raw):
        if not isinstance(item, Mapping):
            raise StrategyFormatError(f"Actions[{position}] 必须是对象")
        condition_raw = _key(item, "Condition", {})
        if condition_raw is None:
            condition_raw = {}
        if not isinstance(condition_raw, Mapping):
            raise StrategyFormatError(f"Actions[{position}].Condition 必须是对象")
        more_raw = _key(item, "MorePriorities", [])
        if more_raw is None:
            more_raw = []
        if not isinstance(more_raw, list):
            raise StrategyFormatError(f"Actions[{position}].MorePriorities 必须是数组")
        more: list[JsonMorePriority] = []
        for extra_position, extra in enumerate(more_raw):
            if not isinstance(extra, Mapping):
                raise StrategyFormatError(
                    f"Actions[{position}].MorePriorities[{extra_position}] 必须是对象"
                )
            more.append(JsonMorePriority(
                expression=_text(_key(extra, "Expression", "")),
                priority=_integer(
                    _key(extra, "Priority", 0),
                    f"Actions[{position}].MorePriorities[{extra_position}].Priority",
                ),
            ))
        actions.append(JsonAction(
            name=_text(_key(item, "Name", "")),
            character=_text(_key(item, "Character", "")),
            action=_text(_key(item, "Action", "")),
            condition=JsonCondition(_text(_key(condition_raw, "Expression", ""))),
            index=_integer(_key(item, "Index", 0), f"Actions[{position}].Index"),
            ensure_cast=_boolean(_key(item, "EnsureCast", False)),
            more_priorities=tuple(more),
        ))

    names = [action.name for action in actions if action.name]
    for name in names:
        if not is_valid_action_name(name, names):
            raise StrategyFormatError(
                "JSON 战斗策略中动作名称无法作为条件标识符解析：" + name
            )
    return JsonCombatStrategy(info, tuple(actions))


def load_json_strategy(path: str | Path) -> JsonCombatStrategy:
    strategy_path = Path(path)
    if not strategy_path.is_file():
        raise FileNotFoundError(f"JSON 战斗策略文件不存在：{strategy_path}")
    return parse_json_strategy(strategy_path.read_text(encoding="utf-8-sig"))


def expand_priorities(
    strategy: JsonCombatStrategy,
    party_names: Iterable[str] | None = None,
) -> list[PrioritizedAction]:
    """Expand MorePriorities and apply BetterGI's stable ascending ordering."""
    party = {name.casefold() for name in party_names or () if name}
    filter_party = bool(party)
    result: list[PrioritizedAction] = []
    order = 0
    for action in strategy.actions:
        if filter_party and action.character and action.character.casefold() not in party:
            continue
        result.append(PrioritizedAction(
            action, action.condition.expression, action.index, order,
        ))
        order += 1
        for extra in action.more_priorities:
            result.append(PrioritizedAction(
                action, extra.expression, extra.priority, order,
            ))
            order += 1
    return sorted(result, key=lambda item: (item.priority, item.order))


class _Kind:
    IDENT = "ident"
    NUMBER = "number"
    BOOL = "bool"
    END = "end"


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str


_SINGLE_TOKENS = {
    "(": "(", ")": ")", ",": ",", "!": "!", "+": "+",
    "-": "-", "*": "*", "/": "/", ">": ">", "<": "<", "=": "=",
}


def _is_letter(character: str) -> bool:
    return character.isalpha()


def _tokenize(expression: str, known_identifiers: Iterable[str]) -> list[_Token]:
    known = {name.casefold() for name in known_identifiers}
    result: list[_Token] = []
    index = 0
    while index < len(expression):
        char = expression[index]
        if char.isspace():
            index += 1
            continue
        pair = expression[index:index + 2]
        if pair in {"&&", "||"}:
            result.append(_Token(pair, pair))
            index += 2
            continue
        if char in _SINGLE_TOKENS:
            result.append(_Token(_SINGLE_TOKENS[char], char))
            index += 1
            continue
        if char.isdigit() or (
            char == "." and index + 1 < len(expression) and expression[index + 1].isdigit()
        ):
            start = index
            while index < len(expression) and (
                expression[index].isdigit() or expression[index] == "."
            ):
                index += 1
            value = expression[start:index]
            try:
                float(value)
            except ValueError as error:
                raise StrategyFormatError(f"无效数字：{value}") from error
            result.append(_Token(_Kind.NUMBER, value))
            continue
        if _is_letter(char):
            start = index
            while index < len(expression) and expression[index].isalnum():
                index += 1
            base_end = index
            # Community strategies contain declared names such as
            # ``木偶-喷冰4.15``.  A dot is part of an identifier only when the
            # complete longest candidate is present in the declared-name set;
            # ordinary arithmetic such as ``t-4.15`` still tokenizes as a
            # subtraction followed by a decimal number.
            if index < len(expression) and expression[index] in {"-", "."}:
                candidate_end = index + 1
                while candidate_end < len(expression) and (
                    expression[candidate_end].isalnum()
                    or expression[candidate_end] in {"-", "."}
                ):
                    candidate_end += 1
                while candidate_end > base_end and (
                    expression[start:candidate_end].casefold() not in known
                ):
                    candidate_end -= 1
                if candidate_end > base_end:
                    index = candidate_end
            word = expression[start:index]
            kind = _Kind.BOOL if word.casefold() in {"true", "false"} else _Kind.IDENT
            result.append(_Token(kind, word))
            continue
        raise StrategyFormatError(f"无法识别的字符：'{char}'")
    result.append(_Token(_Kind.END, ""))
    return result


def is_valid_action_name(name: str, all_action_names: Iterable[str]) -> bool:
    if not name or name.casefold() in {"true", "false"}:
        return False
    if name.casefold() in FUNCTION_NAMES:
        return False
    try:
        tokens = _tokenize(name, [*FUNCTION_NAMES, *all_action_names])
    except StrategyFormatError:
        return False
    return (
        len(tokens) == 2
        and tokens[0].kind == _Kind.IDENT
        and tokens[0].value.casefold() == name.casefold()
    )


@dataclass(frozen=True)
class _Node:
    kind: str
    value: Any = None
    children: tuple["_Node", ...] = ()


class _Parser:
    def __init__(self, tokens: Sequence[_Token]):
        self.tokens = tokens
        self.position = 0

    @property
    def token(self) -> _Token:
        return self.tokens[self.position]

    def take(self, kind: str) -> _Token:
        if self.token.kind != kind:
            raise StrategyFormatError(f"期望 {kind}，实际 {self.token.value or self.token.kind}")
        token = self.token
        self.position += 1
        return token

    def parse(self) -> _Node:
        node = self._or()
        if self.token.kind != _Kind.END:
            raise StrategyFormatError(f"多余的 token：{self.token.value}")
        return node

    def _binary(self, child: Callable[[], _Node], operators: set[str]) -> _Node:
        left = child()
        while self.token.kind in operators:
            operator = self.token.value
            self.position += 1
            left = _Node("binary", operator, (left, child()))
        return left

    def _or(self) -> _Node:
        return self._binary(self._and, {"||"})

    def _and(self) -> _Node:
        return self._binary(self._compare, {"&&"})

    def _compare(self) -> _Node:
        return self._binary(self._add, {">", "<", "="})

    def _add(self) -> _Node:
        return self._binary(self._multiply, {"+", "-"})

    def _multiply(self) -> _Node:
        return self._binary(self._unary, {"*", "/"})

    def _unary(self) -> _Node:
        if self.token.kind in {"!", "-"}:
            operator = self.token.value
            self.position += 1
            return _Node("unary", operator, (self._unary(),))
        return self._primary()

    def _primary(self) -> _Node:
        token = self.token
        if token.kind == "(":
            self.position += 1
            node = self._or()
            self.take(")")
            return node
        if token.kind == _Kind.NUMBER:
            self.position += 1
            return _Node("number", float(token.value))
        if token.kind == _Kind.BOOL:
            self.position += 1
            return _Node("bool", token.value.casefold() == "true")
        if token.kind == _Kind.IDENT:
            self.position += 1
            arguments: list[_Node] = []
            if self.token.kind == "(":
                self.position += 1
                if self.token.kind != ")":
                    arguments.append(self._or())
                    while self.token.kind == ",":
                        self.position += 1
                        arguments.append(self._or())
                self.take(")")
            return _Node("call", token.value, tuple(arguments))
        raise StrategyFormatError(f"意外的 token：{token.value or token.kind}")


@dataclass(frozen=True)
class _Event:
    index: int
    name: str
    at: float


@dataclass(frozen=True)
class _Target:
    index: int | None
    name: str | None


class ConditionEvaluator:
    """Evaluate BetterGI conditions with one-frame-per-cycle visual callbacks."""

    def __init__(
        self,
        *,
        action_names: Iterable[str] = (),
        party_names: Iterable[str] = (),
        active_character: Callable[[], str | None] | None = None,
        q_ready: Callable[[str | None, Any], bool] | None = None,
        e_cd: Callable[[str | None, Any], float] | None = None,
        low_hp: Callable[[Any], bool] | None = None,
        last_check: Callable[[], float] | None = None,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] = print,
    ):
        self.clock = clock
        self.log = log
        self.started_at = clock()
        self.action_names = {name.casefold(): name for name in action_names if name}
        self.party_names = {name.casefold() for name in party_names if name}
        self.active_character = active_character or (lambda: None)
        self.q_ready_callback = q_ready or (lambda _name, _frame: False)
        self.e_cd_callback = e_cd or (lambda _name, _frame: 0.0)
        self.low_hp_callback = low_hp or (lambda _frame: False)
        self.last_check_callback = last_check or (lambda: 0.0)
        self.history: list[_Event] = []
        self.frame: Any = None
        self.current_index = 0
        self.current_character: str | None = None
        self.current_action_name: str | None = None
        self._cycle_cache: dict[tuple[str, str], Any] = {}

    def set_frame(self, frame: Any) -> None:
        self.frame = frame
        self._cycle_cache.clear()

    def update_last_exec_time(self, index: int, name: str) -> None:
        self.history.append(_Event(index, name, self.clock() - self.started_at))

    def evaluate(
        self,
        expression: str,
        current_index: int,
        character_name: str | None = None,
        action_name: str | None = None,
    ) -> bool:
        self.current_index = current_index
        self.current_character = character_name
        self.current_action_name = action_name
        if not expression.strip():
            return True
        try:
            known = [*FUNCTION_NAMES, *self.action_names.values()]
            node = _Parser(_tokenize(expression, known)).parse()
            return self._to_bool(self._eval(node))
        except Exception as error:
            self.log(f"[AutoFight] 条件表达式求值失败：{expression}（{error}）")
            return False

    @staticmethod
    def _to_number(value: Any) -> float:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value > 0
        return False

    def _eval(self, node: _Node) -> Any:
        if node.kind in {"number", "bool"}:
            return node.value
        if node.kind == "unary":
            value = self._eval(node.children[0])
            return not self._to_bool(value) if node.value == "!" else -self._to_number(value)
        if node.kind == "binary":
            left = self._eval(node.children[0])
            if node.value == "&&":
                return self._to_bool(left) and self._to_bool(self._eval(node.children[1]))
            if node.value == "||":
                return self._to_bool(left) or self._to_bool(self._eval(node.children[1]))
            right = self._eval(node.children[1])
            left_number, right_number = self._to_number(left), self._to_number(right)
            return {
                ">": lambda: left_number > right_number,
                "<": lambda: left_number < right_number,
                "=": lambda: abs(left_number - right_number) < 0.0001,
                "+": lambda: left_number + right_number,
                "-": lambda: left_number - right_number,
                "*": lambda: left_number * right_number,
                "/": lambda: left_number / right_number if right_number != 0 else 0.0,
            }[node.value]()
        if node.kind == "call":
            return self._call(str(node.value).casefold(), node.children)
        return False

    def _identifier(self, node: _Node) -> str | None:
        return str(node.value) if node.kind == "call" and not node.children else None

    def _target(self, node: _Node | None) -> _Target:
        if node is None:
            return _Target(self.current_index, self.current_action_name)
        if node.kind == "number":
            return _Target(int(node.value), None)
        identifier = self._identifier(node)
        if identifier is not None:
            canonical = self.action_names.get(identifier.casefold())
            if canonical is None:
                raise StrategyFormatError(f"未知动作名称：{identifier}")
            return _Target(None, canonical)
        return _Target(int(self._to_number(self._eval(node))), None)

    @staticmethod
    def _matches(event: _Event, target: _Target) -> bool:
        if target.index is not None and target.name is not None:
            return event.index == target.index and event.name.casefold() == target.name.casefold()
        if target.name is not None:
            return event.name.casefold() == target.name.casefold()
        return event.index == target.index

    def _elapsed(self, target: _Target) -> float | None:
        now = self.clock() - self.started_at
        for event in reversed(self.history):
            if self._matches(event, target):
                return now - event.at
        return None

    def _visual(self, kind: str, name: str | None, callback: Callable[[], Any]) -> Any:
        cache_key = (kind, (name or "").casefold())
        if cache_key not in self._cycle_cache:
            self._cycle_cache[cache_key] = callback()
        return self._cycle_cache[cache_key]

    def _character_argument(self, args: Sequence[_Node]) -> str | None:
        if args:
            identifier = self._identifier(args[0])
            if identifier is not None:
                return identifier
        return self.current_character

    def _call(self, name: str, args: Sequence[_Node]) -> Any:
        if name == "t":
            return self.clock() - self.started_at
        if name == "last-check":
            return float(self.last_check_callback())
        if name == "since":
            elapsed = self._elapsed(self._target(args[0] if args else None))
            return math.inf if elapsed is None else elapsed
        if name == "count":
            now = self.clock() - self.started_at
            target = self._target(args[0] if args else None)
            start = self._to_number(self._eval(args[1])) if len(args) >= 2 else 0.0
            end = self._to_number(self._eval(args[2])) if len(args) >= 3 else now
            return float(sum(
                1 for event in self.history
                if self._matches(event, target) and start <= event.at <= end
            ))
        if name == "last-exec":
            if not args:
                return False
            seconds = self._to_number(self._eval(args[0]))
            greater = self._to_bool(self._eval(args[1])) if len(args) >= 2 else True
            elapsed = self._elapsed(self._target(args[2] if len(args) >= 3 else None))
            if elapsed is None:
                return greater
            return elapsed > seconds if greater else elapsed < seconds
        if name == "battle-time":
            if not args:
                return False
            seconds = self._to_number(self._eval(args[0]))
            greater = self._to_bool(self._eval(args[1])) if len(args) >= 2 else True
            elapsed = self.clock() - self.started_at
            return elapsed > seconds if greater else elapsed < seconds
        if name in {"min", "max"}:
            if not args:
                return 0.0
            values = [self._to_number(self._eval(argument)) for argument in args]
            return min(values) if name == "min" else max(values)
        if name == "in-party":
            identifier = self._identifier(args[0]) if args else None
            return bool(identifier and identifier.casefold() in self.party_names)
        if name == "onfield":
            active = self.active_character()
            return bool(
                self.current_character and active
                and self.current_character.casefold() == active.casefold()
            )
        if name == "q-ready":
            character = self._character_argument(args)
            return bool(self._visual(
                "q", character,
                lambda: self.q_ready_callback(character, self.frame),
            ))
        if name in {"e-ready", "e-cd"}:
            character = self._character_argument(args)
            remaining = float(self._visual(
                "e", character,
                lambda: self.e_cd_callback(character, self.frame),
            ))
            return remaining <= 0 if name == "e-ready" else max(0.0, remaining)
        if name == "low-hp":
            return bool(self._visual(
                "hp", None, lambda: self.low_hp_callback(self.frame),
            ))
        raise StrategyFormatError(f"未知条件函数：{name}")
