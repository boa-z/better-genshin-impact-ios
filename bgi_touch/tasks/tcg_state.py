"""Pure BetterGI Genius Invokation models and round-planning helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence


class TcgElement(str, Enum):
    OMNI = "omni"
    CRYO = "cryo"
    HYDRO = "hydro"
    PYRO = "pyro"
    ELECTRO = "electro"
    DENDRO = "dendro"
    ANEMO = "anemo"
    GEO = "geo"


ELEMENT_FROM_CHINESE = {
    "全": TcgElement.OMNI,
    "万能": TcgElement.OMNI,
    "冰": TcgElement.CRYO,
    "水": TcgElement.HYDRO,
    "火": TcgElement.PYRO,
    "雷": TcgElement.ELECTRO,
    "草": TcgElement.DENDRO,
    "风": TcgElement.ANEMO,
    "岩": TcgElement.GEO,
}


class TcgPhase(str, Enum):
    UNKNOWN = "unknown"
    PREPARE = "prepare"
    CHARACTER_PICK = "character_pick"
    ROLL = "roll"
    MY_ACTION = "my_action"
    OPPONENT_ACTION = "opponent_action"
    END_PHASE = "end_phase"
    CHARACTER_TAKEN_OUT = "character_taken_out"
    DUEL_END = "duel_end"


@dataclass(frozen=True)
class TcgSkill:
    index: int
    element: TcgElement
    specific_element_cost: int
    any_element_cost: int = 0
    name: str = ""

    @property
    def all_cost(self) -> int:
        return self.specific_element_cost + self.any_element_cost


@dataclass
class TcgCharacter:
    index: int
    name: str
    element: TcgElement
    skills: dict[int, TcgSkill]
    defeated: bool = False
    active: bool = False
    statuses: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class TcgCommand:
    character: str
    skill: int
    dice_delta: int = 0


def wanted_elements(
    commands: Sequence[TcgCommand],
    characters: Mapping[int, TcgCharacter],
    *,
    dice_count: int = 8,
    current_character: int | None = None,
) -> set[TcgElement]:
    """Predict elements usable this round using BetterGI's queue budgeting."""
    by_name = {character.name: character for character in characters.values()}
    wanted = {TcgElement.OMNI}
    spent = 0
    previous = current_character
    for command in commands:
        character = by_name[command.character]
        if character.defeated:
            continue
        if previous is not None and previous != character.index:
            spent += 1
            if spent > dice_count:
                break
        skill = character.skills[command.skill]
        spent += max(0, skill.all_cost + command.dice_delta)
        if spent > dice_count:
            break
        wanted.add(skill.element)
        previous = character.index
    return wanted


def reroll_indices(
    dice: Sequence[TcgElement], wanted: Iterable[TcgElement]
) -> list[int]:
    """Return the dice indexes to select during the roll phase."""
    keep = set(wanted) | {TcgElement.OMNI}
    return [index for index, element in enumerate(dice) if element not in keep]


def tuning_card_count(
    dice: Mapping[TcgElement, int], skill: TcgSkill
) -> int:
    """Cards required to satisfy the skill's specific-element component."""
    available = int(dice.get(TcgElement.OMNI, 0)) + int(dice.get(skill.element, 0))
    return max(0, skill.specific_element_cost - available)


def effective_skill_cost(skill: TcgSkill, dice_delta: int = 0) -> int:
    return max(0, skill.all_cost + int(dice_delta))


def next_living_character(
    commands: Sequence[TcgCommand], characters: Mapping[int, TcgCharacter]
) -> TcgCharacter | None:
    """Use the remaining strategy order, matching BetterGI's defeat recovery."""
    by_name = {character.name: character for character in characters.values()}
    seen: set[int] = set()
    for command in commands:
        character = by_name[command.character]
        if character.index not in seen and not character.defeated:
            return character
        seen.add(character.index)
    return next((character for character in characters.values() if not character.defeated), None)
