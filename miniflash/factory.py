# Magic-state factories: charge a finished program for the supply behind its magic tiles.
# A program consumes a state whenever a MAGIC vertex carries a pair id. The factories sit off the
# chip, one of height x width tiles yielding a state every period steps for the whole program, so
# factory_volume = factories x height x width x steps. The sizing says how many factories a program
# needs: "peak" is the most states one step consumes, "rate" the most in any window of period steps,
# "tiles" every magic tile consumed at least once.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import constant
from .mapping import Coord
from .program import IDLE, MAGIC, Program

SIZINGS = ("peak", "rate", "tiles")


@dataclass(frozen=True)
class Factory:
    height: int = constant.FACTORY_HEIGHT
    width: int = constant.FACTORY_WIDTH
    period: int = constant.FACTORY_PERIOD   # steps per magic state
    sizing: str = constant.FACTORY_SIZING

    @property
    def area(self) -> int:
        return self.height * self.width

    @classmethod
    def parse(cls, text: str) -> Factory:
        # "HxWxP" or "HxWxP:sizing"
        dims, _, sizing = text.partition(":")
        height, width, period = (int(part) for part in dims.lower().split("x"))
        return cls(height, width, period, sizing or cls.sizing)

    def __str__(self) -> str:
        return f"{self.height}x{self.width}x{self.period}:{self.sizing}"


@dataclass(frozen=True)
class Bill:
    factory: Factory
    steps: int
    consumed: tuple[int, ...]   # states consumed per step
    tiles: set[Coord]           # magic tiles consumed at least once
    factories: int

    @property
    def states(self) -> int:
        return sum(self.consumed)

    @property
    def peak(self) -> int:
        return max(self.consumed, default=0)

    @property
    def factory_volume(self) -> int:
        return self.factories * self.factory.area * self.steps

    def total(self, program_volume: int) -> int:
        return program_volume + self.factory_volume


def supply_limits(factory: Optional[Factory], factories: int = 0) -> tuple[int, int, int]:
    # (cooldown, factories, period) for the router and the verifier. No factory leaves the supply
    # unlimited; a factory alone puts one behind every magic tile, so a tile yields a state every
    # period steps; factories = F > 0 shares F of them over the whole chip, F states per period.
    if factories:
        return 1, factories, (factory or Factory()).period
    if factory is None:
        return 1, 0, 1
    return factory.period, 0, factory.period


def demand(program: Program) -> tuple[list[int], set[Coord]]:
    consumed = []
    tiles = set()
    for step in program.steps:
        used = [vertex.tile for vertex in step.of_kind(MAGIC) if vertex.id != IDLE]
        consumed.append(len(used))
        tiles.update(used)
    return consumed, tiles


def count(consumed: list[int], tiles: set[Coord], factory: Factory) -> int:
    if factory.sizing == "tiles":
        return len(tiles)
    if factory.sizing == "peak":
        return max(consumed, default=0)
    window = factory.period
    return max((sum(consumed[i:i + window]) for i in range(max(1, len(consumed) - window + 1))), default=0)


def charge(program: Program, factory: Factory = Factory(), factories: Optional[int] = None) -> Bill:
    # A given factory count overrides the sizing, for pricing the supply the compiler was given.
    consumed, tiles = demand(program)
    return Bill(factory, len(program.steps), tuple(consumed), tiles,
                factories if factories is not None else count(consumed, tiles, factory))
