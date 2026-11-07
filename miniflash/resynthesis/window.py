# A window is a box of tiles over a range of steps inside one stage, where every qubit keeps its tile, chosen so that
# every merge component of those steps lies entirely inside the box or entirely outside it. Nothing crosses its side
# faces, only the qubit pipes through its bottom and top and the magic states it consumes, so a cell synthesised for
# its pairs with the same ports can replace it. Qubits inside the box that join no component are bystanders and pass
# straight through, as do idle magic tiles. A slab is a step range with the closed windows that cover every component
# in it, and it is what extraction scores, since a window only shortens the program when its whole slab does.
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .. import constant
from ..circuit import Problem
from ..mapping import Coord
from ..program import IDLE, MAGIC, QUBIT, ROUTE, Program, Step


# Inclusive tile bounds, rows r0..r1 and columns c0..c1.
@dataclass(frozen=True)
class Box:
    r0: int
    c0: int
    r1: int
    c1: int

    @classmethod
    def of(cls, tiles: Iterable[Coord]) -> Box:
        tiles = list(tiles)
        rows = [tile[0] for tile in tiles]
        cols = [tile[1] for tile in tiles]
        return cls(min(rows), min(cols), max(rows), max(cols))

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def width(self) -> int:
        return self.c1 - self.c0 + 1

    @property
    def area(self) -> int:
        return self.height * self.width

    def contains(self, tile: Coord) -> bool:
        return self.r0 <= tile[0] <= self.r1 and self.c0 <= tile[1] <= self.c1

    def overlaps(self, other: Box) -> bool:
        return not (self.r1 < other.r0 or other.r1 < self.r0 or self.c1 < other.c0 or other.c1 < self.c0)

    def union(self, other: Box) -> Box:
        return Box(min(self.r0, other.r0), min(self.c0, other.c0), max(self.r1, other.r1), max(self.c1, other.c1))

    def tiles(self) -> list[Coord]:
        return [(row, col) for row in range(self.r0, self.r1 + 1) for col in range(self.c0, self.c1 + 1)]

    def to_dict(self) -> dict:
        return {"r0": self.r0, "c0": self.c0, "r1": self.r1, "c1": self.c1}


# One merge of one step, with the pair it realises, every tile it occupies and the qubits it joins.
@dataclass(frozen=True)
class Component:
    step: int
    pair: int
    tiles: frozenset[Coord]        # route tiles and both parties, the consumed magic tile included
    qubits: tuple[int, ...]
    magic: Optional[Coord] = None   # the magic tile consumed

    @property
    def box(self) -> Box:
        return Box.of(self.tiles)

    @classmethod
    def of_step(cls, step: Step, step_index: int) -> list[Component]:
        # The merge components of the step, in pair order.
        components = []
        for tiles in step.components():
            vertices = [step.vertices[tile] for tile in tiles]
            ids = ({vertex.id for vertex in vertices if vertex.kind == ROUTE}
                   | {vertex.id for vertex in vertices if vertex.kind == MAGIC and vertex.id != IDLE})
            (pair,) = ids
            qubits = tuple(sorted(vertex.id for vertex in vertices if vertex.kind == QUBIT))
            magic_tiles = [vertex.tile for vertex in vertices if vertex.kind == MAGIC]
            components.append(cls(step_index, pair, frozenset(tiles), qubits, magic_tiles[0] if magic_tiles else None))
        return sorted(components, key=lambda component: component.pair)


# A closed box over steps start..stop-1 and the components inside it.
@dataclass(frozen=True)
class Window:
    box: Box
    start: int
    stop: int
    components: tuple[Component, ...]
    bystanders: dict[int, Coord]    # the tile of every qubit inside the box that joins no component
    idle_magic: tuple[Coord, ...]   # magic tiles inside the box never consumed in the range

    def __post_init__(self):
        # Components in execution order, by step then pair.
        object.__setattr__(self, "components",
                           tuple(sorted(self.components, key=lambda component: (component.step, component.pair))))
        object.__setattr__(self, "bystanders", dict(self.bystanders))
        object.__setattr__(self, "idle_magic", tuple(sorted(self.idle_magic)))

    @property
    def steps(self) -> int:
        return self.stop - self.start

    @property
    def pairs(self) -> tuple[int, ...]:
        return tuple(component.pair for component in self.components)   # in execution order

    @property
    def participants(self) -> tuple[int, ...]:
        return tuple(sorted({qubit for component in self.components for qubit in component.qubits}))

    @property
    def magic(self) -> tuple[tuple[int, Coord], ...]:
        # The (step, tile) of every magic state consumed, in step order.
        return tuple((component.step, component.magic) for component in self.components if component.magic is not None)

    @property
    def ports(self) -> int:
        # Two per participant, in and out, plus one per magic state consumed.
        return 2 * len(self.participants) + len(self.magic)

    @property
    def cubes(self) -> int:
        return self.box.area * constant.LAYERS_PER_STEP * self.steps

    def critical_path(self, problem: Problem) -> int:
        # The longest dependency chain among the window's pairs, in pairs.
        level: dict[int, int] = {}
        for pair in self.pairs:
            level[pair] = max((level[predecessor] + 1 for predecessor in problem.preds[pair] if predecessor in level),
                              default=1)
        return max(level.values(), default=0)

    def slack(self, problem: Problem) -> int:
        # Steps a depth-optimal cell could save at best, one pair per step being the floor this assumes.
        return self.steps - self.critical_path(problem)

    def check(self, program: Program) -> None:
        # Raises ValueError unless the window is closed on program, with its components exactly those inside the box.
        mine = {(component.step, component.pair) for component in self.components}
        seen = set()
        for step_index in range(self.start, self.stop):
            step = program.steps[step_index]
            for component in Component.of_step(step, step_index):
                inside = [self.box.contains(tile) for tile in component.tiles]
                if any(inside) and not all(inside):
                    raise ValueError(f"step {step_index}: pair {component.pair} crosses the box {self.box}")
                if all(inside):
                    if (step_index, component.pair) not in mine:
                        raise ValueError(f"step {step_index}: pair {component.pair} inside the box but not in the window")
                    seen.add((step_index, component.pair))

            seats = step.qubit_tiles()
            for component in self.components:
                for qubit in component.qubits:
                    if not self.box.contains(seats[qubit]):
                        raise ValueError(f"step {step_index}: participant q{qubit} at {seats[qubit]} outside the box")

        if seen != mine:
            raise ValueError(f"window lists pairs {sorted(mine - seen)} that its steps do not hold")

    def to_dict(self) -> dict:
        return {"box": self.box.to_dict(), "start": self.start, "stop": self.stop,
                "pairs": list(self.pairs), "participants": list(self.participants),
                "bystanders": {qubit: list(tile) for qubit, tile in sorted(self.bystanders.items())},
                "idle_magic": [list(tile) for tile in self.idle_magic],
                "magic": [[step, list(tile)] for step, tile in self.magic]}


# What one lassynth call may be asked to solve, None lifting a limit.
@dataclass(frozen=True)
class Budget:
    area: Optional[int] = constant.WINDOW_AREA
    steps: Optional[int] = constant.WINDOW_STEPS
    ports: Optional[int] = constant.WINDOW_PORTS
    cubes: Optional[int] = None

    def fits(self, window: Window) -> bool:
        return ((self.area is None or window.box.area <= self.area)
                and (self.steps is None or window.steps <= self.steps)
                and (self.ports is None or window.ports <= self.ports)
                and (self.cubes is None or window.cubes <= self.cubes))


# A step range and the closed windows that together hold every component in it.
@dataclass(frozen=True)
class Slab:
    start: int
    stop: int
    windows: tuple[Window, ...]

    @property
    def steps(self) -> int:
        return self.stop - self.start

    @property
    def pairs(self) -> tuple[int, ...]:
        return tuple(sorted(pair for window in self.windows for pair in window.pairs))

    def depth(self, problem: Problem) -> int:
        # Steps the slab needs at best, the longest critical path over its windows.
        return max((window.critical_path(problem) for window in self.windows), default=0)

    def gain(self, problem: Problem) -> int:
        # An upper bound on the steps the slab can lose, zero for a slab without components.
        if not self.windows:
            return 0
        return self.steps - self.depth(problem)

    def holes(self, problem: Problem) -> int:
        # Tile-steps left empty above the windows whose critical path falls short of the slab's depth.
        depth = self.depth(problem)
        return sum((depth - window.critical_path(problem)) * window.box.area for window in self.windows)

    def cuts(self, problem: Problem) -> int:
        # Dependency edges entering the slab from outside, each a qubit whose port the cut pins between two cells.
        mine = set(self.pairs)
        return sum(1 for pair in mine for predecessor in problem.preds[pair] if predecessor not in mine)

    def overlaps(self, other: Slab) -> bool:
        return self.start < other.stop and other.start < self.stop

    def to_dict(self) -> dict:
        return {"start": self.start, "stop": self.stop, "windows": [window.to_dict() for window in self.windows]}
