# Program: the solver's output, one sparse step graph per time step. Each step lists the non-free tiles as
# vertices (coordinate, kind, id) and the merges between them as undirected edges. Kinds are QUBIT with the
# qubit index, MAGIC with the consuming pair or IDLE, and ROUTE with the pair. A pair that executes at a step
# is the connected component holding its parties, and volume is steps x height x width.
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .circuit import Problem
from .mapping import Coord, Mapping
from .route import Route, check_route

QUBIT = "qubit"
MAGIC = "magic"
ROUTE = "route"
IDLE = -1
Edge = frozenset[Coord]


@dataclass(frozen=True)
class Vertex:
    tile: Coord
    kind: str
    id: int

    def __post_init__(self):
        object.__setattr__(self, "tile", tuple(self.tile))


@dataclass
class Step:
    vertices: dict[Coord, Vertex] = field(default_factory=dict)
    edges: set[Edge] = field(default_factory=set)

    def add(self, tile: Coord, kind: str, id: int) -> Vertex:
        tile = tuple(tile)
        vertex = Vertex(tile, kind, id)
        self.vertices[tile] = vertex
        return vertex

    def link(self, tile_a: Coord, tile_b: Coord) -> None:
        tile_a = tuple(tile_a)
        tile_b = tuple(tile_b)
        self.edges.add(frozenset((tile_a, tile_b)))

    def of_kind(self, kind: str) -> list[Vertex]:
        return [vertex for vertex in self.vertices.values() if vertex.kind == kind]

    def qubit_tiles(self) -> dict[int, Coord]:
        return {vertex.id: vertex.tile for vertex in self.vertices.values() if vertex.kind == QUBIT}

    def neighbours(self) -> dict[Coord, set[Coord]]:
        neighbours: dict[Coord, set[Coord]] = {tile: set() for tile in self.vertices}
        for edge in self.edges:
            tile_a, tile_b = tuple(edge)
            neighbours[tile_a].add(tile_b)
            neighbours[tile_b].add(tile_a)
        return neighbours

    def components(self) -> list[frozenset[Coord]]:
        # Connected components with at least one edge, as tile sets.
        neighbours = self.neighbours()
        seen: set[Coord] = set()
        out = []
        for tile in self.vertices:
            if tile in seen or not neighbours[tile]:
                continue
            component = set()
            stack = [tile]
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                stack.extend(neighbours[current] - component)
            seen |= component
            out.append(frozenset(component))
        return out

    @property
    def is_walk(self) -> bool:
        return not self.edges

    def to_dict(self) -> dict:
        return {"vertices": [[vertex.tile[0], vertex.tile[1], vertex.kind, vertex.id]
                             for vertex in sorted(self.vertices.values(), key=lambda vertex: vertex.tile)],
                "edges": sorted([sorted(edge) for edge in self.edges])}

    @classmethod
    def from_dict(cls, data: dict) -> Step:
        step = cls()
        for row, col, kind, id in data["vertices"]:
            step.add((row, col), kind, id)
        for tile_a, tile_b in data["edges"]:
            step.link(tuple(tile_a), tuple(tile_b))
        return step


@dataclass
class Program:
    height: int
    width: int
    steps: list[Step] = field(default_factory=list)

    @property
    def area(self) -> int:
        return self.height * self.width

    @property
    def volume(self) -> int:
        return self.area * len(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    @classmethod
    def from_routes(cls, problem: Problem, mapping: Mapping, schedule: Sequence[Sequence[Route]]) -> Program:
        # Every route is checked against its pair and the mapping, and its edges run from the Z party along the
        # tiles to the X party or the magic tile.
        prog = cls(mapping.height, mapping.width)
        for routes in schedule:
            step = Step()
            for qubit, tile in mapping.qubits.items():
                step.add(tile, QUBIT, qubit)
            consumed = {route.magic: route.pair for route in routes if route.magic is not None}
            for magic_tile in mapping.magic:
                step.add(magic_tile, MAGIC, consumed.get(magic_tile, IDLE))

            for route in routes:
                pair = problem.pairs[route.pair]
                check_route(route, pair, mapping)
                tiles = route.tiles()
                for tile in tiles:
                    step.add(tile, ROUTE, route.pair)
                step.link(mapping.qubits[pair.z], tiles[0])
                for tile_a, tile_b in zip(tiles, tiles[1:]):
                    step.link(tile_a, tile_b)
                step.link(tiles[-1], route.magic if pair.is_magic else mapping.qubits[pair.x])
            prog.steps.append(step)
        return prog

    def to_dict(self) -> dict:
        return {"height": self.height, "width": self.width, "steps": [step.to_dict() for step in self.steps]}

    @classmethod
    def from_dict(cls, data: dict) -> Program:
        return cls(data["height"], data["width"], [Step.from_dict(step) for step in data["steps"]])

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path) -> Program:
        return cls.from_dict(json.loads(Path(path).read_text()))
