# A cell is a window and the lassynth solution that replaces it, lassynth's arrays after its default clean-up, which
# drops unconnected pieces and colours the K pipes. Coordinates inside a cell are lassynth's, so cube (i, j, k) is
# column box.c0 + i, row box.r0 + j and layer k - 1 of the cell, layer 0 holding only the open ends of the input ports.
# The pipes are drawn so that a qubit pipe is red and a merge leaving a qubit through its Z side is blue, as in the
# tile program, whenever the solution changes no colour along time. A K pipe whose two ends disagree carries a Hadamard
# wall and is drawn yellow, and since lassynth does not charge for walls a solution may hold more than it needs.
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import NamedTuple

from .window import Box, Component, Window


class CellPipe(NamedTuple):
    axis: str                     # I, J or K
    cube: tuple[int, int, int]    # the pipe runs from this cube to its successor along axis
    parity: int                   # 0 red, 1 blue
    wall: bool                    # a K pipe that changes colour, a Hadamard


@dataclass(frozen=True)
class Cell:
    window: Window
    depth: int              # layers
    lasre: dict
    proven: bool = False    # every depth below was shown unsatisfiable

    @classmethod
    def of(cls, window: Window, lasre: dict, proven: bool = False) -> Cell:
        # Runs lassynth's default clean-up on a copy of the raw solution.
        from lassynth import LatticeSurgerySolution
        cleaned = LatticeSurgerySolution(lasre=copy.deepcopy(lasre)).after_default_optimizations().lasre
        return cls(window, cleaned["n_k"] - 1, cleaned, proven)

    @property
    def saved(self) -> int:
        # Layers below the window's own.
        return 2 * self.window.steps - self.depth

    def pipes(self) -> list[CellPipe]:
        arrays = self.lasre
        pipes = []
        for i in range(arrays["n_i"]):   # i, j, k index the cubes along the lassynth axes I, J and K
            for j in range(arrays["n_j"]):
                for k in range(arrays["n_k"]):
                    if arrays["ExistI"][i][j][k]:
                        pipes.append(CellPipe("I", (i, j, k), 1 - int(arrays["ColorI"][i][j][k]), False))
                    if arrays["ExistJ"][i][j][k]:
                        pipes.append(CellPipe("J", (i, j, k), int(arrays["ColorJ"][i][j][k]), False))
                    if arrays["ExistK"][i][j][k]:
                        colour_below = int(arrays["ColorKM"][i][j][k])
                        colour_above = int(arrays["ColorKP"][i][j][k])
                        pipes.append(CellPipe("K", (i, j, k), colour_below, colour_below != colour_above))
        return pipes

    def y_cubes(self) -> list[tuple[int, int, int]]:
        arrays = self.lasre
        return [(i, j, k) for i in range(arrays["n_i"]) for j in range(arrays["n_j"]) for k in range(arrays["n_k"])
                if arrays["NodeY"][i][j][k]]

    def magic_cubes(self) -> list[tuple[int, int, int]]:
        # The open cubes of the magic ports, in port order.
        n_qubit_ports = 2 * len(self.window.participants)
        return [tuple(cube) for cube in self.lasre["port_cubes"][n_qubit_ports:]]

    @property
    def walls(self) -> int:
        return sum(1 for pipe in self.pipes() if pipe.wall)

    def to_dict(self) -> dict:
        window = self.window
        return {"window": {"box": window.box.to_dict(), "start": window.start, "stop": window.stop,
                           "components": [[component.step, component.pair, sorted(component.tiles),
                                           list(component.qubits), component.magic] for component in window.components],
                           "bystanders": {str(qubit): list(tile) for qubit, tile in window.bystanders.items()},
                           "idle_magic": [list(tile) for tile in window.idle_magic]},
                "depth": self.depth, "proven": self.proven, "lasre": json.loads(json.dumps(self.lasre, default=list))}

    @classmethod
    def from_dict(cls, data: dict) -> Cell:
        window_dict = data["window"]
        components = tuple(Component(step, pair, frozenset(tuple(tile) for tile in tiles), tuple(qubits),
                                     tuple(magic) if magic else None)
                           for step, pair, tiles, qubits, magic in window_dict["components"])
        window = Window(Box(**window_dict["box"]), window_dict["start"], window_dict["stop"], components,
                        {int(qubit): tuple(tile) for qubit, tile in window_dict["bystanders"].items()},
                        tuple(tuple(tile) for tile in window_dict["idle_magic"]))
        return cls(window, data["depth"], data["lasre"], data["proven"])
