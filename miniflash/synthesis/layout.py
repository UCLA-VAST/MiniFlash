# Layout selection: the shape of the site array. candidate_shapes lists every a x b array with a <= b that holds
# the qubits, by paired-lane grid area, and the compiler tries the first N plus the squarest feasible shape,
# since the flat shapes that minimise area are the ones the budgeted router handles worst. shape_estimates prices
# each row count by full grid area times the layer capacity, a step carrying two T gates and LAYOUT_CX_PER_ROW
# CX pairs per site row, so the statistics policy compiles only its preferred shape. Predictions ignore routes,
# compaction and walks.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..circuit import Problem
from ..constant import LAYOUT_CX_PER_ROW
from ..mapping import Mapping

Shape = tuple[int, int]


def candidate_shapes(n_qubits: int, count: Optional[int] = None, squarest: bool = False) -> list[Shape]:
    # (site_rows, site_cols) with rows <= cols, smallest paired-lane grid area first.
    by_area = []
    for rows in range(1, n_qubits + 1):
        cols = -(-n_qubits // rows)
        if rows > cols:
            break
        mapping = Mapping.paired(rows, cols, n_qubits)
        by_area.append((mapping.height * mapping.width, rows, cols))
    by_area.sort()
    shapes = [(rows, cols) for _, rows, cols in by_area]
    if count is None:
        return shapes

    chosen = shapes[:count]
    if squarest:
        squarest_shape = min(shapes, key=lambda shape: (shape[1] - shape[0], shape))
        if squarest_shape not in chosen:
            chosen.append(squarest_shape)
    return chosen


# A prediction of the shape model, not a routed or verified program.
@dataclass(frozen=True)
class ShapeEstimate:
    shape: Shape
    area: int
    steps: int

    @property
    def volume(self) -> int:
        return self.area * self.steps


def shape_estimates(problem: Problem, cap: float = LAYOUT_CX_PER_ROW) -> list[ShapeEstimate]:
    # Row counts run from one through ceil(sqrt(n_qubits)), and a layer needs at least one step.
    demand = []
    for layer in problem.layers():
        magic_pairs = sum(problem.pairs[pair_id].is_magic for pair_id in layer)
        demand.append((magic_pairs, len(layer) - magic_pairs))

    estimates = []
    for rows in range(1, math.ceil(math.sqrt(problem.n_qubits)) + 1):
        cols = -(-problem.n_qubits // rows)
        mapping = Mapping.paired(rows, cols, problem.n_qubits)
        steps = sum(
            max(1, -(-magic_pairs // (2 * rows)), math.ceil(cx_pairs / (cap * rows)))
            for magic_pairs, cx_pairs in demand
        ) or 1
        estimates.append(ShapeEstimate((rows, cols), mapping.height * mapping.width, steps))
    return estimates


def stats_shape(problem: Problem, cap: float = LAYOUT_CX_PER_ROW) -> Shape:
    # The smallest predicted volume, fewer site rows on ties.
    return min(shape_estimates(problem, cap), key=lambda estimate: (estimate.volume, estimate.shape[0])).shape
