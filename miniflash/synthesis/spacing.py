# Spacing: how much of the array a program keeps, and when. A row is load-bearing when it holds a qubit tile, a
# magic tile some route consumes or a segment endpoint, and columns likewise, so removing every other row and
# column only shortens the segments that cross it and keeps each route straight, chained, disjoint and adjacent
# to its parties. Mode "none" keeps the whole array, "whole" compacts the schedule once, and "partition" cuts the
# routed steps into partitions, each closing when the next step would push its load-bearing rows or columns past
# the budget, and compacts each on its own. The chip is the largest partition height by the largest width with
# every partition anchored at the top-left, and between two partitions each qubit walks one tile per axis per
# step, which never collides because both mappings realise the same site order.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..circuit import Problem
from ..constant import SPACING_EXTRA_LANES
from ..mapping import Coord, Layout, Mapping
from ..program import QUBIT, Program, Step
from ..route import Route, Schedule, Segment


def space(problem: Problem, mapping: Mapping, schedule: Schedule, mode: str = "none",
          windows: Optional[list[tuple[int, int]]] = None, budgets: Optional[tuple[int, int]] = None) -> tuple[Program, list]:
    # Windows come from the router when given, otherwise cut closes them by the row and column budgets.
    if mode == "none":
        return Program.from_routes(problem, mapping, schedule), []
    if mode == "whole":
        mapping, schedule, keep_rows, keep_cols = compact(mapping, schedule)
        return Program.from_routes(problem, mapping, schedule), [keep_rows, keep_cols]
    if windows is None:
        windows = cut(mapping, schedule, *budgets)
    return stage(problem, mapping, schedule, windows)


def load_bearing(mapping: Mapping, schedule: Schedule) -> tuple[set[int], set[int]]:
    rows: set[int] = set()
    cols: set[int] = set()

    def keep(tile: Coord) -> None:
        rows.add(tile[0])
        cols.add(tile[1])

    for tile in mapping.qubits.values():
        keep(tile)
    for step in schedule:
        for route in step:
            if route.magic is not None:
                keep(route.magic)
            for segment in route.segments:
                keep(segment.start)
                keep(segment.end)
    return rows, cols


def compact(mapping: Mapping, schedule: Schedule) -> tuple[Mapping, Schedule, list[int], list[int]]:
    # The compacted mapping and schedule, and the original rows and columns they keep.
    rows, cols = load_bearing(mapping, schedule)
    keep_rows = sorted(rows)
    keep_cols = sorted(cols)
    row_index = {row: i for i, row in enumerate(keep_rows)}
    col_index = {col: i for i, col in enumerate(keep_cols)}

    def remap(tile: Coord) -> Coord:
        return row_index[tile[0]], col_index[tile[1]]

    consumed = {route.magic for step in schedule for route in step if route.magic is not None}
    new_mapping = Mapping(
        len(keep_rows),
        len(keep_cols),
        {qubit: remap(tile) for qubit, tile in mapping.qubits.items()},
        tuple(remap(tile) for tile in mapping.magic if tile in consumed),
    )

    new_schedule: Schedule = []
    for step in schedule:
        routes = []
        for route in step:
            segments = tuple(Segment(remap(segment.start), remap(segment.end)) for segment in route.segments)
            magic = None if route.magic is None else remap(route.magic)
            routes.append(Route(route.pair, segments, magic))
        new_schedule.append(routes)
    return new_mapping, new_schedule, keep_rows, keep_cols


@dataclass(frozen=True)
class Partition:
    start: int                    # first routed step, inclusive
    stop: int                     # last routed step, exclusive
    mapping: Mapping              # compacted and anchored at the top-left
    schedule: Schedule            # the partition's routes on the compacted mapping
    keep_rows: tuple[int, ...]    # original rows kept
    keep_cols: tuple[int, ...]


def cut(mapping: Mapping, schedule: Schedule, row_budget: int, col_budget: int) -> list[tuple[int, int]]:
    # Greedy step windows [start, stop) whose load-bearing rows and columns stay within budget.
    windows = []
    start = 0
    rows, cols = set(), set()
    for step, routes in enumerate(schedule):
        step_rows, step_cols = load_bearing(mapping, [routes])
        if step > start and (len(rows | step_rows) > row_budget or len(cols | step_cols) > col_budget):
            windows.append((start, step))
            start, rows, cols = step, set(), set()
        rows |= step_rows
        cols |= step_cols
    if start < len(schedule):
        windows.append((start, len(schedule)))
    return windows


def walk_steps(before: Mapping, after: Mapping) -> list[Step]:
    # Collision freedom relies on both mappings preserving the site order, so arbitrary permutations are not supported.
    displacement = {
        qubit: (after.qubits[qubit][0] - row, after.qubits[qubit][1] - col)
        for qubit, (row, col) in before.qubits.items()
    }
    duration = max((max(abs(d_row), abs(d_col)) for d_row, d_col in displacement.values()), default=0)

    steps = []
    for elapsed in range(1, duration + 1):
        step = Step()
        for qubit, (row, col) in before.qubits.items():
            d_row, d_col = displacement[qubit]
            tile = (
                row + max(-elapsed, min(elapsed, d_row)),
                col + max(-elapsed, min(elapsed, d_col)),
            )
            step.add(tile, QUBIT, qubit)
        steps.append(step)
    return steps


def budgets(layout: Layout, extra: int = SPACING_EXTRA_LANES, row_budget: Optional[int] = None,
            col_budget: Optional[int] = None) -> tuple[int, int]:
    # Load-bearing rows and columns a partition may retain, the site lines included, unless an explicit budget
    # overrides that axis.
    return (row_budget if row_budget is not None else layout.site_rows + extra,
            col_budget if col_budget is not None else layout.site_cols + extra)


def stage(problem: Problem, mapping: Mapping, schedule: Schedule,
          windows: list[tuple[int, int]]) -> tuple[Program, list[Partition]]:
    partitions = []
    for start, stop in windows:
        compacted, routes, keep_rows, keep_cols = compact(mapping, schedule[start:stop])
        partitions.append(Partition(start, stop, compacted, routes, tuple(keep_rows), tuple(keep_cols)))

    height = max(partition.mapping.height for partition in partitions)
    width = max(partition.mapping.width for partition in partitions)
    program = Program(height, width)
    previous: Optional[Mapping] = None
    for partition in partitions:
        chip = Mapping(height, width, partition.mapping.qubits, partition.mapping.magic)
        if previous is not None:
            program.steps.extend(walk_steps(previous, chip))
        program.steps.extend(Program.from_routes(problem, chip, partition.schedule).steps)
        previous = chip
    return program, partitions
