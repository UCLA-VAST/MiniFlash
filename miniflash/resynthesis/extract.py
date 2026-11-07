# Window extraction. stages finds the maximal runs of steps without a walk, close builds the closed windows of one
# step range by merging components transitively on box overlap, and slabs lists every step range of every stage whose
# windows all fit the budget. extract_greedy picks non-overlapping slabs by estimated gain, and extract_dp finds the
# cheapest chain of slabs through every stage by dynamic programming over the cut positions, cost pricing a slab with
# the terms of Weights. Closure is what makes a window closed, since a component touching a window's box is merged into
# it and qubits do not move inside a stage, so every earlier or later component on a participant is inside the box too.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .. import constant
from ..circuit import Problem
from ..program import MAGIC, QUBIT, Program
from .window import Box, Budget, Component, Slab, Window


def stages(program: Program) -> list[tuple[int, int]]:
    # The (start, stop) of each maximal run of steps with edges, walk steps separating stages and belonging to none.
    out = []
    start = None
    for step_index, step in enumerate(program.steps):
        if step.is_walk:
            if start is not None:
                out.append((start, step_index))
                start = None
        elif start is None:
            start = step_index
    if start is not None:
        out.append((start, len(program.steps)))
    return out


def close(program: Program, start: int, stop: int) -> list[Window]:
    # The closed windows of steps start..stop-1, ordered by box position.
    components = [component for step_index in range(start, stop)
                  for component in Component.of_step(program.steps[step_index], step_index)]
    classes: list[tuple[Box, list[Component]]] = [(component.box, [component]) for component in components]
    merged = True
    while merged:   # a union can overlap a class already passed over, so merge to a fixpoint
        merged = False
        out: list[tuple[Box, list[Component]]] = []
        for box, members in classes:
            for index, (other, other_members) in enumerate(out):
                if box.overlaps(other):
                    out[index] = (box.union(other), other_members + members)
                    merged = True
                    break
            else:
                out.append((box, members))
        classes = out

    first = program.steps[start]
    consumed = {component.magic for component in components if component.magic is not None}
    windows = []
    for box, members in classes:
        participants = {qubit for component in members for qubit in component.qubits}
        bystanders = {vertex.id: vertex.tile for vertex in first.of_kind(QUBIT)
                      if box.contains(vertex.tile) and vertex.id not in participants}
        idle = tuple(vertex.tile for vertex in first.of_kind(MAGIC)
                     if box.contains(vertex.tile) and vertex.tile not in consumed)
        windows.append(Window(box, start, stop, tuple(members), bystanders, idle))
    return sorted(windows, key=lambda window: (window.box.r0, window.box.c0))


def slabs(program: Program, budget: Budget = Budget(), max_steps: Optional[int] = None) -> list[Slab]:
    # Longer ranges only grow windows, so each start stops at the first range whose windows do not all fit the budget.
    if max_steps is None:
        max_steps = budget.steps if budget.steps is not None else len(program.steps)
    out = []
    for stage_start, stage_stop in stages(program):
        for start in range(stage_start, stage_stop):
            for stop in range(start + 1, min(start + max_steps, stage_stop) + 1):
                windows = close(program, start, stop)
                if not all(budget.fits(window) for window in windows):
                    break
                out.append(Slab(start, stop, tuple(windows)))
    return out


def extract_greedy(program: Program, problem: Problem, budget: Budget = Budget(), max_steps: Optional[int] = None,
                   min_gain: int = 1) -> list[Slab]:
    # Non-overlapping slabs by estimated gain, longer and earlier first among equals, those below min_gain dropped.
    ranked = sorted(slabs(program, budget, max_steps),
                    key=lambda slab: (-slab.gain(problem), -slab.steps, slab.start))
    chosen: list[Slab] = []
    for slab in ranked:
        if slab.gain(problem) < min_gain:
            break
        if not any(slab.overlaps(picked) for picked in chosen):
            chosen.append(slab)
    return sorted(chosen, key=lambda slab: slab.start)


# What a chain of slabs pays, in tile-steps, priced so that a step within the budget is always cheaper inside a slab
# than skipped.
@dataclass(frozen=True)
class Weights:
    volume: float = constant.SLAB_VOLUME_WEIGHT   # per tile-step of the estimated slab, its depth times the program area
    holes: float = constant.SLAB_HOLES_WEIGHT     # per tile-step left empty above a window shallower than its slab
    cuts: float = constant.SLAB_CUTS_WEIGHT       # per dependency edge a cut severs
    skip: float = constant.SLAB_SKIP_WEIGHT       # per step left outside every slab, on top of its volume and severed edges


def cost(slab: Slab, problem: Problem, area: int, weights: Weights = Weights()) -> float:
    # The price of one slab on a program of area tiles.
    return (weights.volume * slab.depth(problem) * area + weights.holes * slab.holes(problem)
            + weights.cuts * slab.cuts(problem))


def skip_cost(program: Program, problem: Problem, step_index: int, weights: Weights = Weights()) -> float:
    # The price of leaving the step outside every slab.
    edges = sum(len(problem.preds[component.pair]) for component in Component.of_step(program.steps[step_index], step_index))
    return weights.volume * program.area + weights.cuts * edges + weights.skip


def extract_dp(program: Program, problem: Problem, budget: Budget = Budget(), max_steps: Optional[int] = None,
               weights: Weights = Weights()) -> list[Slab]:
    # best[stop] is the least cost of cutting the stage up to step stop, ending with a slab or with stop - 1 skipped.
    by_range: dict[tuple[int, int], Slab] = {(slab.start, slab.stop): slab
                                              for slab in slabs(program, budget, max_steps)}
    longest = max((slab.steps for slab in by_range.values()), default=1)
    area = program.area
    chosen: list[Slab] = []
    for stage_start, stage_stop in stages(program):
        best: dict[int, float] = {stage_start: 0.0}
        back: dict[int, Optional[Slab]] = {}
        for stop in range(stage_start + 1, stage_stop + 1):
            best[stop] = best[stop - 1] + skip_cost(program, problem, stop - 1, weights)
            back[stop] = None
            for length in range(1, min(longest, stop - stage_start) + 1):
                slab = by_range.get((stop - length, stop))
                if slab is None:
                    continue
                candidate = best[stop - length] + cost(slab, problem, area, weights)
                if candidate < best[stop]:
                    best[stop], back[stop] = candidate, slab

        stop = stage_stop
        while stop > stage_start:
            slab = back[stop]
            if slab is None:
                stop -= 1
            else:
                chosen.append(slab)
                stop = slab.start
    return sorted(chosen, key=lambda slab: slab.start)


def total(chosen: list[Slab], program: Program, problem: Problem, weights: Weights = Weights()) -> float:
    # The cost of non-overlapping slabs over the whole program, the steps of every stage they leave out charged as skips.
    covered = {step_index for slab in chosen for step_index in range(slab.start, slab.stop)}
    skipped = [step_index for stage_start, stage_stop in stages(program)
               for step_index in range(stage_start, stage_stop) if step_index not in covered]
    return (sum(cost(slab, problem, program.area, weights) for slab in chosen)
            + sum(skip_cost(program, problem, step_index, weights) for step_index in skipped))
