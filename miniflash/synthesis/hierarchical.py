# Compile a call tree bottom-up on one virtual mapping. Leaves intern exact pair sequences on the same global
# qubits and parents intern their ordered children, and every program qubit stays an obstacle when a cell is
# routed. The composer preserves each cell's step order and grouping, releases qubits after their last use and
# enforces the flat pair DAG and the magic supply globally. Compaction happens only after composition.
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

from ..circuit import Problem
from ..constant import FACTORY_PERIOD
from ..hierarchy import Hierarchy, Node
from ..mapping import Coord, Mapping
from ..route import Route, RouteError, Schedule, segments_of
from .placement import Workload
from .routing import candidates


@dataclass(frozen=True)
class Call:
    cell: int
    start: int
    stop: int


@dataclass(frozen=True)
class Cell:
    problem: Problem
    calls: tuple[Call, ...] = ()


# Interned cells in postorder, with call ranges local to their parent and frequencies counting expanded
# invocations at every level.
@dataclass(frozen=True)
class Plan:
    cells: tuple[Cell, ...]
    root: Optional[int]
    frequencies: tuple[int, ...]
    bindings: dict[tuple[str, tuple[int, ...]], int]

    def workload(self, n_qubits: int) -> Workload:
        # Only leaves contribute layers, each weighted by its expanded invocation count.
        pairs = []
        layers = []
        frequencies = []
        for cell, count in zip(self.cells, self.frequencies):
            if cell.calls:
                continue
            offset = len(pairs)
            layers.extend([[offset + pair_id for pair_id in layer] for layer in cell.problem.layers()])
            frequencies.extend([count] * cell.problem.depth)
            pairs.extend(cell.problem.pairs)
        return Workload(Problem.from_pairs(n_qubits, pairs), layers, frequencies)

    def stats(self) -> dict:
        leaves = [cell_id for cell_id, cell in enumerate(self.cells) if not cell.calls]
        return dict(cells=len(self.cells), calls=sum(self.frequencies), leaf_cells=len(leaves),
                    composed_cells=len(self.cells) - len(leaves),
                    leaf_calls=sum(self.frequencies[leaf] for leaf in leaves),
                    routed_pairs=sum(len(self.cells[leaf].problem.pairs) for leaf in leaves),
                    total_pairs=0 if self.root is None else len(self.cells[self.root].problem.pairs))


def extract(hierarchy: Hierarchy) -> Plan:
    # Single-qubit Cliffords are free and stay out of the leaf key, T and T-dagger stay distinct, and a repeated
    # bound definition reuses its cell instead of walking its subtree again.
    problem = hierarchy.problem
    cells = []
    ids = {}
    bindings = {}

    def intern(start: int, stop: int, calls: Sequence[Call] = ()) -> int:
        pairs = problem.pairs[start:stop]
        key = ("parent", tuple(call.cell for call in calls)) if calls else ("leaf", pairs)
        if key not in ids:
            ids[key] = len(cells)
            cells.append(Cell(Problem.from_pairs(problem.n_qubits, pairs), tuple(calls)))
        return ids[key]

    def visit(node: Node) -> int:
        binding = (node.name, node.qubits)
        if binding in bindings:
            return bindings[binding]

        children = [child for child in node.children if child.stop > child.start]
        calls = []
        position = node.start
        for child in children:
            if position < child.start:
                calls.append(Call(intern(position, child.start), position - node.start, child.start - node.start))
            calls.append(Call(visit(child), child.start - node.start, child.stop - node.start))
            position = child.stop
        if children and position < node.stop:
            calls.append(Call(intern(position, node.stop), position - node.start, node.stop - node.start))

        cell = intern(node.start, node.stop, calls)
        bindings[binding] = cell
        return cell

    root = visit(hierarchy.tree) if problem.pairs else None
    frequencies = [0] * len(cells)
    if root is not None:
        frequencies[root] = 1
    for cell_id in reversed(range(len(cells))):
        for call in cells[cell_id].calls:
            frequencies[call.cell] += frequencies[cell_id]
    return Plan(tuple(cells), root, tuple(frequencies), bindings)


def _shortest(problem: Problem, mapping: Mapping, pair_id: int, occupied: set[Coord],
              consumed: set[Coord]) -> Optional[Route]:
    # Breadth-first fallback with unrestricted bends, after the candidate search.
    pair = problem.pairs[pair_id]
    targets = {}
    if pair.is_magic:
        for magic in sorted(set(mapping.magic) - consumed):
            for tile in mapping.magic_side(magic):
                targets.setdefault(tile, magic)
    else:
        targets = dict.fromkeys(mapping.x_side(pair.x))

    previous = {tile: None for tile in mapping.z_side(pair.z) if mapping.is_free(tile) and tile not in occupied}
    queue = deque(previous)
    while queue:
        tile = queue.popleft()
        if tile in targets:
            path = [tile]
            while previous[path[-1]] is not None:
                path.append(previous[path[-1]])
            return Route(pair_id, segments_of(path[::-1]), targets[tile])
        for neighbour in mapping.vertical(tile) + mapping.horizontal(tile):
            if neighbour not in previous and neighbour not in occupied and mapping.is_free(neighbour):
                previous[neighbour] = tile
                queue.append(neighbour)
    return None


def _fit(step: list[Route], problem: Problem, mapping: Mapping, occupied: set[Coord],
         consumed: set[Coord]) -> Optional[list[Route]]:
    # A whole cell step is tried transactionally, so a failed attempt reserves nothing.
    occupied, consumed = set(occupied), set(consumed)
    placed = []
    for cached in step:
        route = cached
        if not occupied.isdisjoint(route.tiles()) or route.magic in consumed:
            choices = candidates(problem, mapping, route.pair, occupied, consumed)
            if choices:
                route = min(choices, key=lambda choice: choice.length)
            else:
                route = _shortest(problem, mapping, route.pair, occupied, consumed)
            if route is None:
                return None
        placed.append(route)
        occupied.update(route.tiles())
        if route.magic is not None:
            consumed.add(route.magic)
    return placed


def compose(problem: Problem, mapping: Mapping, calls: Sequence[Call], templates: list[Schedule],
            magic_cooldown: int = 1, factories: int = 0, period: int = FACTORY_PERIOD) -> Schedule:
    # The step DAG adds template-order edges to the flat pair DAG, and a ready step waits as a unit when its
    # routes or magic states do not all fit.
    groups = []
    predecessors = []
    owner = [-1] * len(problem.pairs)
    for call in calls:
        previous = None
        for step in templates[call.cell]:
            if not step:
                continue   # templates are routed with unlimited supply
            group = len(groups)
            groups.append([Route(call.start + cached.pair, cached.segments, cached.magic) for cached in step])
            predecessors.append(set() if previous is None else {previous})
            for route in groups[-1]:
                owner[route.pair] = group
            previous = group
    if any(group < 0 for group in owner):
        raise RouteError("cell templates do not cover every pair")

    for pair_id, preds in enumerate(problem.preds):
        predecessors[owner[pair_id]].update(owner[pred] for pred in preds)
    successors = [[] for _ in groups]
    for group, preds in enumerate(predecessors):
        for pred in preds:
            successors[pred].append(group)
    remaining = [len(preds) for preds in predecessors]
    ready = [group for group, count in enumerate(remaining) if not count]

    schedule: Schedule = []
    busy_until = {}
    states = 0
    done = 0
    while ready:
        step = len(schedule)
        consumed = {magic for magic, until in busy_until.items() if until > step}
        occupied = set()
        placed = []
        waiting = []
        finished = []
        stock = (step * factories) // period - states if factories else None
        for group in ready:
            demand = sum(route.magic is not None for route in groups[group])
            if stock is not None and demand > stock:
                routes = None
            else:
                routes = _fit(groups[group], problem, mapping, occupied, consumed)
            if routes is None:
                waiting.append(group)
                continue
            placed.extend(routes)
            finished.append(group)
            for route in routes:
                occupied.update(route.tiles())
                if route.magic is not None:
                    consumed.add(route.magic)
                    busy_until[route.magic] = step + magic_cooldown
                    states += 1
            if stock is not None:
                stock -= demand
        if not placed and not factories and not any(until > step for until in busy_until.values()):
            raise RouteError("no ready cell step can be routed on the common mapping")

        schedule.append(placed)
        done += len(finished)
        for group in finished:
            for successor in successors[group]:
                remaining[successor] -= 1
                if not remaining[successor]:
                    waiting.append(successor)
        ready = sorted(waiting)

    if done != len(groups):
        raise RouteError("cell step order conflicts with pair dependencies")
    return schedule
