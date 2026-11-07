# Scheduling: which pairs execute in each step and along which routes. A frontier holds the pairs whose
# predecessors have all finished, seeded with the DAG's sources. Each step walks the frontier in pair order,
# places a pair whose shortest free route exists and occupies that route's tiles and magic tile for the rest of
# the step, and a pair that finds no free route stays in the frontier. When the step ends the successors whose
# other predecessors are also done join the frontier and the occupancy clears. Without dependency every step
# executes one pair in program order along the shortest legal route.
from __future__ import annotations

from ..circuit import Problem
from ..mapping import Coord, Mapping
from ..route import Route, RouteError, Schedule, check_route, shortest_magic_route, shortest_route


def schedule(problem: Problem, mapping: Mapping, dependency: bool = True) -> Schedule:
    if not dependency:
        steps = []
        for pair_id, pair in enumerate(problem.pairs):
            if pair.is_magic:
                route = shortest_magic_route(pair_id, pair.z, mapping)
            else:
                route = shortest_route(pair_id, pair.z, pair.x, mapping)
            if route is None:
                raise RouteError(f"pair {pair_id} {pair}: no candidate route on the mapping")
            check_route(route, pair, mapping)
            steps.append([route])
        return steps

    succs = problem.succs()
    remaining = [len(preds) for preds in problem.preds]   # unfinished predecessors per pair
    frontier = [pair_id for pair_id, count in enumerate(remaining) if count == 0]
    done = 0
    steps: Schedule = []
    while frontier:
        occupied: set[Coord] = set()
        consumed: set[Coord] = set()
        placed: list[Route] = []
        waiting: list[int] = []
        for pair_id in frontier:
            pair = problem.pairs[pair_id]
            if pair.is_magic:
                route = shortest_magic_route(pair_id, pair.z, mapping, blocked=occupied, consumed=consumed)
            else:
                route = shortest_route(pair_id, pair.z, pair.x, mapping, blocked=occupied)
            if route is None:
                waiting.append(pair_id)
                continue
            placed.append(route)
            occupied.update(route.tiles())
            if route.magic is not None:
                consumed.add(route.magic)
        if not placed:
            raise RouteError(f"step {len(steps)}: no pair in the frontier {frontier} can be routed")

        steps.append(placed)
        done += len(placed)
        for route in placed:
            for successor in succs[route.pair]:
                remaining[successor] -= 1
                if remaining[successor] == 0:
                    waiting.append(successor)
        frontier = sorted(waiting)

    if done != len(problem.pairs):
        raise RouteError(f"{done} of {len(problem.pairs)} pairs scheduled")
    return steps
