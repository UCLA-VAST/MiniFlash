# Routing: the criticality-driven router with lane budgets. Pairs are ordered by the longest remaining dependency
# chain and then input index, a small frontier is selected by branch and bound over the route candidates to
# maximise criticality placed minus the cost of newly loaded lines, and a larger one greedily by route length plus
# line cost. A partition accumulates the rows and columns its routes load beyond the site lines and closes when no
# ready pair fits within the budget, after which a fresh partition may take one unconstrained greedy step, so the
# budget does not bound the chip. Magic supply is modelled per tile by a cooldown after each consumption and
# globally by F factories each producing one state per period from routed step 0, counting routed steps only.
# When every ready pair is a T waiting for a tile or a state, the router emits an idle step.
from __future__ import annotations

from typing import Optional

from .. import constant
from ..circuit import Problem
from ..mapping import Coord, Mapping
from ..route import Route, RouteError, Schedule, free_routes


def lines_of(route: Route) -> tuple[set[int], set[int]]:
    # The rows and columns a route makes load-bearing, its segment endpoints and magic tile.
    rows, cols = set(), set()
    for segment in route.segments:
        rows.update((segment.start[0], segment.end[0]))
        cols.update((segment.start[1], segment.end[1]))
    if route.magic is not None:
        rows.add(route.magic[0])
        cols.add(route.magic[1])
    return rows, cols


def candidates(problem: Problem, mapping: Mapping, pair_id: int, occupied: set[Coord],
               consumed: set[Coord]) -> list[Route]:
    # Free routes from the Z side of the Z party to the X side of the X party, or to any side of one of the
    # nearest reachable magic tiles.
    pair = problem.pairs[pair_id]
    starts = mapping.z_side(pair.z)
    if not pair.is_magic:
        return free_routes(pair_id, starts, mapping.x_side(pair.x), mapping, blocked=occupied)

    z_tile = mapping.qubits[pair.z]
    found = []
    reachable = 0
    for magic in sorted((magic for magic in mapping.magic if magic not in consumed),
                        key=lambda magic: (abs(magic[0] - z_tile[0]) + abs(magic[1] - z_tile[1]), magic)):
        routes = free_routes(pair_id, starts, mapping.magic_side(magic), mapping, magic=magic, blocked=occupied)
        if routes:
            found.extend(routes)
            reachable += 1
            if reachable == constant.ROUTING_NEAREST_MAGIC:
                break
    return found


def route(problem: Problem, mapping: Mapping, row_budget: int, col_budget: int,
          share: float = constant.ROUTING_SHARE_PENALTY, exact: int = constant.ROUTING_EXACT_FRONTIER,
          magic_cooldown: int = 1, factories: int = 0,
          period: int = constant.FACTORY_PERIOD) -> tuple[Schedule, list[tuple[int, int]]]:
    # The schedule and the partition windows it closed, under the row and column budgets.
    site_rows = {tile[0] for tile in mapping.qubits.values()}
    site_cols = {tile[1] for tile in mapping.qubits.values()}
    succs = problem.succs()
    criticality = problem.criticality()
    remaining = [len(preds) for preds in problem.preds]
    frontier = order(criticality, [pair_id for pair_id, count in enumerate(remaining) if count == 0])

    schedule: Schedule = []
    windows: list[tuple[int, int]] = []
    start = 0
    part_rows: set[int] = set()
    part_cols: set[int] = set()
    busy_until: dict[Coord, int] = {}   # first step each magic tile may be consumed again
    states = 0                          # magic states consumed so far
    while frontier:
        step = len(schedule)
        fresh = step == start
        busy = {magic for magic, until in busy_until.items() if until > step}
        stock = (step * factories) // period - states if factories else None   # states produced but not yet consumed
        placed, waiting, step_rows, step_cols = select(frontier, problem, mapping, site_rows, site_cols, part_rows,
                                                       part_cols, row_budget, col_budget, True, busy, share, exact,
                                                       criticality, stock)
        if not placed and (busy or stock == 0) and all(problem.pairs[pair_id].is_magic for pair_id in frontier):
            schedule.append([])   # every ready pair waits for a magic tile or state, so the partition idles
            continue
        if not placed and not fresh:
            windows.append((start, step))
            start, part_rows, part_cols = step, set(), set()
            continue
        if not placed:
            placed, waiting, step_rows, step_cols = select(frontier, problem, mapping, site_rows, site_cols, part_rows,
                                                           part_cols, row_budget, col_budget, False, busy, share, exact,
                                                           criticality, stock)
        if not placed:
            raise RouteError(f"step {step}: no pair in the frontier {frontier} can be routed")

        part_rows |= step_rows
        part_cols |= step_cols
        for routed in placed:
            if routed.magic is not None:
                busy_until[routed.magic] = step + magic_cooldown
                states += 1
        schedule.append(placed)
        for routed in placed:
            for successor in succs[routed.pair]:
                remaining[successor] -= 1
                if remaining[successor] == 0:
                    waiting.append(successor)
        frontier = order(criticality, waiting)

    windows.append((start, len(schedule)))
    if sum(len(routes) for routes in schedule) != len(problem.pairs):
        raise RouteError("not every pair was scheduled")
    return schedule, windows


def order(criticality: list[int], pairs: list[int]) -> list[int]:
    # Criticality first, program order on ties.
    return sorted(pairs, key=lambda pair_id: (-criticality[pair_id], pair_id))


def select(frontier: list[int], problem: Problem, mapping: Mapping, site_rows: set[int], site_cols: set[int],
           part_rows: set[int], part_cols: set[int], row_budget: int, col_budget: int, strict: bool, busy: set[Coord],
           share: float, exact: int, criticality: list[int],
           stock: Optional[int] = None) -> tuple[list[Route], list[int], set[int], set[int]]:
    # The placed routes, the waiting pairs and the rows and columns loaded this step, by exact selection on a
    # small frontier and greedily otherwise.
    if strict and len(frontier) <= exact:
        return exact_select(frontier, problem, mapping, site_rows, site_cols, part_rows, part_cols, row_budget, col_budget,
                            busy, share, criticality, stock)
    return greedy_select(frontier, problem, mapping, site_rows, site_cols, part_rows, part_cols, row_budget, col_budget,
                         strict, busy, share, criticality, stock)


def exact_select(frontier: list[int], problem: Problem, mapping: Mapping, site_rows: set[int], site_cols: set[int],
                 part_rows: set[int], part_cols: set[int], row_budget: int, col_budget: int, busy: set[Coord],
                 share: float, criticality: list[int],
                 stock: Optional[int] = None) -> tuple[list[Route], list[int], set[int], set[int]]:
    # Branch and bound over the frontier in order, trying each pair's candidates shortest first and then skipping
    # it, with the criticality of every undecided pair as the bound.
    options = {
        pair_id: [(candidate, frozenset(candidate.tiles()), *lines_of(candidate))
                  for candidate in sorted(candidates(problem, mapping, pair_id, set(), set(busy)),
                                          key=lambda route: route.length)]
        for pair_id in frontier
    }
    total_criticality = sum(criticality[pair_id] for pair_id in frontier)
    best_score: Optional[float] = None
    best_chosen: dict[int, Route] = {}
    occupied: set[Coord] = set()
    consumed: set[Coord] = set(busy)
    loaded_rows: set[int] = set(site_rows | part_rows)
    loaded_cols: set[int] = set(site_cols | part_cols)
    chosen: dict[int, Route] = {}
    taken = [0]   # magic states chosen on the current branch

    def search(index: int, score: float) -> None:
        nonlocal best_score, best_chosen
        remaining_reward = sum(criticality[pair_id] for pair_id in frontier[index:])
        if best_score is not None and score + remaining_reward <= best_score:
            return
        if index == len(frontier):
            if best_score is None or score > best_score:
                best_score, best_chosen = score, dict(chosen)
            return

        pair_id = frontier[index]
        for candidate, tiles, rows, cols in options[pair_id]:
            if candidate.magic is not None and (candidate.magic in consumed or (stock is not None and taken[0] >= stock)):
                continue
            if not occupied.isdisjoint(tiles):
                continue
            new_rows, new_cols = rows - loaded_rows, cols - loaded_cols
            if len(loaded_rows | new_rows) > row_budget or len(loaded_cols | new_cols) > col_budget:
                continue
            occupied.update(tiles)
            if candidate.magic is not None:
                consumed.add(candidate.magic)
                taken[0] += 1
            loaded_rows.update(new_rows)
            loaded_cols.update(new_cols)
            chosen[pair_id] = candidate
            lane_cost = share * (len(new_rows) + len(new_cols))
            search(index + 1, score + criticality[pair_id] - lane_cost)

            del chosen[pair_id]
            loaded_rows.difference_update(new_rows)
            loaded_cols.difference_update(new_cols)
            if candidate.magic is not None:
                consumed.discard(candidate.magic)
                taken[0] -= 1
            occupied.difference_update(tiles)
            if best_score == total_criticality:
                return
        search(index + 1, score)   # defer this pair to a later step

    search(0, 0.0)
    placed = [best_chosen[pair_id] for pair_id in frontier if pair_id in best_chosen]
    waiting = [pair_id for pair_id in frontier if pair_id not in best_chosen]
    step_rows: set[int] = set()
    step_cols: set[int] = set()
    for routed in placed:
        rows, cols = lines_of(routed)
        step_rows |= rows
        step_cols |= cols
    return placed, waiting, step_rows, step_cols


def greedy_select(frontier: list[int], problem: Problem, mapping: Mapping, site_rows: set[int], site_cols: set[int],
                  part_rows: set[int], part_cols: set[int], row_budget: int, col_budget: int, strict: bool,
                  busy: set[Coord], share: float, criticality: list[int],
                  stock: Optional[int] = None) -> tuple[list[Route], list[int], set[int], set[int]]:
    occupied: set[Coord] = set()
    consumed: set[Coord] = set(busy)
    step_rows: set[int] = set()
    step_cols: set[int] = set()
    placed: list[Route] = []
    waiting: list[int] = []
    taken = 0
    for pair_id in frontier:
        best, best_score = None, None
        if problem.pairs[pair_id].is_magic and stock is not None and taken >= stock:
            waiting.append(pair_id)
            continue
        for candidate in candidates(problem, mapping, pair_id, occupied, consumed):
            rows, cols = lines_of(candidate)
            new_rows = rows - site_rows - step_rows - part_rows
            new_cols = cols - site_cols - step_cols - part_cols
            if strict and (len(site_rows | part_rows | step_rows | new_rows) > row_budget
                           or len(site_cols | part_cols | step_cols | new_cols) > col_budget):
                continue
            score = candidate.length + share * (len(new_rows) + len(new_cols))
            if best is None or score < best_score:
                best, best_score = candidate, score
        if best is None:
            waiting.append(pair_id)
            continue

        placed.append(best)
        occupied.update(best.tiles())
        if best.magic is not None:
            consumed.add(best.magic)
            taken += 1
        rows, cols = lines_of(best)
        step_rows |= rows
        step_cols |= cols
    return placed, waiting, step_rows, step_cols
