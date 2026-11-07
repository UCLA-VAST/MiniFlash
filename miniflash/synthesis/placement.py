# Placement: where each qubit sits on the array. Index order on the sparse square is the reference point;
# otherwise a seed from the interaction graph, spectral by default or one of initial.ORDERS, is annealed by
# swapping qubits between sites, empty ones included, to lower a routing proxy summed over the ASAP layers of
# the DAG. The proxy sums the weighted collisions the layer scan detects, the rows and columns loaded beyond
# the site lines, and the tentative route tiles. A tentative route runs from the lane-row tile on the Z party's
# Z side along that row to the lane column beside the X party as an L, or along the row to the nearest row-end
# magic tile, ignoring other routes, and only the layers touching the two swapped qubits are re-scored. The
# array is the paired-lane one, and the later stages read the placed order through the returned Layout.
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import constant
from ..circuit import Problem
from ..mapping import Coord, Layout, Mapping
from . import initial


# Distinct placement layers with their execution counts, so repeated cells weight the layer cost without expansion.
@dataclass(frozen=True)
class Workload:
    problem: Problem
    layers: list[list[int]]
    frequencies: list[int]

    def interaction(self) -> dict[tuple[int, int], int]:
        edges: dict[tuple[int, int], int] = {}
        for layer, count in zip(self.layers, self.frequencies):
            for pair_id in layer:
                pair = self.problem.pairs[pair_id]
                if not pair.is_magic:
                    edge = tuple(sorted(pair.qubits))
                    edges[edge] = edges.get(edge, 0) + count
        return edges


INITS = ("spectral", "random", "gordian", "gordian-l", "mincut", "aplace", "mpl")


def interaction_matrix(problem: Problem, workload: Optional[Workload] = None) -> np.ndarray:
    # CX counts plus a floor between every pair, so the graph every initial placement reads is connected.
    n_qubits = problem.n_qubits
    adjacency = np.full((n_qubits, n_qubits), constant.INTERACTION_FLOOR)
    np.fill_diagonal(adjacency, 0.0)
    edges = workload.interaction() if workload is not None else problem.interaction()
    for (qubit_a, qubit_b), weight in edges.items():
        adjacency[qubit_a, qubit_b] += weight
        adjacency[qubit_b, qubit_a] += weight
    return adjacency


def initial_order(init: str, problem: Problem, site_rows: int, site_cols: int,
                  workload: Optional[Workload] = None, seed: int = constant.PLACEMENT_SEED) -> dict[int, tuple[int, int]]:
    if init == "spectral":
        return spectral_order(problem, site_rows, site_cols, workload)
    return initial.ORDERS[init](interaction_matrix(problem, workload), site_rows, site_cols, seed)


def spectral_order(problem: Problem, site_rows: int, site_cols: int,
                   workload: Optional[Workload] = None) -> dict[int, tuple[int, int]]:
    # The Fiedler vector of the Laplacian ranks rows and the next mode orders each row.
    n_qubits = problem.n_qubits
    adjacency = interaction_matrix(problem, workload)
    laplacian = np.diag(adjacency.sum(1)) - adjacency
    _vals, vecs = np.linalg.eigh(laplacian)
    row_scores = vecs[:, 1] if n_qubits > 1 else np.zeros(n_qubits)
    col_scores = vecs[:, 2] if n_qubits > 2 else np.zeros(n_qubits)

    by_row = sorted(range(n_qubits), key=lambda qubit: (row_scores[qubit], qubit))
    order = {}
    for row in range(site_rows):
        chunk = sorted(by_row[row * site_cols:(row + 1) * site_cols], key=lambda qubit: (col_scores[qubit], qubit))
        for col, qubit in enumerate(chunk):
            order[qubit] = (row, col)
    return order


# The layered routing proxy, re-scored per move on the touched layers only.
class _Proxy:
    def __init__(self, problem: Problem, mapping: Mapping,
                 weights: tuple[float, float, float] = (constant.ANNEAL_COLLISION_WEIGHT, constant.ANNEAL_LANE_WEIGHT,
                                                        constant.ANNEAL_LENGTH_WEIGHT),
                 workload: Optional[Workload] = None):
        self.problem = workload.problem if workload is not None else problem
        self.mapping = mapping
        self.collision_weight, self.lane_weight, self.length_weight = weights
        self.tiles = dict(mapping.qubits)
        self.site_rows = {tile[0] for tile in mapping.qubits.values()}
        self.site_cols = {tile[1] for tile in mapping.qubits.values()}
        self.magic = sorted(mapping.magic)
        self.side_magic = [tile for tile in self.magic if tile[1] in (0, mapping.width - 1)]
        self.lane_rows = {row for row in range(mapping.height) if row not in self.site_rows}
        self.lane_cols = {col for col in range(mapping.width) if col not in self.site_cols}
        self.layers = workload.layers if workload is not None else problem.layers()
        self.frequencies = workload.frequencies if workload is not None else [1] * len(self.layers)

        self.of_qubit: dict[int, list[int]] = {qubit: [] for qubit in problem.qubits}
        for layer_index, layer in enumerate(self.layers):
            seen = set()
            for pair_id in layer:
                for qubit in self.problem.pairs[pair_id].qubits:
                    if qubit not in seen:
                        self.of_qubit[qubit].append(layer_index)
                        seen.add(qubit)

        # A tentative route depends only on its parties' tiles, so it is cached on them and a proposal
        # re-derives only the moved qubits' routes.
        self._routes: dict[int, dict[tuple, tuple[list[Coord], set[int], set[int]]]] = {}
        self._magic_near: dict[Coord, Coord] = {}
        self.layer_cost = [self.score(layer_index) for layer_index in range(len(self.layers))]
        self.total = sum(self.layer_cost)

    def side_tile(self, qubit: int, z_side: bool) -> Coord:
        # The lane-row tile on the Z side, or the lane-column tile on the X side.
        row, col = self.tiles[qubit]
        if z_side:
            return (row - 1, col) if row - 1 in self.lane_rows else (row + 1, col)
        return (row, col - 1) if col - 1 in self.lane_cols else (row, col + 1)

    @staticmethod
    def _chain(points: list[Coord]) -> list[Coord]:
        tiles = [points[0]]
        for start, end in zip(points, points[1:]):
            if start[0] == end[0]:
                sign = 1 if end[1] >= start[1] else -1
                tiles.extend((start[0], col) for col in range(start[1] + sign, end[1] + sign, sign))
            else:
                sign = 1 if end[0] >= start[0] else -1
                tiles.extend((row, start[1]) for row in range(start[0] + sign, end[0] + sign, sign))
        return tiles

    def route_tiles(self, pair_id: int) -> tuple[list[Coord], set[int], set[int]]:
        pair = self.problem.pairs[pair_id]
        if pair.is_magic:
            key = (self.tiles[pair.z],)
        else:
            key = (self.tiles[pair.z], self.tiles[pair.x])

        recent = self._routes.get(pair_id)
        if recent is None:
            recent = self._routes[pair_id] = {}
        result = recent.get(key)
        if result is None:
            if len(recent) >= constant.ROUTE_CACHE_PER_PAIR:   # a rejected proposal restores an earlier state
                recent.pop(next(iter(recent)))
            result = recent[key] = self._route_tiles(pair_id)
        return result

    def _route_tiles(self, pair_id: int) -> tuple[list[Coord], set[int], set[int]]:
        pair = self.problem.pairs[pair_id]
        start = self.side_tile(pair.z, z_side=True)   # on a lane row
        magic = None
        if pair.is_magic:
            qubit_row, qubit_col = self.tiles[pair.z]
            magic = self._magic_near.get((qubit_row, qubit_col))
            if magic is None:
                pool = self.side_magic or self.magic
                magic = self._magic_near[(qubit_row, qubit_col)] = min(
                    pool, key=lambda tile: (abs(tile[0] - qubit_row) + abs(tile[1] - qubit_col), tile))
            points = [start, (start[0], magic[1])]   # along the lane row to the tile beside the magic tile
        else:
            end = self.side_tile(pair.x, z_side=False)   # on a lane column
            points = [start, (start[0], end[1]), end]   # row then column, the L

        tiles = self._chain(points)
        rows = {point[0] for point in points}
        cols = {point[1] for point in points}
        if magic is not None:
            rows.add(magic[0])
            cols.add(magic[1])
        return tiles, rows, cols

    def score(self, layer_index: int) -> float:
        # Each tile remembers its latest route and a route counts every earlier owner it meets once, a proxy
        # rather than an all-pairs conflict count.
        seen: dict[Coord, int] = {}
        collisions = 0
        rows: set[int] = set()
        cols: set[int] = set()
        length = 0
        for pair_id in self.layers[layer_index]:
            tiles, route_rows, route_cols = self.route_tiles(pair_id)
            length += len(tiles)
            rows |= route_rows
            cols |= route_cols
            earlier = set()
            for tile in tiles:
                if tile in seen:
                    earlier.add(seen[tile])
                seen[tile] = pair_id
            collisions += len(earlier)

        lanes = len(rows - self.site_rows) + len(cols - self.site_cols)
        return self.frequencies[layer_index] * (self.collision_weight * collisions + self.lane_weight * lanes + self.length_weight * length)

    def swap_delta(self, qubit_a: int, qubit_b: Optional[int],
                   site_b: Coord) -> tuple[float, list[int], dict[int, float]]:
        # The cost change if qubit_a moves to site_b and qubit_b, when there is one, to qubit_a's site.
        touched = sorted(set(self.of_qubit[qubit_a]) | (set(self.of_qubit[qubit_b]) if qubit_b is not None else set()))
        site_a = self.tiles[qubit_a]
        self.tiles[qubit_a] = site_b
        if qubit_b is not None:
            self.tiles[qubit_b] = site_a
        new_cost = {layer_index: self.score(layer_index) for layer_index in touched}
        self.tiles[qubit_a] = site_a
        if qubit_b is not None:
            self.tiles[qubit_b] = site_b
        return sum(new_cost[layer_index] - self.layer_cost[layer_index] for layer_index in touched), touched, new_cost

    def apply(self, qubit_a: int, qubit_b: Optional[int], site_b: Coord, new_cost: dict[int, float]) -> None:
        site_a = self.tiles[qubit_a]
        self.tiles[qubit_a] = site_b
        if qubit_b is not None:
            self.tiles[qubit_b] = site_a
        for layer_index, cost in new_cost.items():
            self.total += cost - self.layer_cost[layer_index]
            self.layer_cost[layer_index] = cost


@dataclass(frozen=True)
class Placement:
    mapping: Mapping
    layout: Layout
    initial_cost: float
    final_cost: float


def square_shape(n_qubits: int, shape: Optional[tuple[int, int]] = None) -> tuple[int, int]:
    return shape if shape is not None else (math.ceil(math.sqrt(n_qubits)),) * 2


def place(problem: Problem, shape: Optional[tuple[int, int]] = None, spectral: bool = False,
          seed: int = constant.PLACEMENT_SEED, iters: Optional[int] = None, anneal: bool = True,
          weights: tuple[float, float, float] = (constant.ANNEAL_COLLISION_WEIGHT, constant.ANNEAL_LANE_WEIGHT,
                                                 constant.ANNEAL_LENGTH_WEIGHT),
          workload: Optional[Workload] = None, init: str = constant.INITIAL_PLACEMENT) -> Placement:
    # Without spectral the qubits sit in index order on the sparse square and nothing is annealed.
    n_qubits = problem.n_qubits
    site_rows, site_cols = square_shape(n_qubits, shape)
    if not spectral:
        return Placement(Mapping.sparse(site_rows, site_cols, n_qubits),
                         Layout.identity(n_qubits, (site_rows, site_cols)), 0.0, 0.0)

    order = initial_order(init, problem, site_rows, site_cols, workload, seed)
    base = Mapping.paired(site_rows, site_cols, n_qubits)
    full = Mapping.paired(site_rows, site_cols, site_rows * site_cols)   # every site filled, to read the geometry
    row_pos = sorted({tile[0] for tile in full.qubits.values()})
    col_pos = sorted({tile[1] for tile in full.qubits.values()})
    site_of = {(row, col): (row_pos[row], col_pos[col]) for row in range(site_rows) for col in range(site_cols)}
    mapping = Mapping(base.height, base.width, {qubit: site_of[site] for qubit, site in order.items()}, base.magic)
    proxy = _Proxy(problem, mapping, weights, workload)
    initial_cost = proxy.total

    if anneal and n_qubits > 1:
        rng = random.Random(seed)
        iters = iters if iters is not None else constant.ANNEAL_PROPOSALS_PER_QUBIT * n_qubits
        sites = list(site_of.values())
        occupant = {tile: qubit for qubit, tile in proxy.tiles.items()}
        start_temp = max(1.0, constant.ANNEAL_START_TEMPERATURE * proxy.total / max(1, sum(proxy.frequencies)))
        end_temp = constant.ANNEAL_END_TEMPERATURE
        for iteration in range(iters):
            temperature = start_temp * (end_temp / start_temp) ** (iteration / max(1, iters - 1))
            qubit_a = rng.randrange(n_qubits)
            site_b = sites[rng.randrange(len(sites))]
            if site_b == proxy.tiles[qubit_a]:
                continue
            qubit_b = occupant.get(site_b)
            delta, touched, new_cost = proxy.swap_delta(qubit_a, qubit_b, site_b)
            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                site_a = proxy.tiles[qubit_a]
                proxy.apply(qubit_a, qubit_b, site_b, new_cost)
                occupant[site_b] = qubit_a
                if qubit_b is not None:
                    occupant[site_a] = qubit_b
                else:
                    del occupant[site_a]

    site_at = {tile: site for site, tile in site_of.items()}
    layout = Layout(site_rows, site_cols, {qubit: site_at[tile] for qubit, tile in proxy.tiles.items()})
    return Placement(Mapping(base.height, base.width, dict(proxy.tiles), base.magic), layout, initial_cost, proxy.total)
