# Initial placement: the site assignment the anneal starts from. Every seed reads the same weighted interaction
# graph, CX counts plus a uniform floor so the graph is connected, and maps each qubit to a virtual site (i, j)
# on a rows x cols array, with continuous coordinates in site units. gordian minimises quadratic wirelength under
# centre-of-gravity constraints and bisects every region until each holds one site, gordian_l reweights each edge
# by 1/|d| for the linear objective, mincut bisects by Kernighan-Lin from random starts with outside qubits as
# terminals, aplace minimises log-sum-exp wirelength plus a bell-shaped bin density penalty by conjugate gradient,
# and mpl clusters by heavy-edge matching, places the coarsest graph and declusters level by level. aplace and mpl
# end with a minimum-cost assignment of qubits to sites.
from __future__ import annotations

import math

import numpy as np

from .. import constant

Order = dict[int, tuple[int, int]]
Region = tuple[int, int, int, int, np.ndarray]   # rows row0..row1-1, cols col0..col1-1, member qubits


# ---- shared pieces
def legalize(positions: np.ndarray, rows: int, cols: int) -> Order:
    # Distinct sites minimising the total squared displacement from positions.
    sites = np.array([(i, j) for i in range(rows) for j in range(cols)], float)
    cost = ((positions[:, None, :] - sites[None, :, :]) ** 2).sum(2)
    site_index = assign(cost)
    return {qubit: (int(sites[site][0]), int(sites[site][1])) for qubit, site in enumerate(site_index)}


def assign(cost: np.ndarray) -> np.ndarray:
    # The Hungarian method on an n x m cost matrix with n <= m gives the column of each row.
    n_rows, n_cols = cost.shape
    u, v = np.zeros(n_rows + 1), np.zeros(n_cols + 1)   # u, v are the row and column potentials
    row_of_col = np.zeros(n_cols + 1, dtype=int)   # row_of_col[j] is the 1-based row holding column j
    came_from = np.zeros(n_cols + 1, dtype=int)
    for row in range(1, n_rows + 1):
        row_of_col[0] = row
        col = 0
        min_values = np.full(n_cols + 1, np.inf)
        used = np.zeros(n_cols + 1, dtype=bool)
        while True:
            used[col] = True
            free = ~used[1:]
            reduced = cost[row_of_col[col] - 1] - u[row_of_col[col]] - v[1:]
            better = free & (reduced < min_values[1:])
            min_values[1:][better] = reduced[better]
            came_from[1:][better] = col
            candidates = np.where(free, min_values[1:], np.inf)
            next_col = int(np.argmin(candidates)) + 1
            delta = candidates[next_col - 1]
            held = np.nonzero(used)[0]
            u[row_of_col[held]] += delta
            v[held] -= delta
            min_values[1:][free] -= delta
            col = next_col
            if row_of_col[col] == 0:
                break
        while col:
            next_col = came_from[col]
            row_of_col[col] = row_of_col[next_col]
            col = next_col

    col_of_row = np.zeros(n_rows, dtype=int)
    for col in range(1, n_cols + 1):
        if row_of_col[col]:
            col_of_row[row_of_col[col] - 1] = col - 1
    return col_of_row


def random_order(weights: np.ndarray, rows: int, cols: int, seed: int = 0) -> Order:
    sites = np.random.default_rng(seed).permutation(rows * cols)[:len(weights)]
    return {qubit: (int(site) // cols, int(site) % cols) for qubit, site in enumerate(sites)}


# ---- gordian and gordian-l
def gordian(weights: np.ndarray, rows: int, cols: int, seed: int = 0, linear: bool = False,
            rounds: int = constant.GORDIAN_L_ROUNDS) -> Order:
    # Deterministic, so seed is unused.
    n_qubits = len(weights)
    regions: list[Region] = [(0, rows, 0, cols, np.arange(n_qubits))]
    positions = np.tile([(rows - 1) / 2, (cols - 1) / 2], (n_qubits, 1)).astype(float)
    while any((row1 - row0) * (col1 - col0) > 1 for row0, row1, col0, col1, _ in regions):
        positions = _qp(weights, regions, positions, linear, rounds)
        regions = [child for region in regions for child in _bisect(weights, region, positions)]
    return {int(qubit): (row0, col0) for row0, row1, col0, col1, members in regions for qubit in members}


def gordian_l(weights: np.ndarray, rows: int, cols: int, seed: int = 0) -> Order:
    return gordian(weights, rows, cols, seed, linear=True)


def _qp(weights: np.ndarray, regions: list[Region], positions: np.ndarray, linear: bool, rounds: int) -> np.ndarray:
    # Per axis, min sum_ij g_ij (p_i - p_j)^2 with each region's mean at its centre, g reweighted by 1/|d| when linear.
    n_qubits = len(weights)
    constraints = np.zeros((len(regions), n_qubits))
    centres = np.zeros((len(regions), 2))
    for index, (row0, row1, col0, col1, members) in enumerate(regions):
        constraints[index, members] = 1.0 / len(members)
        centres[index] = ((row0 + row1 - 1) / 2, (col0 + col1 - 1) / 2)
    zero = np.zeros((len(constraints), len(constraints)))

    solved = positions.copy()
    for axis in (0, 1):
        coord = positions[:, axis]
        for _ in range(rounds if linear else 1):
            if linear:
                reweighted = weights / np.maximum(np.abs(coord[:, None] - coord[None, :]), constant.GORDIAN_L_MIN_DISTANCE)
            else:
                reweighted = weights
            laplacian = np.diag(reweighted.sum(1)) - reweighted
            kkt = np.block([[2 * laplacian, constraints.T], [constraints, zero]])
            coord = np.linalg.solve(kkt, np.concatenate([np.zeros(n_qubits), centres[:, axis]]))[:n_qubits]
        solved[:, axis] = coord
    return solved


def _bisect(weights: np.ndarray, region: Region, positions: np.ndarray) -> list[Region]:
    row0, row1, col0, col1, members = region
    if len(members) == 0:
        return []
    capacity = (row1 - row0) * (col1 - col0)
    if capacity == 1:
        return [region]

    if row1 - row0 >= col1 - col0:
        split = (row1 - row0) // 2
        axis = 0
        first, second = (row0, row0 + split, col0, col1), (row0 + split, row1, col0, col1)
        first_capacity = split * (col1 - col0)
        line = row0 + split - 0.5
    else:
        split = (col1 - col0) // 2
        axis = 1
        first, second = (row0, row1, col0, col0 + split), (row0, row1, col0 + split, col1)
        first_capacity = split * (row1 - row0)
        line = col0 + split - 0.5

    # The halves take qubits in proportion to capacity, rounded toward the QP's own split.
    coord = positions[:, axis]
    expected = len(members) * first_capacity / capacity
    below = int((coord[members] < line).sum())
    n_first = min((math.floor(expected), math.ceil(expected)), key=lambda count: (abs(count - below), count))
    n_first = min(max(n_first, len(members) - (capacity - first_capacity)), first_capacity, len(members))
    ranked = members[np.lexsort((members, coord[members]))]
    side = np.sign(line - coord)   # outside qubits pull from their side of the line
    side[ranked[:n_first]], side[ranked[n_first:]] = 1.0, -1.0
    _kl(weights, members, side)
    halves = [(*first, members[side[members] > 0]), (*second, members[side[members] < 0])]
    return [half for half in halves if len(half[4])]


def _kl(weights: np.ndarray, members: np.ndarray, side: np.ndarray) -> None:
    # Best-pair swaps across the cut while the weighted cut drops.
    for _ in range(len(members)):
        first, second = members[side[members] > 0], members[side[members] < 0]
        if len(first) == 0 or len(second) == 0:
            return
        gain = -side * (weights @ side)   # cut reduction if one qubit flips
        swap_gain = gain[first][:, None] + gain[second][None, :] - 2 * weights[np.ix_(first, second)]
        best = int(np.argmax(swap_gain))
        if swap_gain.flat[best] <= 1e-9:
            return
        first_index, second_index = divmod(best, len(second))
        side[first[first_index]], side[second[second_index]] = -1.0, 1.0


# ---- recursive min-cut
def mincut(weights: np.ndarray, rows: int, cols: int, seed: int = 0, starts: int = constant.MINCUT_STARTS) -> Order:
    # An outside qubit pulls toward the half on its side of the cut line, judged from its region's centre at the
    # start of the level.
    rng = np.random.default_rng(seed)
    n_qubits = len(weights)
    regions: list[Region] = [(0, rows, 0, cols, np.arange(n_qubits))]
    while any((row1 - row0) * (col1 - col0) > 1 for row0, row1, col0, col1, _ in regions):
        centre = np.zeros((n_qubits, 2))
        for row0, row1, col0, col1, members in regions:
            centre[members] = ((row0 + row1 - 1) / 2, (col0 + col1 - 1) / 2)
        regions = [child for region in regions for child in _cut(weights, region, centre, rng, starts)]
    return {int(qubit): (row0, col0) for row0, row1, col0, col1, members in regions for qubit in members}


def _cut(weights: np.ndarray, region: Region, centre: np.ndarray, rng: np.random.Generator,
         starts: int) -> list[Region]:
    row0, row1, col0, col1, members = region
    capacity = (row1 - row0) * (col1 - col0)
    if capacity == 1:
        return [region]

    if row1 - row0 >= col1 - col0:
        split = (row1 - row0) // 2
        axis = 0
        first, second = (row0, row0 + split, col0, col1), (row0 + split, row1, col0, col1)
        first_capacity = split * (col1 - col0)
        line = row0 + split - 0.5
    else:
        split = (col1 - col0) // 2
        axis = 1
        first, second = (row0, row1, col0, col0 + split), (row0, row1, col0 + split, col1)
        first_capacity = split * (row1 - row0)
        line = col0 + split - 0.5

    n_members = len(members)
    expected = n_members * first_capacity / capacity
    sizes = sorted({min(max(size, n_members - (capacity - first_capacity)), first_capacity, n_members)
                    for size in (math.floor(expected), math.ceil(expected))})
    terminal = np.sign(line - centre[:, axis])   # +1 for the first half, -1 for the second and 0 on the line
    terminal[members] = 0.0
    member_weights = weights[np.ix_(members, members)]
    pull = weights[members] @ terminal   # net terminal weight toward the first half
    reach = weights[members] @ np.abs(terminal)   # total terminal weight

    best, best_cut = None, np.inf
    for n_first in sizes:
        for _ in range(starts):
            side = np.full(n_members, -1.0)
            side[rng.permutation(n_members)[:n_first]] = 1.0
            side = _kl_passes(member_weights, pull, side)
            cut = (member_weights * (side[:, None] != side[None, :])).sum() / 2 + (reach - side * pull).sum() / 2
            if cut < best_cut - 1e-9:
                best, best_cut = side, cut
    halves = [(*first, members[best > 0]), (*second, members[best < 0])]
    return [half for half in halves if len(half[4])]


def _kl_passes(member_weights: np.ndarray, pull: np.ndarray, side: np.ndarray) -> np.ndarray:
    # Each pass swaps the best unlocked pair across the cut until one side runs out, even at a loss, then keeps
    # the best prefix, and the loop stops when a pass gains nothing.
    side = side.copy()
    while True:
        field = member_weights @ side + pull   # weighted side of each qubit's neighbours plus its terminal pull
        current = side.copy()
        locked = np.zeros(len(side), dtype=bool)
        swaps = []
        total = 0.0
        best_total = 0.0
        best_count = 0
        while True:
            first = np.nonzero((current > 0) & ~locked)[0]
            second = np.nonzero((current < 0) & ~locked)[0]
            if len(first) == 0 or len(second) == 0:
                break
            gain = -current * field   # cut reduction if one qubit flips
            swap_gain = gain[first][:, None] + gain[second][None, :] - 2 * member_weights[np.ix_(first, second)]
            first_index, second_index = divmod(int(np.argmax(swap_gain)), len(second))
            node_u, node_v = first[first_index], second[second_index]
            total += swap_gain[first_index, second_index]
            field += -2 * member_weights[:, node_u] + 2 * member_weights[:, node_v]   # u flips to -1 and v to +1
            current[node_u], current[node_v] = -1.0, 1.0
            locked[node_u] = locked[node_v] = True
            swaps.append((node_u, node_v))
            if total > best_total + 1e-9:
                best_total, best_count = total, len(swaps)

        if best_count == 0:
            return side
        for node_u, node_v in swaps[:best_count]:
            side[node_u], side[node_v] = -1.0, 1.0


# ---- aplace and mpl, wirelength plus a smooth density penalty
def _bell(coords: np.ndarray, centers: np.ndarray, width: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # The bell-shaped overlap of cells of the given width with the density bins, and its derivative in coords.
    bin_width = constant.DENSITY_BIN
    delta = coords[:, None] - centers[None, :]
    distance = np.abs(delta)
    half = (width / 2)[:, None]
    near_coef = (4 / ((width + 2 * bin_width) * (width + 4 * bin_width)))[:, None]
    far_coef = (2 / (bin_width * (width + 4 * bin_width)))[:, None]
    near = distance <= half + bin_width
    far = (distance > half + bin_width) & (distance <= half + 2 * bin_width)
    overlap = np.where(near, 1 - near_coef * distance ** 2,
                       np.where(far, far_coef * (distance - half - 2 * bin_width) ** 2, 0.0))
    d_overlap = np.where(near, -2 * near_coef * distance,
                         np.where(far, 2 * far_coef * (distance - half - 2 * bin_width), 0.0)) * np.sign(delta)
    return overlap, d_overlap


def _centers(n_sites: int) -> np.ndarray:
    # Bin centres across sites 0..n_sites-1, the array spanning [-0.5, n_sites - 0.5].
    return np.arange(-0.5 + constant.DENSITY_BIN / 2, n_sites - 0.5, constant.DENSITY_BIN)


def _density(positions: np.ndarray, size: np.ndarray, rows: int, cols: int) -> tuple[np.ndarray, tuple]:
    width = np.sqrt(size)
    bell_rows, d_bell_rows = _bell(positions[:, 0], _centers(rows), width)
    bell_cols, d_bell_cols = _bell(positions[:, 1], _centers(cols), width)
    scale = size / np.maximum(bell_rows.sum(1) * bell_cols.sum(1), 1e-12)   # each cell deposits its area
    return (scale[:, None] * bell_rows).T @ bell_cols, (bell_rows, d_bell_rows, bell_cols, d_bell_cols, scale)


def _objective(weights: np.ndarray, size: np.ndarray, rows: int, cols: int, log_sum_exp: bool, density_weight: float):
    target = size.sum() / (rows * cols) * constant.DENSITY_BIN ** 2
    gamma = constant.APLACE_SMOOTHING

    def evaluate(positions: np.ndarray) -> tuple[float, np.ndarray]:
        grad = np.zeros_like(positions)
        wirelength = 0.0
        for axis in (0, 1):
            diff = positions[:, axis][:, None] - positions[:, axis][None, :]
            if log_sum_exp:   # the two-pin log-sum-exp 2 gamma log(2 cosh(d / 2 gamma))
                scaled = diff / (2 * gamma)
                wirelength += (weights * 2 * gamma * np.logaddexp(scaled, -scaled)).sum() / 2
                grad[:, axis] += (weights * np.tanh(scaled)).sum(1)
            else:
                wirelength += (weights * diff ** 2).sum() / 2
                grad[:, axis] += 2 * (weights * diff).sum(1)

        density, (bell_rows, d_bell_rows, bell_cols, d_bell_cols, scale) = _density(positions, size, rows, cols)
        excess = density - target
        grad[:, 0] += density_weight * 2 * scale * ((d_bell_rows @ excess) * bell_cols).sum(1)
        grad[:, 1] += density_weight * 2 * scale * ((bell_rows @ excess) * d_bell_cols).sum(1)
        return wirelength + density_weight * (excess ** 2).sum(), grad
    return evaluate


def _cg(objective, positions: np.ndarray, iters: int, upper: np.ndarray) -> np.ndarray:
    # Projected Polak-Ribiere+ conjugate gradient with backtracking.
    value, grad = objective(positions)
    direction = -grad
    for _ in range(iters):
        step = 1.0 / max(np.abs(direction).max(), 1e-12)   # the first trial moves a cell one site
        while True:
            trial = np.clip(positions + step * direction, 0.0, upper)
            trial_value, trial_grad = objective(trial)
            if trial_value <= value + 1e-4 * (grad * (trial - positions)).sum() or step < 1e-6:
                break
            step *= 0.5

        beta = max(0.0, (trial_grad * (trial_grad - grad)).sum() / max((grad * grad).sum(), 1e-12))
        direction = -trial_grad + beta * direction
        if (trial_grad * direction).sum() >= 0:
            direction = -trial_grad
        positions, value, grad = trial, trial_value, trial_grad
    return positions


def _spread(weights: np.ndarray, size: np.ndarray, positions: np.ndarray, rows: int, cols: int, log_sum_exp: bool,
            start: float = constant.DENSITY_START, outer: int = constant.DENSITY_ROUNDS,
            inner: int = constant.DENSITY_ITERS, overflow: float = constant.DENSITY_OVERFLOW) -> np.ndarray:
    # The density weight starts small relative to the wirelength gradient, so wirelength clusters first, and
    # doubles per round until the overflow drops below the threshold.
    upper = np.array([rows - 1, cols - 1], float)
    wirelength_grad = _objective(weights, size, rows, cols, log_sum_exp, 0.0)(positions)[1]
    density_grad = _objective(weights, size, rows, cols, log_sum_exp, 1.0)(positions)[1] - wirelength_grad
    density_weight = start * np.abs(wirelength_grad).sum() / max(np.abs(density_grad).sum(), 1e-12)
    for _ in range(outer):
        positions = _cg(_objective(weights, size, rows, cols, log_sum_exp, density_weight), positions, inner, upper)
        density, _ = _density(positions, size, rows, cols)
        if np.maximum(density - constant.DENSITY_BIN ** 2, 0.0).sum() / size.sum() < overflow:
            break
        density_weight *= 2
    return positions


def _cloud(n_cells: int, rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    centre = np.array([(rows - 1) / 2, (cols - 1) / 2])
    return centre + rng.uniform(-constant.CLOUD_SPREAD, constant.CLOUD_SPREAD, (n_cells, 2)) * np.array([rows, cols])


def aplace(weights: np.ndarray, rows: int, cols: int, seed: int = 0) -> Order:
    rng = np.random.default_rng(seed)
    n_qubits = len(weights)
    positions = _spread(weights, np.ones(n_qubits), _cloud(n_qubits, rows, cols, rng), rows, cols, log_sum_exp=True)
    return legalize(positions, rows, cols)


def mpl(weights: np.ndarray, rows: int, cols: int, seed: int = 0, coarsest: int = constant.MPL_COARSEST) -> Order:
    # Down the V-cycle heavy-edge matching coarsens while it shrinks the graph enough, and on the way up each
    # child starts at its cluster's position and the level is re-solved.
    rng = np.random.default_rng(seed)
    n_qubits = len(weights)
    cluster_cap = max(2, int(constant.MPL_CLUSTER_FRACTION * n_qubits))
    levels = []
    level_weights = weights
    size = np.ones(n_qubits)
    while len(level_weights) > coarsest:
        parent = _match(level_weights, size, cluster_cap, rng)
        n_clusters = int(parent.max()) + 1
        if n_clusters > constant.MPL_SHRINK * len(level_weights):
            break
        clusters = np.zeros((len(level_weights), n_clusters))
        clusters[np.arange(len(level_weights)), parent] = 1.0
        levels.append((level_weights, size, parent))
        level_weights = clusters.T @ level_weights @ clusters
        size = clusters.T @ size
        np.fill_diagonal(level_weights, 0.0)

    upper = np.array([rows - 1, cols - 1], float)
    positions = _spread(level_weights, size, _cloud(len(level_weights), rows, cols, rng), rows, cols, log_sum_exp=False)
    for fine_weights, fine_sizes, parent in reversed(levels):
        positions = np.clip(positions[parent] + rng.normal(0.0, constant.MPL_JITTER, (len(fine_weights), 2)), 0.0, upper)
        positions = _spread(fine_weights, fine_sizes, positions, rows, cols, log_sum_exp=False)
    return legalize(positions, rows, cols)


def _match(weights: np.ndarray, size: np.ndarray, cluster_cap: int, rng: np.random.Generator) -> np.ndarray:
    # Heavy-edge matching by w_uv / (s_u s_v), merging only pairs with a real interaction above the floor and
    # within the size cap, gives the cluster id of each node.
    n_nodes = len(weights)
    parent = np.full(n_nodes, -1)
    next_cluster = 0
    for node in rng.permutation(n_nodes):
        if parent[node] >= 0:
            continue
        rating = weights[node] / (size[node] * size)
        eligible = (parent < 0) & (size[node] + size <= cluster_cap) & (rating > constant.INTERACTION_FLOOR * (1 + 1e-6))
        eligible[node] = False
        parent[node] = next_cluster
        if eligible.any():
            mate = int(np.argmax(np.where(eligible, rating, -np.inf)))
            parent[mate] = next_cluster
        next_cluster += 1
    return parent


ORDERS = {"random": random_order, "gordian": gordian, "gordian-l": gordian_l, "mincut": mincut, "aplace": aplace, "mpl": mpl}
