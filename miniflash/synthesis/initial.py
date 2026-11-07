"""Initial placement: the site assignment the anneal starts from.

Every seed reads the same weighted interaction graph `W` (CX counts between qubit
pairs plus a uniform 1e-3 between all pairs, so the graph is connected) and returns
qubit -> virtual site (i, j) on a rows x cols site array. Continuous coordinates
are site indices: row i, column j, unit pitch.

    spectral   placement.spectral_order: Laplacian modes 2 and 3 rank rows, then columns
    random     a uniformly random assignment, the start of DasCot (SA alone)
    gordian    GORDIAN: quadratic wirelength under centre-of-gravity constraints,
               alternating with bisection of every region until each holds one site
    gordian-l  GORDIAN-L: the same, with the linear objective, each QP reweighting an
               edge by 1/|d| from the previous solution
    mincut     Breuer's recursive min-cut: bisect every region by Kernighan-Lin from
               random starts, outside qubits as propagated terminals, no coordinates
    aplace     APlace: log-sum-exp wirelength plus a bell-shaped bin density penalty,
               conjugate gradient, the penalty doubling until the spread is even
    mpl        mPL: cluster by heavy-edge matching, place the coarsest graph with
               quadratic wirelength plus density, then decluster and relax level by level

aplace and mpl end with a minimum-cost assignment of qubits to sites (`legalize`).
These are compact ports of each method's core formulation to a few hundred qubits,
not the original IC placers.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

Order = Dict[int, Tuple[int, int]]
EPS = 0.25          # GORDIAN-L: smallest distance an edge is reweighted by, in sites
GAMMA = 0.5         # APlace: log-sum-exp smoothing, in sites


# ---- shared pieces -------------------------------------------------------------------
def legalize(pos: np.ndarray, rows: int, cols: int) -> Order:
    """Distinct sites minimizing the total squared displacement from `pos`."""
    sites = np.array([(i, j) for i in range(rows) for j in range(cols)], float)
    cost = ((pos[:, None, :] - sites[None, :, :]) ** 2).sum(2)
    cols_of = assign(cost)
    return {q: (int(sites[s][0]), int(sites[s][1])) for q, s in enumerate(cols_of)}


def assign(cost: np.ndarray) -> np.ndarray:
    """Hungarian method for an n x m cost matrix, n <= m: the column of each row."""
    n, m = cost.shape
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)          # p[j]: the row (1-based) holding column j
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            free = ~used[1:]
            cur = cost[p[j0] - 1] - u[p[j0]] - v[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            cand = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            held = np.nonzero(used)[0]
            u[p[held]] += delta
            v[held] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    out = np.zeros(n, dtype=int)
    for j in range(1, m + 1):
        if p[j]:
            out[p[j] - 1] = j - 1
    return out


def random_order(W: np.ndarray, rows: int, cols: int, seed: int = 0) -> Order:
    sites = np.random.default_rng(seed).permutation(rows * cols)[:len(W)]
    return {q: (int(s) // cols, int(s) % cols) for q, s in enumerate(sites)}


# ---- GORDIAN and GORDIAN-L -----------------------------------------------------------
Region = Tuple[int, int, int, int, np.ndarray]     # rows r0..r1-1, cols c0..c1-1, member qubits


def gordian(W: np.ndarray, rows: int, cols: int, seed: int = 0, linear: bool = False, rounds: int = 3) -> Order:
    """Solve the constrained QP, bisect every region along its longer side by the QP
    coordinate (refined by KL swaps with terminal propagation), repeat until every
    region is one site. Deterministic; `seed` is unused."""
    n = len(W)
    regions: List[Region] = [(0, rows, 0, cols, np.arange(n))]
    pos = np.tile([(rows - 1) / 2, (cols - 1) / 2], (n, 1)).astype(float)
    while any((r1 - r0) * (c1 - c0) > 1 for r0, r1, c0, c1, _ in regions):
        pos = _qp(W, regions, pos, linear, rounds)
        regions = [child for region in regions for child in _bisect(W, region, pos)]
    return {int(q): (r0, c0) for r0, r1, c0, c1, members in regions for q in members}


def gordian_l(W: np.ndarray, rows: int, cols: int, seed: int = 0) -> Order:
    return gordian(W, rows, cols, seed, linear=True)


def _qp(W, regions, pos, linear, rounds):
    """min sum_ij g_ij (p_i - p_j)^2 per axis s.t. each region's mean sits at its centre.
    Linear: g_ij = w_ij / max(|p_i - p_j|, EPS) from the previous solution, `rounds` times."""
    n = len(W)
    A = np.zeros((len(regions), n))
    b = np.zeros((len(regions), 2))
    for k, (r0, r1, c0, c1, members) in enumerate(regions):
        A[k, members] = 1.0 / len(members)
        b[k] = ((r0 + r1 - 1) / 2, (c0 + c1 - 1) / 2)
    zero = np.zeros((len(A), len(A)))
    out = pos.copy()
    for axis in (0, 1):
        x = pos[:, axis]
        for _ in range(rounds if linear else 1):
            g = W / np.maximum(np.abs(x[:, None] - x[None, :]), EPS) if linear else W
            lap = np.diag(g.sum(1)) - g
            kkt = np.block([[2 * lap, A.T], [A, zero]])
            x = np.linalg.solve(kkt, np.concatenate([np.zeros(n), b[:, axis]]))[:n]
        out[:, axis] = x
    return out


def _bisect(W, region: Region, pos) -> List[Region]:
    r0, r1, c0, c1, members = region
    if len(members) == 0:
        return []
    cap = (r1 - r0) * (c1 - c0)
    if cap == 1:
        return [region]
    if r1 - r0 >= c1 - c0:
        k, axis = (r1 - r0) // 2, 0
        first, second, cap1 = (r0, r0 + k, c0, c1), (r0 + k, r1, c0, c1), k * (c1 - c0)
        line = r0 + k - 0.5
    else:
        k, axis = (c1 - c0) // 2, 1
        first, second, cap1 = (r0, r1, c0, c0 + k), (r0, r1, c0 + k, c1), k * (r1 - r0)
        line = c0 + k - 0.5
    coord = pos[:, axis]
    # area balance: the share of qubits proportional to capacity, rounded toward the QP's own split
    frac = len(members) * cap1 / cap
    below = int((coord[members] < line).sum())
    n1 = min((math.floor(frac), math.ceil(frac)), key=lambda c: (abs(c - below), c))
    n1 = min(max(n1, len(members) - (cap - cap1)), cap1, len(members))
    ranked = members[np.lexsort((members, coord[members]))]
    side = np.sign(line - coord)                    # outside qubits pull from their side of the line
    side[ranked[:n1]], side[ranked[n1:]] = 1.0, -1.0
    _kl(W, members, side)
    halves = [(*first, members[side[members] > 0]), (*second, members[side[members] < 0])]
    return [h for h in halves if len(h[4])]


def _kl(W, members, side):
    """Best-pair swaps across the cut while the weighted cut drops."""
    for _ in range(len(members)):
        a, b = members[side[members] > 0], members[side[members] < 0]
        if len(a) == 0 or len(b) == 0:
            return
        gain = -side * (W @ side)                       # cut reduction if one qubit flips
        swap = gain[a][:, None] + gain[b][None, :] - 2 * W[np.ix_(a, b)]
        k = int(np.argmax(swap))
        if swap.flat[k] <= 1e-9:
            return
        i, j = divmod(k, len(b))
        side[a[i]], side[b[j]] = -1.0, 1.0


# ---- recursive min-cut ---------------------------------------------------------------
def mincut(W: np.ndarray, rows: int, cols: int, seed: int = 0, starts: int = 4) -> Order:
    """Breuer's recursive min-cut with Dunlop-Kernighan terminal propagation. Level by
    level, every region is cut across its longer side; the qubits split in proportion
    to the two halves' capacity, by the best of `starts` random partitions refined by
    Kernighan-Lin. A qubit outside the region pulls toward the half on its side of the
    cut line, judged from the centre of its region at the start of the level."""
    rng = np.random.default_rng(seed)
    n = len(W)
    regions: List[Region] = [(0, rows, 0, cols, np.arange(n))]
    while any((r1 - r0) * (c1 - c0) > 1 for r0, r1, c0, c1, _ in regions):
        centre = np.zeros((n, 2))
        for r0, r1, c0, c1, members in regions:
            centre[members] = ((r0 + r1 - 1) / 2, (c0 + c1 - 1) / 2)
        regions = [child for region in regions for child in _cut(W, region, centre, rng, starts)]
    return {int(q): (r0, c0) for r0, r1, c0, c1, members in regions for q in members}


def _cut(W, region: Region, centre, rng, starts) -> List[Region]:
    r0, r1, c0, c1, members = region
    cap = (r1 - r0) * (c1 - c0)
    if cap == 1:
        return [region]
    if r1 - r0 >= c1 - c0:
        k, axis = (r1 - r0) // 2, 0
        first, second, cap1 = (r0, r0 + k, c0, c1), (r0 + k, r1, c0, c1), k * (c1 - c0)
        line = r0 + k - 0.5
    else:
        k, axis = (c1 - c0) // 2, 1
        first, second, cap1 = (r0, r1, c0, c0 + k), (r0, r1, c0 + k, c1), k * (r1 - r0)
        line = c0 + k - 0.5
    m = len(members)
    frac = m * cap1 / cap
    sizes = sorted({min(max(s, m - (cap - cap1)), cap1, m) for s in (math.floor(frac), math.ceil(frac))})
    terminal = np.sign(line - centre[:, axis])          # +1 first half, -1 second, 0 on the line
    terminal[members] = 0.0
    Wm = W[np.ix_(members, members)]
    pull = W[members] @ terminal                        # net terminal weight toward the first half
    reach = W[members] @ np.abs(terminal)               # total terminal weight
    best, best_cut = None, np.inf
    for n1 in sizes:
        for _ in range(starts):
            s = np.full(m, -1.0)
            s[rng.permutation(m)[:n1]] = 1.0
            s = _kl_passes(Wm, pull, s)
            cut = (Wm * (s[:, None] != s[None, :])).sum() / 2 + (reach - s * pull).sum() / 2
            if cut < best_cut - 1e-9:
                best, best_cut = s, cut
    halves = [(*first, members[best > 0]), (*second, members[best < 0])]
    return [h for h in halves if len(h[4])]


def _kl_passes(Wm, pull, s):
    """Kernighan-Lin on the member graph `Wm`, with `pull` from fixed terminals: each pass
    swaps the best unlocked pair across the cut until one side runs out, even at a loss,
    then keeps the best prefix; stops when a pass gains nothing."""
    s = s.copy()
    while True:
        h = Wm @ s + pull
        cur = s.copy()
        locked = np.zeros(len(s), dtype=bool)
        swaps, total, best_total, best_k = [], 0.0, 0.0, 0
        while True:
            a = np.nonzero((cur > 0) & ~locked)[0]
            b = np.nonzero((cur < 0) & ~locked)[0]
            if len(a) == 0 or len(b) == 0:
                break
            gain = -cur * h                             # cut reduction if one qubit flips
            pair = gain[a][:, None] + gain[b][None, :] - 2 * Wm[np.ix_(a, b)]
            i, j = divmod(int(np.argmax(pair)), len(b))
            u, v = a[i], b[j]
            total += pair[i, j]
            h += -2 * Wm[:, u] + 2 * Wm[:, v]           # u: +1 -> -1, v: -1 -> +1
            cur[u], cur[v] = -1.0, 1.0
            locked[u] = locked[v] = True
            swaps.append((u, v))
            if total > best_total + 1e-9:
                best_total, best_k = total, len(swaps)
        if best_k == 0:
            return s
        for u, v in swaps[:best_k]:
            s[u], s[v] = -1.0, 1.0


# ---- APlace and mPL: wirelength plus a smooth density penalty --------------------------
BIN = 0.5           # density bins per site pitch: 2 x 2 bins per site, each holding a quarter qubit


def _bell(x, centers, width):
    """APlace's bell-shaped overlap of cells (width `width`) with bins of width BIN: value, d/dx."""
    d = x[:, None] - centers[None, :]
    ad = np.abs(d)
    half = (width / 2)[:, None]
    a = (4 / ((width + 2 * BIN) * (width + 4 * BIN)))[:, None]
    b = (2 / (BIN * (width + 4 * BIN)))[:, None]
    near, far = ad <= half + BIN, (ad > half + BIN) & (ad <= half + 2 * BIN)
    p = np.where(near, 1 - a * ad ** 2, np.where(far, b * (ad - half - 2 * BIN) ** 2, 0.0))
    dp = np.where(near, -2 * a * ad, np.where(far, 2 * b * (ad - half - 2 * BIN), 0.0)) * np.sign(d)
    return p, dp


def _centers(k):
    """Bin centres across sites 0..k-1, the array spanning [-0.5, k - 0.5]."""
    return np.arange(-0.5 + BIN / 2, k - 0.5, BIN)


def _density(P, size, rows, cols):
    width = np.sqrt(size)
    py, dpy = _bell(P[:, 0], _centers(rows), width)
    px, dpx = _bell(P[:, 1], _centers(cols), width)
    scale = size / np.maximum(py.sum(1) * px.sum(1), 1e-12)   # each cell deposits its area
    return (scale[:, None] * py).T @ px, (py, dpy, px, dpx, scale)


def _objective(W, size, rows, cols, lse, lam):
    target = size.sum() / (rows * cols) * BIN * BIN

    def f(P):
        grad = np.zeros_like(P)
        wl = 0.0
        for axis in (0, 1):
            diff = P[:, axis][:, None] - P[:, axis][None, :]
            if lse:   # 2-pin log-sum-exp: 2*gamma*log(2*cosh(d / (2*gamma)))
                z = diff / (2 * GAMMA)
                wl += (W * 2 * GAMMA * np.logaddexp(z, -z)).sum() / 2
                grad[:, axis] += (W * np.tanh(z)).sum(1)
            else:
                wl += (W * diff ** 2).sum() / 2
                grad[:, axis] += 2 * (W * diff).sum(1)
        D, (py, dpy, px, dpx, scale) = _density(P, size, rows, cols)
        E = D - target
        grad[:, 0] += lam * 2 * scale * ((dpy @ E) * px).sum(1)
        grad[:, 1] += lam * 2 * scale * ((py @ E) * dpx).sum(1)
        return wl + lam * (E ** 2).sum(), grad
    return f


def _cg(f, P, iters, hi):
    """Projected nonlinear conjugate gradient (Polak-Ribiere+) with backtracking."""
    val, g = f(P)
    d = -g
    for _ in range(iters):
        step = 1.0 / max(np.abs(d).max(), 1e-12)          # the first trial moves a cell one site
        while True:
            Q = np.clip(P + step * d, 0.0, hi)
            v, gq = f(Q)
            if v <= val + 1e-4 * (g * (Q - P)).sum() or step < 1e-6:
                break
            step *= 0.5
        beta = max(0.0, (gq * (gq - g)).sum() / max((g * g).sum(), 1e-12))
        d = -gq + beta * d
        if (gq * d).sum() >= 0:
            d = -gq
        P, val, g = Q, v, gq
    return P


def _spread(W, size, P, rows, cols, lse, start=0.1, outer=16, inner=30, overflow=0.1):
    """Penalty loop: the density weight starts at `start` times the gradient-norm ratio
    (small from a cold start, so wirelength clusters first) and doubles per round
    until the overflow drops below `overflow`."""
    hi = np.array([rows - 1, cols - 1], float)
    wl_grad = _objective(W, size, rows, cols, lse, 0.0)(P)[1]
    den_grad = _objective(W, size, rows, cols, lse, 1.0)(P)[1] - wl_grad
    lam = start * np.abs(wl_grad).sum() / max(np.abs(den_grad).sum(), 1e-12)
    for _ in range(outer):
        P = _cg(_objective(W, size, rows, cols, lse, lam), P, inner, hi)
        D, _ = _density(P, size, rows, cols)
        if np.maximum(D - BIN * BIN, 0.0).sum() / size.sum() < overflow:
            break
        lam *= 2
    return P


def _cloud(n, rows, cols, rng):
    """A random cloud over the middle half of the array."""
    centre = np.array([(rows - 1) / 2, (cols - 1) / 2])
    return centre + rng.uniform(-0.25, 0.25, (n, 2)) * np.array([rows, cols])


def aplace(W: np.ndarray, rows: int, cols: int, seed: int = 0) -> Order:
    rng = np.random.default_rng(seed)
    n = len(W)
    P = _spread(W, np.ones(n), _cloud(n, rows, cols, rng), rows, cols, lse=True)
    return legalize(P, rows, cols)


def mpl(W: np.ndarray, rows: int, cols: int, seed: int = 0, coarsest: int = 4) -> Order:
    """V-cycle down: heavy-edge matching while it shrinks the graph by 10%; up: each
    child starts at its cluster's position and the level is re-solved."""
    rng = np.random.default_rng(seed)
    n = len(W)
    cap = max(2, n // 4)                               # no cluster beyond a quarter of the qubits
    levels = []
    Wl, size = W, np.ones(n)
    while len(Wl) > coarsest:
        parent = _match(Wl, size, cap, rng)
        k = int(parent.max()) + 1
        if k > 0.9 * len(Wl):
            break
        S = np.zeros((len(Wl), k))
        S[np.arange(len(Wl)), parent] = 1.0
        levels.append((Wl, size, parent))
        Wl, size = S.T @ Wl @ S, S.T @ size
        np.fill_diagonal(Wl, 0.0)
    hi = np.array([rows - 1, cols - 1], float)
    P = _spread(Wl, size, _cloud(len(Wl), rows, cols, rng), rows, cols, lse=False)
    for Wf, sf, parent in reversed(levels):
        P = np.clip(P[parent] + rng.normal(0.0, 0.1, (len(Wf), 2)), 0.0, hi)
        P = _spread(Wf, sf, P, rows, cols, lse=False)
    return legalize(P, rows, cols)


def _match(W, size, cap, rng) -> np.ndarray:
    """Heavy-edge matching, rating w_uv / (s_u s_v); only pairs with a real interaction
    (above the uniform 1e-3) and within the size cap merge. Cluster id per node."""
    n = len(W)
    parent = np.full(n, -1)
    nxt = 0
    for u in rng.permutation(n):
        if parent[u] >= 0:
            continue
        rating = W[u] / (size[u] * size)
        ok = (parent < 0) & (size[u] + size <= cap) & (rating > 1e-3 * (1 + 1e-6))
        ok[u] = False
        parent[u] = nxt
        if ok.any():
            v = int(np.argmax(np.where(ok, rating, -np.inf)))
            parent[v] = nxt
        nxt += 1
    return parent


ORDERS = {"random": random_order, "gordian": gordian, "gordian-l": gordian_l, "mincut": mincut, "aplace": aplace, "mpl": mpl}
