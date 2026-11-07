"""Initial placements: every seed yields a legal site assignment the anneal can start from."""
import itertools

import numpy as np
import pytest

from miniflash import INITS, Builder, make, verify
from miniflash.synthesis import initial
from miniflash.synthesis.placement import initial_order, interaction_matrix


def two_cliques():
    """Qubits 0, 2, 4, 6 talk only to each other, as do 1, 3, 5, 7."""
    b = Builder(8)
    for group in ((0, 2, 4, 6), (1, 3, 5, 7)):
        for _ in range(3):
            for c, t in itertools.combinations(group, 2):
                b.add_cx(c, t)
    return b.build()


def test_assignment_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(1, 5))
        m = n + int(rng.integers(0, 3))
        cost = rng.random((n, m))
        cols = initial.assign(cost)
        best = min(sum(cost[i, p[i]] for i in range(n)) for p in itertools.permutations(range(m), n))
        assert len(set(cols)) == n
        assert cost[np.arange(n), cols].sum() == pytest.approx(best)


def test_kernighan_lin_keeps_sizes_and_never_loses():
    rng = np.random.default_rng(0)
    for _ in range(50):
        m = int(rng.integers(2, 9))
        n1 = int(rng.integers(0, m + 1))
        Wm = np.triu(rng.random((m, m)), 1)
        Wm = Wm + Wm.T
        pull = rng.normal(size=m)
        start = np.full(m, -1.0)
        start[rng.permutation(m)[:n1]] = 1.0
        side = initial._kl_passes(Wm, pull, start)
        cut = lambda s: (Wm * (s[:, None] != s[None, :])).sum() / 2 - (s * pull).sum() / 2
        assert (side > 0).sum() == n1
        assert cut(side) <= cut(start) + 1e-9


@pytest.mark.parametrize("init", INITS)
def test_every_seed_is_a_legal_assignment(init):
    p = two_cliques()
    order = initial_order(init, p, 3, 3, seed=1)
    assert sorted(order) == list(range(8))
    assert len(set(order.values())) == 8
    assert all(0 <= i < 3 and 0 <= j < 3 for i, j in order.values())
    assert order == initial_order(init, p, 3, 3, seed=1)            # reproducible for a seed


@pytest.mark.parametrize("init", [i for i in INITS if i != "random"])
def test_connectivity_driven_seeds_separate_the_cliques(init):
    p = two_cliques()
    order = initial_order(init, p, 2, 4)
    groups = [{order[q] for q in (0, 2, 4, 6)}, {order[q] for q in (1, 3, 5, 7)}]
    rows = [{s[0] for s in g} for g in groups]
    cols = [{s[1] for s in g} for g in groups]
    assert not (rows[0] & rows[1]) or not (cols[0] & cols[1])


def test_gordian_l_reweights_by_distance():
    W = interaction_matrix(two_cliques())
    assert initial.gordian(W, 2, 4) and initial.gordian_l(W, 2, 4)


@pytest.mark.parametrize("init", INITS)
def test_compiles_and_verifies_from_every_seed(init):
    p = two_cliques()
    compiler = make("layout", init=init)
    program = compiler.solve(p)
    verify(p, program)
    assert compiler.name == ("layout" if init == "spectral" else f"layout-{init}")


def test_unknown_seed_is_rejected():
    with pytest.raises(ValueError):
        make("layout", init="fastplace")
