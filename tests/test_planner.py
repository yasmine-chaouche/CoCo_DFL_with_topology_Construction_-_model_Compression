"""Checks the exact LP solver of Eq. (27) against scipy and sanity-checks Algorithm 3.

Run:  python -m pytest tests   (or)   python tests/test_planner.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coco.consensus import floyd_warshall  # noqa: E402
from coco.network import is_connected, round_time, slowest_link_factor  # noqa: E402
from coco.planner import (CoCoPlanner, PlanContext, circulant, complete,  # noqa: E402
                          solve_ratio_lp)


def random_ctx(m, rng, dmax_frac):
    X = rng.normal(size=(m, 8))
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    ctx = PlanContext(A_base=complete(m), D=D, Dmax=0.0, b_in=rng.uniform(1, 20, m),
                      b_out=rng.uniform(0.5, 10, m), t_cp=rng.uniform(2.5, 7.5, m), model_size=111.7)
    ctx.Dmax = dmax_frac * D.mean()
    return ctx


def scipy_lp(A, ctx, r_lo, r_hi):
    from scipy.optimize import linprog
    m = len(A)
    beta = slowest_link_factor(A, ctx.b_in, ctx.b_out, ctx.model_size)
    # variables [r_1..r_m, t]; minimise t
    c = np.r_[np.zeros(m), 1.0]
    A_ub, b_ub = [], []
    for i in range(m):  # t_cp_i + beta_i r_i <= t
        row = np.zeros(m + 1)
        row[i], row[m] = beta[i], -1
        A_ub.append(row)
        b_ub.append(-ctx.t_cp[i])
    w = (A * ctx.D).sum(axis=0)  # -(w . r) <= Dmax m^2 - sum D
    A_ub.append(np.r_[-w, 0.0])
    b_ub.append(ctx.Dmax * m * m - ctx.D.sum())
    res = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub,
                  bounds=[(r_lo, r_hi)] * m + [(None, None)], method="highs")
    return res.fun if res.status == 0 else np.inf


def test_lp_matches_scipy():
    rng = np.random.default_rng(0)
    for trial in range(200):
        m = int(rng.integers(4, 15))
        ctx = random_ctx(m, rng, rng.uniform(0.0, 1.2))
        A = circulant(m, int(rng.integers(1, (m - 1) // 2 + 1)) * 2) if trial % 2 else complete(m)
        for tb in ("min_traffic", "max_ratio"):
            r, t = solve_ratio_lp(A, ctx, 0.01, 1.0, tb)
            t_ref = scipy_lp(A, ctx, 0.01, 1.0)
            if np.isinf(t_ref):
                assert r is None
                continue
            assert abs(t - t_ref) <= 1e-6 * t_ref, (trial, t, t_ref)
            assert abs(t - round_time(A, r, ctx.b_in, ctx.b_out, ctx.t_cp, ctx.model_size)) < 1e-6 * t
            lhs = ((1 - A * r[None, :]) * ctx.D).sum() / m ** 2
            assert lhs <= ctx.Dmax + 1e-6 * max(1, ctx.Dmax)


def test_coco_improves_and_stays_feasible():
    rng = np.random.default_rng(1)
    for _ in range(30):
        m = 20
        ctx = random_ctx(m, rng, rng.uniform(0.3, 1.0))
        r0, t0 = solve_ratio_lp(complete(m), ctx, 0.01, 1.0)
        A, r = CoCoPlanner().plan(ctx)
        assert is_connected(A) and (A == A.T).all() and np.trace(A) == 0
        t = round_time(A, r, ctx.b_in, ctx.b_out, ctx.t_cp, ctx.model_size)
        if r0 is not None:
            assert t <= t0 + 1e-9
            lhs = ((1 - A * r[None, :]) * ctx.D).sum() / m ** 2
            assert lhs <= ctx.Dmax + 1e-6


def test_floyd_warshall():
    W = np.array([[0, 1, np.inf], [1, 0, 2], [np.inf, 2, 0]], dtype=float)
    assert floyd_warshall(W)[0, 2] == 3


if __name__ == "__main__":
    test_lp_matches_scipy()
    test_coco_improves_and_stays_feasible()
    test_floyd_warshall()
    print("all tests passed")
