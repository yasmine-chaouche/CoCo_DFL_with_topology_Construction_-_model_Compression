"""Coordinator-side planning: CoCo (Algorithm 3) and the baselines.

Every planner exposes ``plan(ctx) -> (A, r)`` where ``ctx`` is a ``PlanContext``
holding the statuses reported by the workers (Algorithm 2, line 2).
"""
from dataclasses import dataclass

import numpy as np

from .network import is_connected, link_times, round_time, slowest_link_factor


@dataclass
class PlanContext:
    A_base: np.ndarray   # available P2P links (A_b)
    D: np.ndarray        # pairwise consensus distances D_ij (estimated)
    Dmax: float          # consensus distance threshold (Eq. 28)
    b_in: np.ndarray     # inbound bandwidth (Mb/s)
    b_out: np.ndarray    # outbound bandwidth (Mb/s)
    t_cp: np.ndarray     # computation time tau * c_i (s)
    model_size: float    # B, uncompressed model size (Mb)
    round_idx: int = 0


# --------------------------------------------------------------------------- LP

def solve_ratio_lp(A, ctx, r_lo, r_hi, tiebreak="min_traffic"):
    """Exact solution of the linear program (27) for a fixed topology A.

        min_r  t = max_{e_ij in E} t_cp_i + r_i B / b_ij
        s.t.   (1/m^2) sum_i sum_j (1 - a_ij r_j) D_ij <= Dmax,   r_lo <= r <= r_hi

    Worker i's slowest link gives t_i(r_i) = t_cp_i + r_i * beta_i, and the consensus
    constraint is equivalent to  sum_j w_j r_j >= C  with  w_j = sum_i a_ij D_ij  and
    C = sum_ij D_ij - m^2 Dmax.  For a target round time t the largest admissible
    ratios are r_j(t) = clip((t - t_cp_j) / beta_j, r_lo, r_hi), and
    f(t) = sum_j w_j r_j(t) is piecewise linear and non-decreasing, so the optimal
    t* is found exactly by scanning its breakpoints.

    Among the optimal solutions, ``tiebreak`` picks either the largest ratios that
    fit within t* ("max_ratio") or the ones with minimal traffic ("min_traffic",
    a fractional knapsack solved greedily).

    Returns (r, t) or (None, inf) if infeasible.
    """
    m = len(A)
    r_lo = np.broadcast_to(np.asarray(r_lo, dtype=float), (m,)).copy()
    r_hi = np.broadcast_to(np.asarray(r_hi, dtype=float), (m,)).copy()
    beta = slowest_link_factor(A, ctx.b_in, ctx.b_out, ctx.model_size)
    t_cp = ctx.t_cp
    w = (A * ctx.D).sum(axis=0)
    C = ctx.D.sum() - m * m * ctx.Dmax
    tol = 1e-9 * max(1.0, abs(C))

    if w @ r_hi < C - tol:
        return None, np.inf

    r_at = lambda t: np.clip((t - t_cp) / beta, r_lo, r_hi)
    f = lambda t: w @ r_at(t)
    t_lo = float(np.max(t_cp + r_lo * beta))
    if f(t_lo) >= C - tol:
        t_star = t_lo
    else:
        bps = np.unique(np.concatenate([t_cp + r_lo * beta, t_cp + r_hi * beta]))
        bps = bps[bps > t_lo]
        prev_t, prev_f = t_lo, f(t_lo)
        t_star = float(bps[-1])
        for bp in bps:
            fb = f(bp)
            if fb >= C - tol:
                # f is linear on [prev_t, bp]
                t_star = prev_t + (C - prev_f) * (bp - prev_t) / max(fb - prev_f, 1e-300)
                break
            prev_t, prev_f = bp, fb

    upper = r_at(t_star)
    if tiebreak == "max_ratio":
        r = upper
    else:
        deg = A.sum(axis=1)
        r = r_lo.copy()
        deficit = C - w @ r
        if deficit > tol:
            # cheapest consensus reduction per unit of traffic first
            order = np.argsort(-(w / np.maximum(deg, 1)))
            for j in order:
                if deficit <= tol or w[j] <= 0:
                    break
                room = upper[j] - r[j]
                inc = min(room, deficit / w[j])
                r[j] += inc
                deficit -= inc * w[j]
            r = np.minimum(np.maximum(r, r_lo), upper)
    t = float(np.max(t_cp + r * beta))
    return r, t


# --------------------------------------------------------------------- CoCo

def undirected_links(A):
    i, j = np.nonzero(np.triu(A, 1))
    return list(zip(i.tolist(), j.tolist()))


class CoCoPlanner:
    """Algorithm 3: joint topology construction and compression-ratio selection.

    ``fixed_ratio`` turns it into the FCR ablation (topology search only),
    ``fixed_topology`` into the FNT ablation (ratios only).
    """

    def __init__(self, r_min=0.01, r_max=1.0, tiebreak="min_traffic",
                 fixed_ratio=None, fixed_topology=None):
        self.r_min, self.r_max, self.tiebreak = r_min, r_max, tiebreak
        self.fixed_ratio, self.fixed_topology = fixed_ratio, fixed_topology
        self.last_stats = {}

    def _bounds(self):
        if self.fixed_ratio is not None:
            return self.fixed_ratio, self.fixed_ratio
        return self.r_min, self.r_max

    def _solve(self, A, ctx):
        lo, hi = self._bounds()
        return solve_ratio_lp(A, ctx, lo, hi, self.tiebreak)

    def plan(self, ctx):
        A = (self.fixed_topology if self.fixed_topology is not None else ctx.A_base).copy()
        np.fill_diagonal(A, 0)
        r, t = self._solve(A, ctx)                                  # line 2
        if r is None:
            # D_max not reachable even with the base topology: best effort, send more
            r = np.full(len(A), self._bounds()[1], dtype=float)
            t = round_time(A, r, ctx.b_in, ctx.b_out, ctx.t_cp, ctx.model_size)
            self.last_stats = {"lp_feasible": False, "lp_solves": 1}
            return A, r
        n_solves = 1
        if self.fixed_topology is not None:
            self.last_stats = {"lp_feasible": True, "lp_solves": n_solves}
            return A, r

        flag, s = True, len(A)
        while True:                                                 # line 3
            T = link_times(A, r, ctx.b_in, ctx.b_out, ctx.t_cp, ctx.model_size)
            T = np.maximum(T, T.T)                                  # undirected link time
            s = int(np.floor(np.sqrt(A.sum()))) if flag else s // 2  # lines 5-8
            s = max(s, 1)
            links = undirected_links(A)
            links.sort(key=lambda e: -T[e])
            E = links[:s]                                           # line 9
            A, r, t, flag = self._adjust(A, E, t, r, ctx)           # line 10
            n_solves += 1
            if not flag and s == 1:                                 # line 11
                break
        self.last_stats = {"lp_feasible": True, "lp_solves": n_solves}
        return A, r

    def _adjust(self, A, E, t, r, ctx):
        """Function AjustCR (lines 13-23)."""
        A2 = A.copy()
        for i, j in E:
            A2[i, j] = A2[j, i] = 0
            if not is_connected(A2):                                # keep lambda_2(L) > 0
                A2[i, j] = A2[j, i] = 1
        r2, t2 = self._solve(A2, ctx)
        if r2 is not None and t2 < t * (1 - 1e-9):
            return A2, r2, t2, True
        return A, r, t, False


# ----------------------------------------------------------------- baselines

def circulant(m, degree):
    """k-regular circulant graph (ring for degree 2, complete for degree m-1)."""
    A = np.zeros((m, m), dtype=int)
    offsets = list(range(1, degree // 2 + 1))
    if degree % 2 == 1:
        if m % 2:
            raise ValueError("odd degree needs an even number of workers")
        offsets.append(m // 2)
    for i in range(m):
        for o in offsets:
            A[i, (i + o) % m] = A[(i + o) % m, i] = 1
    np.fill_diagonal(A, 0)
    return A


def complete(m):
    return np.ones((m, m), dtype=int) - np.eye(m, dtype=int)


class FixedPlanner:
    """Fixed topology and identical, fixed compression ratio.

    Used for the DCD / Choco style baselines (degree 5 / ring) and LCT
    (complete graph, ratio 1). Note: these reuse the element-wise aggregation of
    Eq. (6) instead of the original DCD / Choco-SGD update rules.
    """

    def __init__(self, A, ratio):
        self.A, self.ratio = A, ratio
        self.last_stats = {}

    def plan(self, ctx):
        return self.A.copy(), np.full(len(self.A), self.ratio)


class SAPSPlanner:
    """SAPS-like adaptive peer selection with a fixed, identical compression ratio.

    Every round the topology is a maximum-bandwidth spanning tree over the
    available links, using the pairwise capacity min(b_out_i, b_in_j)
    (a simplification of the peer selection of Tang et al.).
    """

    def __init__(self, ratio):
        self.ratio = ratio
        self.last_stats = {}

    def plan(self, ctx):
        m = len(ctx.A_base)
        cap = np.minimum(ctx.b_out[:, None], ctx.b_in[None, :])
        cap = np.minimum(cap, cap.T) * (ctx.A_base > 0)
        # Prim's algorithm on -capacity
        A = np.zeros((m, m), dtype=int)
        in_tree = np.zeros(m, dtype=bool)
        in_tree[0] = True
        best = cap[0].copy()
        parent = np.zeros(m, dtype=int)
        for _ in range(m - 1):
            cand = np.where(in_tree, -np.inf, best)
            v = int(np.argmax(cand))
            A[v, parent[v]] = A[parent[v], v] = 1
            in_tree[v] = True
            better = cap[v] > best
            parent[better] = v
            best = np.maximum(best, cap[v])
        return A, np.full(m, self.ratio)


def build_planner(name, m, ratio, r_min, tiebreak):
    """name in {coco, dcd, choco, saps, lct, fcr, fnt-ring, fnt-fc}."""
    if name == "coco":
        return CoCoPlanner(r_min=r_min, tiebreak=tiebreak)
    if name == "fcr":
        return CoCoPlanner(fixed_ratio=ratio, tiebreak=tiebreak)
    if name == "fnt-ring":
        return CoCoPlanner(r_min=r_min, tiebreak=tiebreak, fixed_topology=circulant(m, 2))
    if name == "fnt-fc":
        return CoCoPlanner(r_min=r_min, tiebreak=tiebreak, fixed_topology=complete(m))
    if name == "dcd":
        deg = min(5, m - 1)
        return FixedPlanner(circulant(m, deg - (deg % 2) * (m % 2)), ratio)
    if name == "choco":
        return FixedPlanner(circulant(m, 2), ratio)
    if name == "saps":
        return SAPSPlanner(ratio)
    if name == "lct":
        return FixedPlanner(complete(m), 1.0)
    raise ValueError(f"unknown algorithm {name}")
