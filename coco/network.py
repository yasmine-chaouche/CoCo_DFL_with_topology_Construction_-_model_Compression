"""Simulated heterogeneous / dynamic edge network (Sections 2.1, 4.1, 5.2, 5.5).

Bandwidths are in Mb/s, model sizes in Mb, times in seconds.
"""
import numpy as np


class NetworkSim:
    """Per-round inbound/outbound bandwidth and computation time of each worker.

    modes:
      heter_dynamic : different across workers and varying over rounds (default in the paper)
      heter_static  : evenly spread across workers, fixed over time
      homo_static   : identical for all workers, fixed over time
    """

    def __init__(self, m, mode, rng, in_range=(1.0, 20.0), out_range=(0.5, 10.0),
                 comp_range=(2.5, 7.5), volatility=0.3):
        self.m, self.mode, self.rng = m, mode, rng
        self.in_range, self.out_range, self.comp_range = in_range, out_range, comp_range
        self.volatility = volatility
        if mode == "heter_dynamic":
            self.b_in = rng.uniform(*in_range, size=m)
            self.b_out = rng.uniform(*out_range, size=m)
        elif mode == "heter_static":
            perm = rng.permutation(m)
            self.b_in = np.linspace(*in_range, m)
            self.b_out = np.linspace(*out_range, m)
            self.cp = np.linspace(*comp_range, m)[perm]
        elif mode == "homo_static":
            self.b_in = np.full(m, np.mean(in_range))
            self.b_out = np.full(m, np.mean(out_range))
            self.cp = np.full(m, np.mean(comp_range))
        else:
            raise ValueError(f"unknown network mode {mode}")

    def swap(self, i, j):
        """Exchange the bandwidth configuration of two workers (Fig. 13b)."""
        for arr in (self.b_in, self.b_out):
            arr[[i, j]] = arr[[j, i]]

    def step(self):
        """Returns (b_in, b_out, t_cp) for the next round; t_cp = tau * c_i (Eq. 15)."""
        if self.mode == "heter_dynamic":
            walk = lambda v, lo_hi: np.clip(
                v * np.exp(self.rng.normal(0, self.volatility, self.m)), *lo_hi)
            self.b_in = walk(self.b_in, self.in_range)
            self.b_out = walk(self.b_out, self.out_range)
            cp = self.rng.uniform(*self.comp_range, size=self.m)
        else:
            cp = self.cp
        return self.b_in.copy(), self.b_out.copy(), np.array(cp, dtype=float)


def degrees(A):
    return A.sum(axis=1)


def link_bandwidth(A, b_in, b_out):
    """Eq. (1): b_ij = min(b_out_i / |N_i|, b_in_j / |N_j|) for every link of A (0 elsewhere)."""
    deg = np.maximum(degrees(A), 1)
    out_share = (b_out / deg)[:, None]
    in_share = (b_in / deg)[None, :]
    return np.where(A > 0, np.minimum(out_share, in_share), 0.0)


def slowest_link_factor(A, b_in, b_out, model_size):
    """beta_i = B / min_{j in N_i} b_ij, so that worker i's communication time is r_i * beta_i (Eq. 16)."""
    bw = link_bandwidth(A, b_in, b_out)
    bw = np.where(A > 0, bw, np.inf)
    return model_size / bw.min(axis=1)


def link_times(A, r, b_in, b_out, t_cp, model_size):
    """Eq. (17): t_ij = t_cp_i + r_i * B / b_ij for every link (0 elsewhere)."""
    bw = link_bandwidth(A, b_in, b_out)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = t_cp[:, None] + (r * model_size)[:, None] / bw
    return np.where(A > 0, t, 0.0)


def round_time(A, r, b_in, b_out, t_cp, model_size):
    """Eq. (18): round time = slowest link."""
    return float(link_times(A, r, b_in, b_out, t_cp, model_size).max())


def is_connected(A):
    m = len(A)
    seen = np.zeros(m, dtype=bool)
    stack = [0]
    seen[0] = True
    while stack:
        u = stack.pop()
        for v in np.nonzero(A[u])[0]:
            if not seen[v]:
                seen[v] = True
                stack.append(v)
    return bool(seen.all())


def algebraic_connectivity(A):
    """lambda_2 of the Laplacian (Eq. 19b)."""
    L = np.diag(degrees(A)) - A
    return float(np.sort(np.linalg.eigvalsh(L))[1])
