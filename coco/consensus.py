"""Consensus distance measurement / estimation (Sections 2.3 and 4.2) and the
adaptive threshold D_max (Eq. 28)."""
import numpy as np
import torch


@torch.no_grad()
def measured_distance(x_i, x_j, z_j):
    """Eq. (23): estimate ||x_i - x_j|| from the parameters of x_j received through mask z_j."""
    d = x_i.numel()
    nnz = z_j.sum().item()
    diff = (x_i - x_j)[z_j]
    return float(torch.sqrt(d * diff.pow(2).sum() / max(nnz, 1)).item())


@torch.no_grad()
def true_consensus_distance(X):
    """Eqs. (2)-(3): mean_i ||x_bar - x_i||, only available in simulation."""
    x_bar = X.mean(dim=0, keepdim=True)
    return float((X - x_bar).norm(dim=1).mean().item())


def floyd_warshall(W):
    """All-pairs shortest paths; W[i, j] = edge length (np.inf if no edge)."""
    D = W.copy()
    np.fill_diagonal(D, 0.0)
    for p in range(len(D)):
        D = np.minimum(D, D[:, p:p + 1] + D[p:p + 1, :])
    return D


class ConsensusEstimator:
    """Coordinator-side matrix of pairwise consensus distances D_ij.

    Linked pairs use the distances reported by workers (Eq. 23). Unlinked pairs
    use the triangle-inequality bound min_p (D_ip + D_pj) computed iteratively
    via Floyd-Warshall (Eqs. 24-25), smoothed with an EMA (Eq. 26).
    """

    def __init__(self, m, beta1=0.5):
        self.m, self.beta1 = m, beta1
        self.D = np.zeros((m, m))

    def update(self, A, reported):
        """A: topology of the round; reported[i, j] = D_ij measured by worker i (for a_ij = 1)."""
        linked = A > 0
        # links are undirected: average the two directional measurements
        sym = np.where(linked & linked.T, 0.5 * (reported + reported.T), reported)
        W = np.where(linked, sym, np.inf)
        D_hat = floyd_warshall(W)
        D_hat = np.where(np.isfinite(D_hat), D_hat, self.D)
        new = np.where(linked, sym, (1 - self.beta1) * self.D + self.beta1 * D_hat)
        np.fill_diagonal(new, 0.0)
        self.D = new
        return self.D

    def average(self):
        """Estimated average consensus distance, D_i ~ (1/m) sum_j D_ij."""
        return float(self.D.sum() / self.m ** 2)


class DmaxTracker:
    """Eq. (28): EMA of the average local-update norm (1/m) sum_i ||g_i||."""

    def __init__(self, beta2=0.1, scale=1.0):
        self.beta2, self.scale = beta2, scale
        self.value = None

    def update(self, update_norms):
        g = float(np.mean(update_norms))
        self.value = g if self.value is None else (1 - self.beta2) * self.value + self.beta2 * g
        return self.get()

    def get(self):
        return 0.0 if self.value is None else self.scale * self.value
