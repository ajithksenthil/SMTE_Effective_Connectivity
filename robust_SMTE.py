import os
import math
import warnings
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import networkx as nx
import nibabel as nib
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
from numba import njit, prange

###############################################################################
# Research‑grade Symbolic Matrix Transfer Entropy (SMTE) toolbox
# Canvas version – bug‑fixed (non‑negative SMTE) – 2025‑04‑30
###############################################################################

__all__ = ["SMTE"]

###############################################################################
# Benjamini–Hochberg FDR helper
###############################################################################

def fdr_bh(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Return boolean mask of p‑values that survive BH‑FDR."""
    flat = pvals.flatten()
    m = flat.size
    idx = np.argsort(flat)
    thr = q * (np.arange(1, m + 1) / m)
    passed = np.zeros_like(flat, dtype=bool)
    passed[idx] = flat[idx] <= thr
    return passed.reshape(pvals.shape)

###############################################################################
# Symbolisation functions
###############################################################################

@njit
def _ordinal_pattern_index(window: np.ndarray) -> int:
    """Lehmer code (ordinal pattern index) for a 1‑d window."""
    rank = window.argsort()
    code = 0
    fact = 1
    for i in range(len(rank) - 1, -1, -1):
        code += rank[i] * fact
        fact *= (len(rank) - i)
    return code


def ordinal_symbols(ts: np.ndarray, order: int = 3) -> np.ndarray:
    if len(ts) < order:
        raise ValueError("time‑series shorter than ordinal order")
    N = len(ts) - order + 1
    symbols = np.empty(N, dtype=np.int64)
    for i in range(N):
        symbols[i] = _ordinal_pattern_index(ts[i : i + order])
    return symbols


def vq_symbols(ts: np.ndarray, n_symbols: int) -> np.ndarray:
    labs = KMeans(n_clusters=n_symbols, n_init="auto", random_state=0).fit_predict(
        ts.reshape(-1, 1)
    )
    return labs.astype(np.int64)


def discretise_symbols(
    ts: np.ndarray, n_symbols: int, strategy: str = "uniform"
) -> np.ndarray:
    est = KBinsDiscretizer(n_bins=n_symbols, encode="ordinal", strategy=strategy)
    return est.fit_transform(ts.reshape(-1, 1)).astype(np.int64).ravel()

###############################################################################
# Main SMTE class
###############################################################################


class SMTE:
    """Symbolic Matrix Transfer Entropy analysis for fMRI or generic data."""

    def __init__(
        self,
        n_symbols: int = 5,
        TR: float = 2.0,
        symboliser: str = "uniform",
        alpha: float = 0.05,
        max_lag: Optional[int] = None,
    ):
        self.n_symbols = n_symbols
        self.TR = TR
        self.symboliser = symboliser  # 'uniform', 'quantile', 'ordinal', 'vq'
        self.alpha = alpha
        self.max_lag = max_lag or 6  # default ≈12 s for TR=2

    # ------------------------------------------------------------------
    # Symbolisation dispatcher
    # ------------------------------------------------------------------
    def symbolise(self, ts: np.ndarray) -> np.ndarray:
        if self.symboliser == "ordinal":
            return ordinal_symbols(ts, order=3)
        elif self.symboliser == "vq":
            return vq_symbols(ts, self.n_symbols)
        else:
            return discretise_symbols(ts, self.n_symbols, strategy=self.symboliser)

    # ------------------------------------------------------------------
    # Gram matrix (Numba‑accelerated)
    # ------------------------------------------------------------------
    @staticmethod
    @njit(parallel=True)
    def _gram(symbols: np.ndarray) -> np.ndarray:
        N = symbols.size
        G = np.empty((N, N), dtype=np.float64)
        for i in prange(N):
            for j in range(N):
                G[i, j] = 1.0 if symbols[i] == symbols[j] else 0.0
        eps = 1e-10
        G_norm = G + eps
        return G_norm / np.sum(G_norm)

    # ------------------------------------------------------------------
    # Truncation helper
    # ------------------------------------------------------------------
    @staticmethod
    def _truncate(x: np.ndarray, y: np.ndarray, lag: int):
        if lag == 0:
            return x[1:], x[:-1], y[:-1]
        return x[lag:], x[:-lag], y[:-lag]

    # ------------------------------------------------------------------
    # Von‑Neumann entropy (α = 2) – guaranteed non‑negative
    # ------------------------------------------------------------------
    @staticmethod
    def _S(G: np.ndarray) -> float:
        eps = 1e-10
        G_norm = G / np.sum(G)  # proper probability normalization
        eig = np.linalg.eigvalsh(G_norm)
        eig = np.clip(eig, eps, 1.0)  # avoid tiny negatives
        S = -np.sum(eig * np.log(eig))
        return max(S, 0.0)

    # ------------------------------------------------------------------
    # Pairwise SMTE
    # ------------------------------------------------------------------
    def smte(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
        best, best_lag = -np.inf, 0
        for lag in range(self.max_lag + 1):
            x_now, x_past, y_past = self._truncate(x, y, lag)
            
            # Use histogram-based joint entropy for computational efficiency
            n_bins = min(self.n_symbols, len(np.unique(x_now)))
            
            # Joint histogram H(X_t, X_{t-1})
            hist_x_xpast, _ = np.histogramdd([x_now, x_past], bins=n_bins)
            hist_x_xpast = hist_x_xpast + 1e-10  # regularization
            prob_x_xpast = hist_x_xpast / np.sum(hist_x_xpast)
            H_x_xpast = -np.sum(prob_x_xpast * np.log(prob_x_xpast + 1e-10))
            
            # Joint histogram H(X_t, X_{t-1}, Y_{t-1})
            hist_x_xpast_ypast, _ = np.histogramdd([x_now, x_past, y_past], bins=n_bins)
            hist_x_xpast_ypast = hist_x_xpast_ypast + 1e-10
            prob_x_xpast_ypast = hist_x_xpast_ypast / np.sum(hist_x_xpast_ypast)
            H_x_xpast_ypast = -np.sum(prob_x_xpast_ypast * np.log(prob_x_xpast_ypast + 1e-10))
            
            # Individual histograms
            hist_xpast, _ = np.histogram(x_past, bins=n_bins)
            hist_xpast = hist_xpast + 1e-10
            prob_xpast = hist_xpast / np.sum(hist_xpast)
            H_xpast = -np.sum(prob_xpast * np.log(prob_xpast + 1e-10))
            
            hist_xpast_ypast, _ = np.histogramdd([x_past, y_past], bins=n_bins)
            hist_xpast_ypast = hist_xpast_ypast + 1e-10
            prob_xpast_ypast = hist_xpast_ypast / np.sum(hist_xpast_ypast)
            H_xpast_ypast = -np.sum(prob_xpast_ypast * np.log(prob_xpast_ypast + 1e-10))
            
            # Transfer entropy: TE = H(X_t|X_{t-1}) - H(X_t|X_{t-1}, Y_{t-1})
            val = H_x_xpast - H_xpast - H_x_xpast_ypast + H_xpast_ypast
            val = max(val, 0.0)  # enforce theoretical ≥0
            
            if val > best:
                best, best_lag = val, lag
        return best, best_lag

    # ------------------------------------------------------------------
    # Surrogate test (circular shift)
    # ------------------------------------------------------------------
    def smte_surrogate_p(
        self, x: np.ndarray, y: np.ndarray, n_surr: int = 500
    ) -> Tuple[float, float]:
        orig, _ = self.smte(x, y)
        null = np.empty(n_surr)
        for k in range(n_surr):
            shift = np.random.randint(1, len(y) - 1)
            null[k], _ = self.smte(x, np.roll(y, shift))
        p = (np.sum(null >= orig) + 1) / (n_surr + 1)
        return orig, p

    # ------------------------------------------------------------------
    # Full SMTE network with BH‑FDR + diagnostics
    # ------------------------------------------------------------------
    def smte_network(self, data: np.ndarray):
        n_roi = data.shape[0]
        symbols = [self.symbolise(ts) for ts in data]

        S = np.zeros((n_roi, n_roi))
        P = np.ones((n_roi, n_roi))
        L = np.zeros((n_roi, n_roi), dtype=int)

        for i in range(n_roi):
            for j in range(n_roi):
                if i == j:
                    continue
                smte_val, p_val = self.smte_surrogate_p(symbols[i], symbols[j])
                S[i, j] = smte_val
                P[i, j] = p_val
                _, L[i, j] = self.smte(symbols[i], symbols[j])

        # Diagnostics
        print("Smallest raw p‑value:", P.min())

        mask = fdr_bh(P, q=self.alpha)
        print("Significant edges (BH‑FDR):")
        edges = [
            (i, j, S[i, j], P[i, j]) for i in range(n_roi) for j in range(n_roi) if mask[i, j]
        ]
        if not edges:
            print("  None survived q=", self.alpha)
        else:
            for i, j, s, p in sorted(edges, key=lambda t: t[3]):
                print(f"  {i} → {j} | SMTE={s:.3f} | p={p:.4g}")

        return S, P, L


###############################################################################
# Quick synthetic demo when run as script
###############################################################################

if __name__ == "__main__":
    np.random.seed(0)
    n_roi, T = 5, 300
    data = np.random.randn(n_roi, T)
    # Inject simple 0 → 1 causal link (lag 1)
    data[1, 1:] += 0.6 * data[0, :-1]

    smte = SMTE(n_symbols=5, symboliser="ordinal", alpha=0.1)
    S, P, L = smte.smte_network(data)
###############################################################################
# Synthetic data utilities & validation
###############################################################################

from scipy.signal import fftconvolve
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

DEFAULT_HRF = np.exp(-np.arange(0, 30) / 4.0)  # simple exponential HRF


def _apply_hrf(ts: np.ndarray, hrf: np.ndarray) -> np.ndarray:
    """Convolve each ROI with an HRF in the *forward* direction."""
    out = np.zeros_like(ts)
    for i in range(ts.shape[0]):
        out[i] = fftconvolve(ts[i], hrf, mode="same")
    return out


def generate_synthetic_data(
    n_roi: int,
    n_time: int,
    causal_spec: List[Tuple[int, int, int, float]],
    snr_db: float = 0.0,
    hrf: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic multivariate time‑series with known causal links.

    Parameters
    ----------
    n_roi, n_time : int
        Number of regions and time‑points.
    causal_spec : list of tuples
        Each tuple = (source, target, lag, strength).  `strength` is *post‑noise*.
    snr_db : float
        Desired signal‑to‑noise ratio in decibels. 0 dB ==> signal power == noise power.
    hrf : ndarray | None
        If provided, each ROI is convolved with this HRF to emulate BOLD.
    """
    rng = np.random.default_rng(seed)
    base_noise = rng.standard_normal((n_roi, n_time))

    signal = np.zeros_like(base_noise)
    for src, tgt, lag, g in causal_spec:
        if lag >= n_time:
            raise ValueError("lag longer than time‑series")
        signal[tgt, lag:] += g * base_noise[src, :-lag]

    # scale noise to achieve requested SNR
    sig_pow = np.var(signal)
    if sig_pow == 0:
        sig_pow = 1e-12
    noise_pow_target = sig_pow / (10 ** (snr_db / 10))
    noise_scale = math.sqrt(noise_pow_target)
    data = signal + noise_scale * base_noise

    if hrf is not None:
        data = _apply_hrf(data, hrf)

    # ground‑truth adjacency (directed)
    gt = np.zeros((n_roi, n_roi))
    for src, tgt, _, g in causal_spec:
        gt[tgt, src] = g
    return data, gt


# -------------------------------------------------------------------------
# Evaluation helpers
# -------------------------------------------------------------------------

def evaluate_prediction(S: np.ndarray, P: np.ndarray, gt: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """Return standard metrics comparing SMTE edges to ground truth."""
    pred = (P < alpha).astype(int)
    y_true = gt.flatten() > 0
    y_pred = pred.flatten() > 0

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_true, (1 - P.flatten())) if (~np.isnan(P)).any() else 0.5
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm,
    }


def sweep_snr_test(
    smte_obj: "SMTE",
    n_roi: int = 6,
    n_time: int = 400,
    snr_grid: Tuple[float, ...] = (0.0, 3.0, 6.0, 10.0),
    n_rep: int = 5,
):
    """Run Monte‑Carlo sweep over SNRs and print averaged metrics."""
    causal = [(0, 1, 1, 0.8), (2, 3, 2, 0.6), (4, 5, 1, 0.7)]
    for snr in snr_grid:
        mets = []
        for r in range(n_rep):
            data, gt = generate_synthetic_data(n_roi, n_time, causal, snr_db=snr, seed=100 + r)
            S, P, _ = smte_obj.smte_network(data)
            mets.append(evaluate_prediction(S, P, gt, alpha=smte_obj.alpha))
        # average metrics
        keys = mets[0].keys()
        avg = {k: np.mean([m[k] for m in mets]) for k in keys}
        print(f"SNR={snr:>4} dB", {k: (float(v) if isinstance(v, np.generic) else v) for k, v in avg.items() if k != "confusion_matrix"})
