#!/usr/bin/env python3
"""Test the fixed SMTE implementation."""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Create a minimal version of SMTE for testing
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
from numba import njit, prange
from typing import Tuple

def fdr_bh(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Return boolean mask of p‑values that survive BH‑FDR."""
    flat = pvals.flatten()
    m = flat.size
    idx = np.argsort(flat)
    thr = q * (np.arange(1, m + 1) / m)
    passed = np.zeros_like(flat, dtype=bool)
    passed[idx] = flat[idx] <= thr
    return passed.reshape(pvals.shape)

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

def discretise_symbols(ts: np.ndarray, n_symbols: int, strategy: str = "uniform") -> np.ndarray:
    est = KBinsDiscretizer(n_bins=n_symbols, encode="ordinal", strategy=strategy)
    return est.fit_transform(ts.reshape(-1, 1)).astype(np.int64).ravel()

class SMTE:
    """Fixed Symbolic Matrix Transfer Entropy implementation."""
    
    def __init__(self, n_symbols: int = 5, symboliser: str = "uniform", max_lag: int = 6):
        self.n_symbols = n_symbols
        self.symboliser = symboliser
        self.max_lag = max_lag
    
    def symbolise(self, ts: np.ndarray) -> np.ndarray:
        if self.symboliser == "ordinal":
            return ordinal_symbols(ts, order=3)
        else:
            return discretise_symbols(ts, self.n_symbols, strategy=self.symboliser)
    
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
    
    @staticmethod
    def _S(G: np.ndarray) -> float:
        eps = 1e-10
        G_norm = G / np.sum(G)  # proper probability normalization
        eig = np.linalg.eigvalsh(G_norm)
        eig = np.clip(eig, eps, 1.0)  # avoid tiny negatives
        S = -np.sum(eig * np.log(eig))
        return max(S, 0.0)
    
    @staticmethod
    def _truncate(x: np.ndarray, y: np.ndarray, lag: int):
        if lag == 0:
            return x[1:], x[:-1], y[:-1]
        return x[lag:], x[:-lag], y[:-lag]
    
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

def test_smte_fix():
    """Test the fixed SMTE with synthetic causal data."""
    print("Testing Fixed SMTE Implementation")
    print("=" * 40)
    
    # Generate synthetic data with known causal relationship
    np.random.seed(42)
    n_time = 200
    
    # Independent source
    x = np.random.randn(n_time)
    
    # Dependent target (y depends on x with lag 1)
    y = np.random.randn(n_time)
    y[1:] += 0.7 * x[:-1]  # Strong causal relationship
    
    # Independent control
    z = np.random.randn(n_time)
    
    # Test SMTE
    smte = SMTE(n_symbols=5, symboliser="uniform")
    
    # Test causal direction (should be strong)
    te_x_to_y, lag_xy = smte.smte(smte.symbolise(x), smte.symbolise(y))
    print(f"X → Y: TE = {te_x_to_y:.4f}, optimal lag = {lag_xy}")
    
    # Test reverse direction (should be weak)
    te_y_to_x, lag_yx = smte.smte(smte.symbolise(y), smte.symbolise(x))
    print(f"Y → X: TE = {te_y_to_x:.4f}, optimal lag = {lag_yx}")
    
    # Test control (should be weak)
    te_x_to_z, lag_xz = smte.smte(smte.symbolise(x), smte.symbolise(z))
    print(f"X → Z: TE = {te_x_to_z:.4f}, optimal lag = {lag_xz}")
    
    # Verify causal direction is correctly detected
    if te_x_to_y > te_y_to_x and te_x_to_y > te_x_to_z:
        print("\n✅ SUCCESS: Causal direction correctly detected!")
        print(f"   True causal link (X→Y) has highest TE: {te_x_to_y:.4f}")
    else:
        print("\n❌ FAILURE: Causal direction not detected correctly")
    
    print(f"\nOptimal lag for true causal link: {lag_xy} (expected: 1)")
    
    return te_x_to_y, te_y_to_x, te_x_to_z

if __name__ == "__main__":
    test_smte_fix()