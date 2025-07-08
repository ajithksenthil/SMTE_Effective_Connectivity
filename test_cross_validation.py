#!/usr/bin/env python3
"""Quick test for cross-validation parameter optimization."""

import numpy as np
from adaptive_smte_v1 import AdaptiveSMTE


def test_cross_validation_mode():
    print("\n🚀 QUICK TEST: CROSS-VALIDATION PARAMETER OPTIMIZATION")

    np.random.seed(0)
    n_regions = 5
    n_timepoints = 90
    data = np.random.randn(n_regions, n_timepoints)
    ground_truth = np.zeros((n_regions, n_regions))
    data[1, 2:] += 0.5 * data[0, :-2]
    ground_truth[1, 0] = 0.5

    smte = AdaptiveSMTE(adaptive_mode='cross_validation', quick_optimization=True, n_permutations=10)
    connectivity, _, info = smte.compute_adaptive_connectivity(data, ground_truth)

    print(f"Best params: {info.get('applied_params')}")
    print(f"Connectivity[1,0]: {connectivity[1,0]:.4f}")
    assert info.get('applied_params') is not None
    assert connectivity.shape == (n_regions, n_regions)


if __name__ == "__main__":
    test_cross_validation_mode()
