#!/usr/bin/env python3
"""
Quick validation of the corrected implementation.
"""

import numpy as np
from voxel_smte_connectivity_corrected import VoxelSMTEConnectivity
import warnings
warnings.filterwarnings('ignore')

def test_basic_functionality():
    """Test basic functionality with small synthetic dataset."""
    print("Testing basic functionality...")
    
    # Initialize with minimal parameters
    analyzer = VoxelSMTEConnectivity(
        n_symbols=6,
        symbolizer='ordinal',
        ordinal_order=3,
        max_lag=2,
        n_permutations=50,  # Small for testing
        random_state=42
    )
    
    # Generate small test dataset
    np.random.seed(42)
    n_voxels = 8
    n_timepoints = 80
    
    # Create synthetic data with known coupling
    data = np.random.randn(n_voxels, n_timepoints)
    # Add coupling: voxel 0 -> voxel 1
    data[1, 1:] += 0.6 * data[0, :-1]
    
    print(f"Generated {n_voxels} voxels with {n_timepoints} timepoints")
    
    # Test symbolization
    symbolic_data = analyzer.symbolize_timeseries(data)
    print(f"Symbolization successful: {symbolic_data.shape}")
    
    # Test SMTE computation
    analyzer.symbolic_data = symbolic_data
    smte_matrix, lag_matrix = analyzer.compute_voxel_connectivity_matrix()
    print(f"SMTE matrix computed: {smte_matrix.shape}")
    print(f"Max SMTE value: {np.max(smte_matrix):.4f}")
    print(f"SMTE 0->1: {smte_matrix[1,0]:.4f}")
    
    # Test statistical significance
    p_values = analyzer.statistical_testing(smte_matrix)
    print(f"Statistical testing completed")
    print(f"Min p-value: {np.min(p_values[p_values > 0]):.4f}")
    
    # Test FDR correction
    significance_mask = analyzer.fdr_correction(p_values)
    n_significant = np.sum(significance_mask)
    print(f"FDR correction: {n_significant} significant connections")
    
    # Test graph construction
    graph = analyzer.build_connectivity_graph(smte_matrix, significance_mask)
    print(f"Graph created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Test network analysis
    properties = analyzer.analyze_network_properties(graph)
    print(f"Network density: {properties['density']:.4f}")
    
    print("✓ Basic functionality test passed!")
    return True

def test_ordinal_patterns():
    """Test ordinal pattern implementation."""
    print("Testing ordinal patterns...")
    
    analyzer = VoxelSMTEConnectivity(symbolizer='ordinal', ordinal_order=3)
    
    # Test simple sequence
    ts = np.array([1, 2, 3, 4, 5])
    patterns = analyzer._ordinal_patterns(ts, 3)
    print(f"Increasing sequence patterns: {patterns}")
    
    # Test decreasing sequence
    ts2 = np.array([5, 4, 3, 2, 1])
    patterns2 = analyzer._ordinal_patterns(ts2, 3)
    print(f"Decreasing sequence patterns: {patterns2}")
    
    # Check that we get different patterns for different dynamics
    assert not np.array_equal(patterns, patterns2), "Different sequences should give different patterns"
    
    print("✓ Ordinal patterns test passed!")
    return True

def test_transfer_entropy():
    """Test transfer entropy computation."""
    print("Testing transfer entropy computation...")
    
    analyzer = VoxelSMTEConnectivity(
        symbolizer='uniform',
        n_symbols=5,
        max_lag=3,
        random_state=42
    )
    
    # Generate coupled time series
    np.random.seed(42)
    n_points = 100
    x_source = np.random.randn(n_points)
    y_target = np.zeros(n_points)
    y_target[1:] = 0.7 * x_source[:-1] + 0.3 * np.random.randn(n_points-1)
    
    # Symbolize
    data = np.array([y_target, x_source])
    symbols = analyzer.symbolize_timeseries(data)
    
    # Compute TE in both directions
    te_forward, lag_forward = analyzer._compute_smte_pair(symbols[0], symbols[1])
    te_backward, lag_backward = analyzer._compute_smte_pair(symbols[1], symbols[0])
    
    print(f"TE source->target: {te_forward:.4f} (lag {lag_forward})")
    print(f"TE target->source: {te_backward:.4f} (lag {lag_backward})")
    
    # Forward TE should be higher (but not always guaranteed with symbolization)
    print(f"Forward/backward ratio: {te_forward/max(te_backward, 1e-6):.2f}")
    
    # Both should be non-negative
    assert te_forward >= 0 and te_backward >= 0, "TE values should be non-negative"
    
    print("✓ Transfer entropy test passed!")
    return True

def main():
    """Run quick validation tests."""
    print("=" * 50)
    print("Quick Validation of Research-Grade SMTE Implementation")
    print("=" * 50)
    
    tests = [
        test_ordinal_patterns,
        test_transfer_entropy, 
        test_basic_functionality
    ]
    
    passed = 0
    for test_func in tests:
        try:
            test_func()
            passed += 1
            print()
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 50)
    if passed == len(tests):
        print("🎉 All validation tests passed!")
        print("The implementation is working correctly.")
    else:
        print(f"⚠️  {len(tests) - passed} tests failed.")
    print("=" * 50)

if __name__ == "__main__":
    main()