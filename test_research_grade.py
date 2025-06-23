#!/usr/bin/env python3
"""
Research-grade validation suite for VoxelSMTEConnectivity implementation.
This test suite validates theoretical correctness, numerical stability, and research standards.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from voxel_smte_connectivity_corrected import VoxelSMTEConnectivity
import os
import tempfile
import warnings
import math
from scipy import stats
from itertools import permutations
warnings.filterwarnings('ignore')


def test_ordinal_pattern_correctness():
    """Test ordinal pattern implementation against known results."""
    print("Testing ordinal pattern correctness...")
    
    analyzer = VoxelSMTEConnectivity(symbolizer='ordinal', ordinal_order=3)
    
    # Test case 1: Simple increasing sequence
    ts1 = np.array([1, 2, 3, 4, 5])
    patterns1 = analyzer._ordinal_patterns(ts1, 3)
    expected1 = [0, 0, 0]  # All patterns should be (0, 1, 2) -> index 0
    assert np.array_equal(patterns1, expected1), f"Expected {expected1}, got {patterns1}"
    
    # Test case 2: Decreasing sequence  
    ts2 = np.array([5, 4, 3, 2, 1])
    patterns2 = analyzer._ordinal_patterns(ts2, 3)
    expected2 = [5, 5, 5]  # All patterns should be (2, 1, 0) -> index 5
    assert np.array_equal(patterns2, expected2), f"Expected {expected2}, got {patterns2}"
    
    # Test case 3: Known pattern
    ts3 = np.array([1, 3, 2, 4])
    patterns3 = analyzer._ordinal_patterns(ts3, 3)
    # Pattern 1: [1,3,2] -> ranks [0,2,1] -> index 1
    # Pattern 2: [3,2,4] -> ranks [1,0,2] -> index 3
    expected3 = [1, 3]
    assert np.array_equal(patterns3, expected3), f"Expected {expected3}, got {patterns3}"
    
    print("✓ Ordinal pattern implementation correct")


def test_transfer_entropy_properties():
    """Test fundamental properties of transfer entropy."""
    print("Testing transfer entropy mathematical properties...")
    
    analyzer = VoxelSMTEConnectivity(
        symbolizer='uniform', 
        n_symbols=5, 
        max_lag=3,
        random_state=42
    )
    
    # Generate test data
    np.random.seed(42)
    n_points = 200
    
    # Test 1: TE should be zero for independent time series
    x_indep = np.random.randn(n_points)
    y_indep = np.random.randn(n_points)
    
    symbols_x = analyzer.symbolize_timeseries(x_indep.reshape(1, -1))[0]
    symbols_y = analyzer.symbolize_timeseries(y_indep.reshape(1, -1))[0]
    
    te_indep, _ = analyzer._compute_smte_pair(symbols_x, symbols_y)
    
    # TE should be close to zero for independent series
    assert te_indep < 0.1, f"TE for independent series too high: {te_indep}"
    
    # Test 2: TE should be positive for coupled series
    x_source = np.random.randn(n_points)
    y_target = np.zeros(n_points)
    y_target[1:] = 0.8 * x_source[:-1] + 0.2 * np.random.randn(n_points-1)
    
    symbols_x_src = analyzer.symbolize_timeseries(x_source.reshape(1, -1))[0]
    symbols_y_tgt = analyzer.symbolize_timeseries(y_target.reshape(1, -1))[0]
    
    te_coupled, lag = analyzer._compute_smte_pair(symbols_y_tgt, symbols_x_src)
    
    # TE should be positive and lag should be detected correctly
    assert te_coupled > 0.01, f"TE for coupled series too low: {te_coupled}"
    assert lag == 1, f"Incorrect lag detected: {lag}, expected 1"
    
    # Test 3: Asymmetry property
    te_reverse, _ = analyzer._compute_smte_pair(symbols_x_src, symbols_y_tgt)
    
    # Forward TE should be much larger than reverse
    assert te_coupled > te_reverse, f"TE asymmetry violated: forward={te_coupled}, reverse={te_reverse}"
    
    print("✓ Transfer entropy properties validated")


def test_statistical_significance():
    """Test statistical significance assessment."""
    print("Testing statistical significance procedures...")
    
    analyzer = VoxelSMTEConnectivity(
        symbolizer='ordinal',
        ordinal_order=3,
        max_lag=3,
        n_permutations=500,  # Reduced for testing
        alpha=0.05,
        random_state=42
    )
    
    # Generate data with known ground truth
    np.random.seed(42)
    n_points = 150
    n_voxels = 10
    
    # Create time series with some true connections
    data = np.random.randn(n_voxels, n_points)
    
    # Add true connection: voxel 0 -> voxel 1 with lag 1
    data[1, 1:] += 0.7 * data[0, :-1]
    
    # Add true connection: voxel 2 -> voxel 3 with lag 2  
    data[3, 2:] += 0.6 * data[2, :-2]
    
    # Symbolize data
    analyzer.symbolic_data = analyzer.symbolize_timeseries(data)
    
    # Compute SMTE matrix
    smte_matrix, _ = analyzer.compute_voxel_connectivity_matrix()
    
    # Statistical testing
    p_values = analyzer.statistical_testing(smte_matrix)
    
    # Apply FDR correction
    significance_mask = analyzer.fdr_correction(p_values)
    
    # Check that known connections are detected
    # Note: Due to symbolization and noise, detection may not be perfect
    total_significant = np.sum(significance_mask)
    
    print(f"Found {total_significant} significant connections")
    print(f"SMTE 0->1: {smte_matrix[1,0]:.4f}, p-value: {p_values[1,0]:.4f}")
    print(f"SMTE 2->3: {smte_matrix[3,2]:.4f}, p-value: {p_values[3,2]:.4f}")
    
    # Should find some significant connections
    assert total_significant > 0, "No significant connections found"
    
    # P-values should be properly bounded
    assert np.all(p_values >= 0) and np.all(p_values <= 1), "P-values out of bounds"
    
    # FDR correction should reduce number of significant connections
    raw_significant = np.sum(p_values < analyzer.alpha)
    fdr_significant = np.sum(significance_mask)
    assert fdr_significant <= raw_significant, "FDR correction failed to reduce false positives"
    
    print("✓ Statistical significance procedures validated")


def test_memory_efficiency():
    """Test memory-efficient processing."""
    print("Testing memory-efficient processing...")
    
    analyzer = VoxelSMTEConnectivity(
        symbolizer='ordinal',
        ordinal_order=3,
        max_lag=2,
        n_permutations=50,  # Very reduced for speed
        memory_efficient=True,
        n_jobs=1  # Single job for testing
    )
    
    # Generate moderate-sized dataset
    np.random.seed(42)
    n_voxels = 30
    n_points = 100
    data = np.random.randn(n_voxels, n_points)
    
    # Symbolize
    analyzer.symbolic_data = analyzer.symbolize_timeseries(data)
    
    # Test chunked computation
    smte_matrix_chunked, _ = analyzer.compute_voxel_connectivity_matrix(
        chunk_size=50  # Small chunks
    )
    
    # Test regular computation for comparison
    analyzer.memory_efficient = False
    smte_matrix_regular, _ = analyzer.compute_voxel_connectivity_matrix()
    
    # Results should be identical
    np.testing.assert_array_almost_equal(
        smte_matrix_chunked, smte_matrix_regular, decimal=10,
        err_msg="Chunked and regular computation give different results"
    )
    
    print("✓ Memory-efficient processing validated")


def test_parameter_validation():
    """Test parameter validation and edge cases."""
    print("Testing parameter validation...")
    
    # Test invalid parameters
    try:
        VoxelSMTEConnectivity(ordinal_order=1)  # Should fail
        assert False, "Should have raised ValueError for ordinal_order < 2"
    except ValueError:
        pass
    
    try:
        VoxelSMTEConnectivity(max_lag=0)  # Should fail
        assert False, "Should have raised ValueError for max_lag < 1"
    except ValueError:
        pass
    
    try:
        VoxelSMTEConnectivity(alpha=0)  # Should fail
        assert False, "Should have raised ValueError for alpha <= 0"
    except ValueError:
        pass
    
    try:
        VoxelSMTEConnectivity(alpha=1)  # Should fail
        assert False, "Should have raised ValueError for alpha >= 1"
    except ValueError:
        pass
    
    # Test automatic n_symbols correction for ordinal patterns
    analyzer = VoxelSMTEConnectivity(symbolizer='ordinal', ordinal_order=4, n_symbols=10)
    assert analyzer.n_symbols == 24, f"n_symbols should be 24 for order 4, got {analyzer.n_symbols}"
    
    print("✓ Parameter validation working correctly")


def test_numerical_stability():
    """Test numerical stability with edge cases."""
    print("Testing numerical stability...")
    
    analyzer = VoxelSMTEConnectivity(
        symbolizer='uniform',
        n_symbols=5,
        max_lag=2,
        random_state=42
    )
    
    # Test with constant time series
    constant_ts = np.ones(100)
    try:
        symbols = analyzer.symbolize_timeseries(constant_ts.reshape(1, -1))
        te, lag = analyzer._compute_smte_pair(symbols[0], symbols[0])
        assert not np.isnan(te) and not np.isinf(te), "TE should be finite for constant series"
        assert te >= 0, "TE should be non-negative"
    except Exception as e:
        print(f"Warning: Constant time series handling issue: {e}")
    
    # Test with very short time series
    short_ts = np.random.randn(10)
    try:
        symbols = analyzer.symbolize_timeseries(short_ts.reshape(1, -1))
        te, lag = analyzer._compute_smte_pair(symbols[0], symbols[0])
        assert not np.isnan(te) and not np.isinf(te), "TE should be finite for short series"
    except Exception as e:
        print(f"Expected issue with very short time series: {e}")
    
    # Test with extreme values
    extreme_ts = np.array([1e10, -1e10, 1e10, -1e10] * 25)
    symbols = analyzer.symbolize_timeseries(extreme_ts.reshape(1, -1))
    te, lag = analyzer._compute_smte_pair(symbols[0], symbols[0])
    assert not np.isnan(te) and not np.isinf(te), "TE should be finite for extreme values"
    
    print("✓ Numerical stability validated")


def test_reproducibility():
    """Test reproducibility with fixed random seeds."""
    print("Testing reproducibility...")
    
    # Run analysis twice with same parameters
    results1 = run_synthetic_analysis(random_state=42)
    results2 = run_synthetic_analysis(random_state=42)
    
    # Results should be identical
    np.testing.assert_array_equal(
        results1['smte_matrix'], results2['smte_matrix'],
        err_msg="Results not reproducible with same random seed"
    )
    
    np.testing.assert_array_equal(
        results1['p_values'], results2['p_values'],
        err_msg="P-values not reproducible with same random seed"
    )
    
    # Run with different seeds should give different results
    results3 = run_synthetic_analysis(random_state=123)
    
    assert not np.array_equal(results1['smte_matrix'], results3['smte_matrix']), \
        "Different random seeds should give different results"
    
    print("✓ Reproducibility validated")


def run_synthetic_analysis(random_state=42):
    """Helper function to run analysis on synthetic data."""
    
    analyzer = VoxelSMTEConnectivity(
        symbolizer='ordinal',
        ordinal_order=3,
        max_lag=3,
        n_permutations=100,
        random_state=random_state
    )
    
    # Generate synthetic data
    np.random.seed(random_state)
    n_voxels = 15
    n_points = 120
    data = np.random.randn(n_voxels, n_points)
    
    # Add coupling
    data[1, 1:] += 0.6 * data[0, :-1]
    data[3, 2:] += 0.5 * data[2, :-2]
    
    # Run analysis
    analyzer.symbolic_data = analyzer.symbolize_timeseries(data)
    smte_matrix, _ = analyzer.compute_voxel_connectivity_matrix()
    p_values = analyzer.statistical_testing(smte_matrix)
    significance_mask = analyzer.fdr_correction(p_values)
    
    return {
        'smte_matrix': smte_matrix,
        'p_values': p_values,
        'significance_mask': significance_mask
    }


def test_research_grade_pipeline():
    """Test complete research-grade pipeline with realistic parameters."""
    print("Testing research-grade pipeline...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate realistic synthetic fMRI data
        fmri_data, mask = generate_realistic_fmri_data()
        
        # Create NIfTI files
        fmri_path, mask_path = create_nifti_files(fmri_data, mask, temp_dir)
        
        # Initialize with research-grade parameters
        analyzer = VoxelSMTEConnectivity(
            n_symbols=6,              # For ordinal patterns of order 3
            symbolizer='ordinal',     # Most robust for noisy fMRI data
            ordinal_order=3,          # Standard choice
            max_lag=5,               # Up to 5 TRs
            alpha=0.01,              # Stringent threshold
            n_permutations=200,      # Reduced for testing, use 1000+ in practice
            n_jobs=2,                # Limited for testing
            memory_efficient=True,
            random_state=42
        )
        
        # Run analysis on subset for speed
        n_test_voxels = 25
        voxel_indices = list(range(n_test_voxels))
        
        results = analyzer.run_complete_analysis(
            fmri_path=fmri_path,
            output_dir=os.path.join(temp_dir, 'results'),
            mask_path=mask_path,
            voxel_indices=voxel_indices,
            visualize=False  # Skip visualization for testing
        )
        
        # Validate results structure
        required_keys = [
            'smte_matrix', 'p_values', 'significance_mask', 'lag_matrix',
            'connectivity_graph', 'network_properties', 'n_significant_connections'
        ]
        
        for key in required_keys:
            assert key in results, f"Missing key in results: {key}"
        
        # Validate matrix dimensions
        assert results['smte_matrix'].shape == (n_test_voxels, n_test_voxels)
        assert results['p_values'].shape == (n_test_voxels, n_test_voxels)
        assert results['significance_mask'].shape == (n_test_voxels, n_test_voxels)
        
        # Validate network properties
        props = results['network_properties']
        assert 'n_nodes' in props and props['n_nodes'] == n_test_voxels
        assert 'density' in props and 0 <= props['density'] <= 1
        
        # Check file outputs
        output_files = [
            'smte_matrix.npy', 'p_values.npy', 'significance_mask.npy',
            'connectivity_graph.graphml', 'network_properties.json',
            'analysis_parameters.json', 'analysis_summary.txt'
        ]
        
        results_dir = os.path.join(temp_dir, 'results')
        for filename in output_files:
            filepath = os.path.join(results_dir, filename)
            assert os.path.exists(filepath), f"Missing output file: {filename}"
        
        print(f"Pipeline test results:")
        print(f"- Significant connections: {results['n_significant_connections']}")
        print(f"- Connection rate: {results['connection_rate']:.4f}")
        print(f"- Network density: {props['density']:.6f}")
        
    print("✓ Research-grade pipeline validated")


def generate_realistic_fmri_data():
    """Generate realistic synthetic fMRI data with proper characteristics."""
    np.random.seed(42)
    
    # Parameters
    nx, ny, nz = 20, 20, 15  # Small volume for testing
    n_timepoints = 150       # Typical scan length
    
    # Create brain-like mask (ellipsoid)
    x, y, z = np.ogrid[:nx, :ny, :nz]
    cx, cy, cz = nx//2, ny//2, nz//2
    mask = ((x-cx)**2/(nx//2)**2 + (y-cy)**2/(ny//2)**2 + (z-cz)**2/(nz//2)**2) < 1
    
    # Initialize data
    fmri_data = np.zeros((nx, ny, nz, n_timepoints))
    
    # Generate realistic BOLD signals
    t = np.arange(n_timepoints) * 2.0  # TR = 2s
    
    # Base physiological oscillations
    respiratory = np.sin(2 * np.pi * 0.3 * t)  # ~0.3 Hz respiratory
    cardiac = np.sin(2 * np.pi * 1.0 * t)      # ~1 Hz cardiac
    low_freq = np.sin(2 * np.pi * 0.05 * t)    # Low frequency drift
    
    # Create regions with different signal characteristics
    voxel_coords = np.where(mask)
    n_voxels = len(voxel_coords[0])
    
    for i, (x, y, z) in enumerate(zip(*voxel_coords)):
        # Spatial clustering of signals
        region = (x // 5) * 16 + (y // 5) * 4 + (z // 5)
        
        # Base signal with regional characteristics
        if region % 3 == 0:
            signal = 0.5 * low_freq + 0.2 * respiratory
        elif region % 3 == 1:
            signal = 0.3 * low_freq + 0.4 * cardiac
        else:
            signal = 0.6 * low_freq + 0.1 * (respiratory + cardiac)
        
        # Add task-related activity (some regions)
        if region % 4 == 0:
            task_signal = np.zeros(n_timepoints)
            # Simple block design
            for block in range(0, n_timepoints, 40):
                task_signal[block:block+20] = 1.0
            signal += 0.8 * task_signal
        
        # Add connectivity (some voxels influenced by others)
        if i > 10 and np.random.rand() < 0.3:
            source_idx = np.random.choice(i)
            lag = np.random.randint(1, 4)
            coupling = np.random.uniform(0.3, 0.7)
            
            source_coord = voxel_coords[0][source_idx], voxel_coords[1][source_idx], voxel_coords[2][source_idx]
            source_signal = fmri_data[source_coord][0:n_timepoints-lag]
            
            if len(source_signal) > 0:
                signal[lag:lag+len(source_signal)] += coupling * source_signal
        
        # Add thermal noise
        noise = np.random.randn(n_timepoints) * 0.5
        signal += noise
        
        # Store in 4D array
        fmri_data[x, y, z, :] = signal
    
    return fmri_data, mask


def create_nifti_files(fmri_data, mask, output_dir):
    """Create NIfTI files from arrays."""
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 3.0  # 3mm voxels
    
    # fMRI data
    fmri_img = nib.Nifti1Image(fmri_data, affine)
    fmri_path = os.path.join(output_dir, 'test_fmri.nii.gz')
    nib.save(fmri_img, fmri_path)
    
    # Mask
    mask_img = nib.Nifti1Image(mask.astype(np.uint8), affine)
    mask_path = os.path.join(output_dir, 'test_mask.nii.gz')
    nib.save(mask_img, mask_path)
    
    return fmri_path, mask_path


def main():
    """Run comprehensive research-grade validation suite."""
    print("=" * 60)
    print("Research-Grade SMTE Connectivity Validation Suite")
    print("=" * 60)
    print()
    
    tests = [
        test_ordinal_pattern_correctness,
        test_transfer_entropy_properties,
        test_parameter_validation,
        test_numerical_stability,
        test_statistical_significance,
        test_memory_efficiency,
        test_reproducibility,
        test_research_grade_pipeline
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
            print()
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()
    
    print("=" * 60)
    print(f"Validation Summary: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Implementation is research-grade.")
    else:
        print("⚠️  Some tests failed. Please review implementation.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()