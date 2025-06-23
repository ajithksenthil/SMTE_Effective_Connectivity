#!/usr/bin/env python3
"""
Test script for VoxelSMTEConnectivity implementation.
This script demonstrates the complete pipeline with synthetic fMRI data.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from voxel_smte_connectivity import VoxelSMTEConnectivity
import os
import tempfile
import warnings
warnings.filterwarnings('ignore')


def generate_synthetic_fmri_data(n_voxels: int = 100, 
                                n_timepoints: int = 200,
                                tr: float = 2.0,
                                noise_level: float = 0.3,
                                seed: int = 42) -> tuple:
    """
    Generate synthetic 4D fMRI data with known connectivity patterns.
    
    Parameters:
    -----------
    n_voxels : int
        Number of voxels per dimension (creates n_voxels^3 total voxels)
    n_timepoints : int
        Number of time points
    tr : float
        Repetition time in seconds
    noise_level : float
        Noise level (0-1)
    seed : int
        Random seed
        
    Returns:
    --------
    tuple
        (fmri_data, mask, connectivity_ground_truth)
    """
    np.random.seed(seed)
    
    # Create 3D spatial grid
    dim = int(np.ceil(np.cbrt(n_voxels)))
    fmri_data = np.zeros((dim, dim, dim, n_timepoints))
    
    # Generate base time series with different frequencies
    t = np.arange(n_timepoints) * tr
    base_signals = []
    
    # Signal 1: Low frequency oscillation
    base_signals.append(np.sin(2 * np.pi * 0.05 * t))
    
    # Signal 2: Higher frequency
    base_signals.append(np.sin(2 * np.pi * 0.1 * t))
    
    # Signal 3: Mixed frequencies
    base_signals.append(np.sin(2 * np.pi * 0.05 * t) + 0.5 * np.sin(2 * np.pi * 0.15 * t))
    
    # Create connectivity patterns
    connectivity_matrix = np.zeros((n_voxels, n_voxels))
    
    # Assign signals to voxels with spatial clustering
    voxel_signals = np.zeros((n_voxels, n_timepoints))
    signal_assignments = np.zeros(n_voxels, dtype=int)
    
    for i in range(n_voxels):
        # Assign signals with some spatial structure
        if i < n_voxels // 3:
            signal_idx = 0
        elif i < 2 * n_voxels // 3:
            signal_idx = 1
        else:
            signal_idx = 2
            
        signal_assignments[i] = signal_idx
        voxel_signals[i] = base_signals[signal_idx]
        
        # Add some lagged dependencies
        if i > 0 and np.random.rand() < 0.3:  # 30% chance of dependency
            source_idx = np.random.choice(i)
            lag = np.random.randint(1, 4)
            coupling_strength = np.random.uniform(0.3, 0.7)
            
            # Add lagged influence
            if lag < n_timepoints:
                voxel_signals[i, lag:] += coupling_strength * voxel_signals[source_idx, :-lag]
                connectivity_matrix[i, source_idx] = coupling_strength
    
    # Add noise
    noise = np.random.randn(n_voxels, n_timepoints) * noise_level
    voxel_signals += noise
    
    # Place voxel signals into 4D array
    mask = np.zeros((dim, dim, dim), dtype=bool)
    voxel_idx = 0
    
    for x in range(dim):
        for y in range(dim):
            for z in range(dim):
                if voxel_idx < n_voxels:
                    fmri_data[x, y, z, :] = voxel_signals[voxel_idx]
                    mask[x, y, z] = True
                    voxel_idx += 1
                else:
                    break
    
    return fmri_data, mask, connectivity_matrix


def create_test_nifti_files(fmri_data: np.ndarray, 
                           mask: np.ndarray,
                           output_dir: str) -> tuple:
    """
    Create NIfTI files for testing.
    
    Parameters:
    -----------
    fmri_data : np.ndarray
        4D fMRI data
    mask : np.ndarray
        3D brain mask
    output_dir : str
        Output directory
        
    Returns:
    --------
    tuple
        (fmri_path, mask_path)
    """
    # Create affine matrix (identity for simplicity)
    affine = np.eye(4)
    
    # Create and save fMRI NIfTI
    fmri_img = nib.Nifti1Image(fmri_data, affine)
    fmri_path = os.path.join(output_dir, 'test_fmri.nii.gz')
    nib.save(fmri_img, fmri_path)
    
    # Create and save mask NIfTI
    mask_img = nib.Nifti1Image(mask.astype(np.uint8), affine)
    mask_path = os.path.join(output_dir, 'test_mask.nii.gz')
    nib.save(mask_img, mask_path)
    
    return fmri_path, mask_path


def test_symbolization_methods():
    """Test different symbolization methods."""
    print("Testing symbolization methods...")
    
    # Generate test time series
    t = np.linspace(0, 10, 100)
    ts1 = np.sin(2 * np.pi * 0.5 * t) + 0.1 * np.random.randn(100)
    ts2 = np.sin(2 * np.pi * 0.3 * t + np.pi/4) + 0.1 * np.random.randn(100)
    
    test_data = np.array([ts1, ts2])
    
    methods = ['uniform', 'quantile', 'ordinal', 'vq']
    
    plt.figure(figsize=(15, 10))
    
    for i, method in enumerate(methods):
        analyzer = VoxelSMTEConnectivity(
            n_symbols=5,
            symbolizer=method,
            max_lag=3
        )
        
        symbolic_data = analyzer.symbolize_timeseries(test_data)
        
        # Plot original and symbolic data
        plt.subplot(4, 2, 2*i + 1)
        plt.plot(t, ts1, label='TS1', alpha=0.7)
        plt.plot(t, ts2, label='TS2', alpha=0.7)
        plt.title(f'{method.title()} - Original')
        plt.legend()
        
        plt.subplot(4, 2, 2*i + 2)
        plt.plot(symbolic_data[0], label='TS1 symbols', marker='o', markersize=3)
        plt.plot(symbolic_data[1], label='TS2 symbols', marker='s', markersize=3)
        plt.title(f'{method.title()} - Symbolic')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('symbolization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Symbolization methods tested successfully!")


def test_smte_computation():
    """Test SMTE computation with known relationships."""
    print("Testing SMTE computation...")
    
    # Create time series with known relationship
    n_points = 200
    ts_source = np.random.randn(n_points)
    
    # Create target with lag-1 dependency
    ts_target = np.zeros(n_points)
    ts_target[1:] = 0.6 * ts_source[:-1] + 0.4 * np.random.randn(n_points-1)
    ts_target[0] = np.random.randn()
    
    # Create independent series for comparison
    ts_independent = np.random.randn(n_points)
    
    analyzer = VoxelSMTEConnectivity(
        n_symbols=5,
        symbolizer='uniform',
        max_lag=5
    )
    
    # Symbolize
    test_data = np.array([ts_target, ts_source, ts_independent])
    symbolic_data = analyzer.symbolize_timeseries(test_data)
    analyzer.symbolic_data = symbolic_data
    
    # Compute SMTE values
    smte_source_to_target, lag_st = analyzer._compute_smte_pair(symbolic_data[0], symbolic_data[1])
    smte_target_to_source, lag_ts = analyzer._compute_smte_pair(symbolic_data[1], symbolic_data[0])
    smte_indep_to_target, lag_it = analyzer._compute_smte_pair(symbolic_data[0], symbolic_data[2])
    
    print(f"SMTE source -> target: {smte_source_to_target:.4f} (lag: {lag_st})")
    print(f"SMTE target -> source: {smte_target_to_source:.4f} (lag: {lag_ts})")
    print(f"SMTE independent -> target: {smte_indep_to_target:.4f} (lag: {lag_it})")
    
    # Expected: source -> target should be higher than others
    assert smte_source_to_target > smte_target_to_source, "Source->Target should be higher"
    assert smte_source_to_target > smte_indep_to_target, "Source->Target should be higher than independent"
    
    print("SMTE computation test passed!")


def run_full_pipeline_test():
    """Run complete pipeline test with synthetic data."""
    print("Running full pipeline test...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Generate synthetic data
        print("Generating synthetic fMRI data...")
        fmri_data, mask, ground_truth = generate_synthetic_fmri_data(
            n_voxels=50,  # Small for testing
            n_timepoints=150,
            noise_level=0.2
        )
        
        # Create NIfTI files
        fmri_path, mask_path = create_test_nifti_files(fmri_data, mask, temp_dir)
        
        # Initialize analyzer
        analyzer = VoxelSMTEConnectivity(
            n_symbols=5,
            symbolizer='uniform',
            max_lag=3,
            alpha=0.05,
            n_permutations=100,  # Reduced for testing
            n_jobs=2,
            memory_efficient=True
        )
        
        # Run analysis on subset of voxels
        n_test_voxels = 20
        voxel_indices = list(range(n_test_voxels))
        
        print("Running SMTE connectivity analysis...")
        results = analyzer.run_complete_analysis(
            fmri_path=fmri_path,
            output_dir=os.path.join(temp_dir, 'results'),
            mask_path=mask_path,
            voxel_indices=voxel_indices,
            visualize=True
        )
        
        # Analyze results
        print(f"\nAnalysis Results:")
        print(f"- Number of voxels analyzed: {n_test_voxels}")
        print(f"- Number of significant connections: {results['n_significant_connections']}")
        print(f"- Network density: {results['network_properties']['density']:.4f}")
        print(f"- Mean connection strength: {results['network_properties']['mean_connection_strength']:.4f}")
        
        # Plot results
        plt.figure(figsize=(15, 5))
        
        # SMTE matrix
        plt.subplot(1, 3, 1)
        plt.imshow(results['smte_matrix'], cmap='viridis', aspect='auto')
        plt.colorbar(label='SMTE Value')
        plt.title('SMTE Matrix')
        plt.xlabel('Source Voxel')
        plt.ylabel('Target Voxel')
        
        # P-values
        plt.subplot(1, 3, 2)
        plt.imshow(-np.log10(results['p_values'] + 1e-10), cmap='hot', aspect='auto')
        plt.colorbar(label='-log10(p-value)')
        plt.title('Statistical Significance')
        plt.xlabel('Source Voxel')
        plt.ylabel('Target Voxel')
        
        # Significant connections
        plt.subplot(1, 3, 3)
        plt.imshow(results['significance_mask'].astype(int), cmap='binary', aspect='auto')
        plt.colorbar(label='Significant')
        plt.title('Significant Connections\n(FDR corrected)')
        plt.xlabel('Source Voxel')
        plt.ylabel('Target Voxel')
        
        plt.tight_layout()
        plt.savefig('pipeline_test_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Network analysis
        if results['connectivity_graph'].number_of_edges() > 0:
            print(f"\nNetwork Properties:")
            props = results['network_properties']
            print(f"- Hub nodes: {len(props['hub_nodes'])}")
            print(f"- Mean in-degree centrality: {np.mean(list(props['in_degree_centrality'].values())):.4f}")
            print(f"- Mean out-degree centrality: {np.mean(list(props['out_degree_centrality'].values())):.4f}")
        
        print("Full pipeline test completed successfully!")
        
        return results


def performance_benchmark():
    """Benchmark performance with different settings."""
    print("Running performance benchmark...")
    
    # Test different numbers of voxels
    voxel_counts = [10, 20, 50]
    methods = ['uniform', 'ordinal']
    
    results = []
    
    for n_voxels in voxel_counts:
        for method in methods:
            print(f"Testing {n_voxels} voxels with {method} symbolization...")
            
            # Generate synthetic data
            fmri_data, mask, _ = generate_synthetic_fmri_data(
                n_voxels=n_voxels,
                n_timepoints=100,
                noise_level=0.3
            )
            
            # Initialize analyzer
            analyzer = VoxelSMTEConnectivity(
                n_symbols=5,
                symbolizer=method,
                max_lag=2,
                n_permutations=50  # Reduced for speed
            )
            
            # Extract time series
            voxel_ts = fmri_data[mask]
            
            # Time symbolization
            import time
            start_time = time.time()
            symbolic_data = analyzer.symbolize_timeseries(voxel_ts)
            symbolization_time = time.time() - start_time
            
            # Time SMTE computation (subset)
            analyzer.symbolic_data = symbolic_data
            start_time = time.time()
            n_test = min(10, n_voxels)
            smte_matrix = np.zeros((n_test, n_test))
            
            for i in range(n_test):
                for j in range(n_test):
                    if i != j:
                        smte_val, _ = analyzer._compute_smte_pair(
                            symbolic_data[i], symbolic_data[j]
                        )
                        smte_matrix[i, j] = smte_val
            
            smte_time = time.time() - start_time
            
            results.append({
                'n_voxels': n_voxels,
                'method': method,
                'symbolization_time': symbolization_time,
                'smte_time': smte_time,
                'total_time': symbolization_time + smte_time
            })
            
            print(f"  Symbolization: {symbolization_time:.2f}s, SMTE: {smte_time:.2f}s")
    
    # Plot benchmark results
    import pandas as pd
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    for method in methods:
        method_data = df[df['method'] == method]
        plt.plot(method_data['n_voxels'], method_data['symbolization_time'], 
                marker='o', label=method)
    plt.xlabel('Number of Voxels')
    plt.ylabel('Symbolization Time (s)')
    plt.title('Symbolization Performance')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    for method in methods:
        method_data = df[df['method'] == method]
        plt.plot(method_data['n_voxels'], method_data['smte_time'], 
                marker='o', label=method)
    plt.xlabel('Number of Voxels')
    plt.ylabel('SMTE Computation Time (s)')
    plt.title('SMTE Performance')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    for method in methods:
        method_data = df[df['method'] == method]
        plt.plot(method_data['n_voxels'], method_data['total_time'], 
                marker='o', label=method)
    plt.xlabel('Number of Voxels')
    plt.ylabel('Total Time (s)')
    plt.title('Total Performance')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('performance_benchmark.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Performance benchmark completed!")


def main():
    """Run all tests."""
    print("=== VoxelSMTEConnectivity Test Suite ===\n")
    
    try:
        # Test 1: Symbolization methods
        test_symbolization_methods()
        print()
        
        # Test 2: SMTE computation
        test_smte_computation()
        print()
        
        # Test 3: Full pipeline
        run_full_pipeline_test()
        print()
        
        # Test 4: Performance benchmark
        performance_benchmark()
        print()
        
        print("=== All tests completed successfully! ===")
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()