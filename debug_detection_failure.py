#!/usr/bin/env python3
"""
Debug Detection Failure: Find out why SMTE framework detects nothing
Systematic investigation of the detection pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
from voxel_smte_connectivity_corrected import VoxelSMTEConnectivity

def create_perfect_test_case():
    """Create the most obvious possible connection for testing."""
    
    print("🔍 CREATING PERFECT TEST CASE")
    print("=" * 40)
    
    # Very simple, obvious test case
    n_timepoints = 200
    
    # Source signal: clean sine wave
    t = np.arange(n_timepoints)
    source_signal = np.sin(2 * np.pi * 0.1 * t / 10)  # 0.1 Hz
    
    # Target signal: exact copy of source with 1-step lag (perfect connection)
    target_signal = np.zeros(n_timepoints)
    target_signal[1:] = source_signal[:-1]  # Perfect 1-step lag
    
    # Stack into data matrix
    data = np.array([source_signal, target_signal])
    
    print(f"✅ Created perfect test case:")
    print(f"   Signal 1: Clean sine wave")
    print(f"   Signal 2: Perfect copy with 1-step lag")
    print(f"   Expected connection: Signal 1 → Signal 2 (strength = 1.0)")
    
    return data

def debug_smte_pipeline():
    """Debug each step of the SMTE pipeline."""
    
    print("\n🔬 DEBUGGING SMTE PIPELINE")
    print("=" * 40)
    
    # Create perfect test data
    data = create_perfect_test_case()
    
    # Initialize SMTE with minimal parameters
    smte = VoxelSMTEConnectivity(
        n_symbols=2,           # Simplest possible
        ordinal_order=2,       # Simplest possible 
        max_lag=3,             # Allow lag detection
        n_permutations=10,     # Fast testing
        random_state=42
    )
    
    print("\n1. TESTING SYMBOLIZATION")
    print("-" * 25)
    
    try:
        symbolic_data = smte.symbolize_timeseries(data)
        print(f"✅ Symbolization successful")
        print(f"   Original shape: {data.shape}")
        print(f"   Symbolic shape: {symbolic_data.shape}")
        print(f"   Unique symbols in signal 1: {np.unique(symbolic_data[0])}")
        print(f"   Unique symbols in signal 2: {np.unique(symbolic_data[1])}")
        
        # Check if signals are properly symbolized
        if len(np.unique(symbolic_data[0])) < 2:
            print(f"⚠️ Warning: Signal 1 has only {len(np.unique(symbolic_data[0]))} unique symbols")
        if len(np.unique(symbolic_data[1])) < 2:
            print(f"⚠️ Warning: Signal 2 has only {len(np.unique(symbolic_data[1]))} unique symbols")
            
    except Exception as e:
        print(f"❌ Symbolization failed: {e}")
        return False
    
    print("\n2. TESTING SMTE COMPUTATION")
    print("-" * 25)
    
    try:
        smte.fmri_data = data
        smte.mask = np.ones(data.shape[0], dtype=bool)
        smte.symbolic_data = symbolic_data
        
        connectivity_matrix, _ = smte.compute_voxel_connectivity_matrix()
        print(f"✅ SMTE computation successful")
        print(f"   Connectivity matrix shape: {connectivity_matrix.shape}")
        print(f"   Connection 1→2: {connectivity_matrix[0, 1]:.6f}")
        print(f"   Connection 2→1: {connectivity_matrix[1, 0]:.6f}")
        
        # Check if any non-zero values
        max_value = np.max(connectivity_matrix)
        print(f"   Maximum SMTE value: {max_value:.6f}")
        
        if max_value == 0:
            print("⚠️ Warning: All SMTE values are zero!")
        
    except Exception as e:
        print(f"❌ SMTE computation failed: {e}")
        return False
    
    print("\n3. TESTING STATISTICAL TESTING")
    print("-" * 30)
    
    try:
        p_values = smte.statistical_testing(connectivity_matrix)
        print(f"✅ Statistical testing successful")
        print(f"   P-values shape: {p_values.shape}")
        print(f"   P-value 1→2: {p_values[0, 1]:.6f}")
        print(f"   P-value 2→1: {p_values[1, 0]:.6f}")
        
        # Check p-value range
        min_p = np.min(p_values[p_values > 0])
        max_p = np.max(p_values)
        print(f"   P-value range: {min_p:.6f} to {max_p:.6f}")
        
        # Check how many are significant at different levels
        sig_001 = np.sum(p_values < 0.01)
        sig_005 = np.sum(p_values < 0.05)
        sig_010 = np.sum(p_values < 0.10)
        
        print(f"   Significant at p < 0.01: {sig_001}")
        print(f"   Significant at p < 0.05: {sig_005}") 
        print(f"   Significant at p < 0.10: {sig_010}")
        
    except Exception as e:
        print(f"❌ Statistical testing failed: {e}")
        return False
    
    print("\n4. TESTING FDR CORRECTION")
    print("-" * 25)
    
    try:
        # Test different alpha levels
        for alpha in [0.01, 0.05, 0.10, 0.20]:
            smte.alpha = alpha
            significance_mask = smte.fdr_correction(p_values)
            n_significant = np.sum(significance_mask)
            
            print(f"   α = {alpha}: {n_significant} significant connections")
            
            if n_significant > 0:
                print(f"   ✅ Found significant connections at α = {alpha}")
                sig_connections = np.where(significance_mask)
                for i, j in zip(sig_connections[0], sig_connections[1]):
                    print(f"     Connection {i}→{j}: SMTE={connectivity_matrix[i,j]:.6f}, p={p_values[i,j]:.6f}")
        
    except Exception as e:
        print(f"❌ FDR correction failed: {e}")
        return False
    
    print("\n5. MANUAL VERIFICATION")
    print("-" * 20)
    
    # Manual check: compute simple correlation as sanity check
    from scipy.stats import pearsonr
    
    # Check if the 1-lag relationship exists
    signal1 = data[0]
    signal2_lagged = data[1]
    
    # Test correlation between signal1[:-1] and signal2[1:]
    corr, p_corr = pearsonr(signal1[:-1], signal2_lagged[1:])
    print(f"Manual correlation check (1-lag): r={corr:.6f}, p={p_corr:.6f}")
    
    # Direct test: signal1[t] should predict signal2[t+1]
    corr_direct, p_direct = pearsonr(signal1[:-1], signal2_lagged[1:])
    print(f"Direct lag-1 correlation: r={corr_direct:.6f}, p={p_direct:.6f}")
    
    if abs(corr_direct) > 0.8:
        print("✅ Strong correlation confirmed - connection should be detectable")
    else:
        print("⚠️ Weak correlation - may explain detection failure")
    
    return True

def test_uncorrected_detection():
    """Test detection without multiple comparison correction."""
    
    print("\n6. TESTING WITHOUT FDR CORRECTION")
    print("-" * 35)
    
    data = create_perfect_test_case()
    
    smte = VoxelSMTEConnectivity(
        n_symbols=2,
        ordinal_order=2,
        max_lag=3,
        n_permutations=50,  # More permutations for stable p-values
        random_state=42
    )
    
    smte.fmri_data = data
    smte.mask = np.ones(data.shape[0], dtype=bool)
    
    symbolic_data = smte.symbolize_timeseries(data)
    smte.symbolic_data = symbolic_data
    connectivity_matrix, _ = smte.compute_voxel_connectivity_matrix()
    p_values = smte.statistical_testing(connectivity_matrix)
    
    print(f"Raw p-values without correction:")
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if i != j:
                print(f"  Connection {i}→{j}: SMTE={connectivity_matrix[i,j]:.6f}, p={p_values[i,j]:.6f}")
    
    # Check significance at different thresholds WITHOUT correction
    for threshold in [0.001, 0.01, 0.05, 0.10, 0.20, 0.50]:
        significant = p_values < threshold
        n_sig = np.sum(significant)
        print(f"Raw p < {threshold}: {n_sig} significant")
        
        if n_sig > 0:
            print(f"  ✅ DETECTION SUCCESS at raw p < {threshold}")
            return True
    
    print("❌ No detection even without multiple comparison correction")
    return False

def main():
    """Run comprehensive debugging of detection failure."""
    
    print("🚨 DEBUGGING SMTE DETECTION FAILURE")
    print("=" * 60)
    print("Investigating why framework detects nothing even under perfect conditions")
    print("=" * 60)
    
    # Run pipeline debugging
    pipeline_success = debug_smte_pipeline()
    
    if pipeline_success:
        print("\n📊 Pipeline completed - testing uncorrected detection...")
        uncorrected_success = test_uncorrected_detection()
        
        if uncorrected_success:
            print("\n🎯 DIAGNOSIS: Multiple comparison correction too conservative")
            print("SOLUTION: Implement less stringent correction or uncorrected option")
        else:
            print("\n🚨 DIAGNOSIS: Fundamental issue in SMTE computation or statistical testing")
            print("SOLUTION: Review SMTE algorithm implementation and surrogate data generation")
    else:
        print("\n🚨 DIAGNOSIS: Pipeline failure in symbolization or SMTE computation")
        print("SOLUTION: Debug symbolization parameters and SMTE algorithm")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR IMPROVEMENT")
    print("="*60)
    
    print("\n1. IMMEDIATE FIXES:")
    print("   - Add option to disable multiple comparison correction")
    print("   - Implement alternative statistical tests (permutation, bootstrap)")
    print("   - Add diagnostic outputs for each pipeline step")
    
    print("\n2. PARAMETER OPTIMIZATION:")
    print("   - Test different symbolization parameters")
    print("   - Optimize ordinal pattern parameters") 
    print("   - Increase number of permutations for stable p-values")
    
    print("\n3. VALIDATION IMPROVEMENTS:")
    print("   - Add known-working reference implementations")
    print("   - Test against published SMTE results")
    print("   - Implement synthetic data with known ground truth")
    
    print("\n4. ALTERNATIVE APPROACHES:")
    print("   - Implement Transfer Entropy variants")
    print("   - Add correlation-based connectivity for comparison")
    print("   - Consider different time series analysis methods")

if __name__ == "__main__":
    main()