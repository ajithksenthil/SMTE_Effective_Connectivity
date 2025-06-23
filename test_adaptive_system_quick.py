#!/usr/bin/env python3
"""
Quick Test of Adaptive Temporal Resolution System
Focused test to validate the adaptive parameter optimization.
"""

import numpy as np
import logging
from adaptive_temporal_system import AdaptiveTemporalSystem, AdaptiveSMTEConnectivity

def quick_test_adaptive_system():
    """Quick test of the adaptive temporal system."""
    
    print("🚀 QUICK TEST: ADAPTIVE TEMPORAL RESOLUTION SYSTEM")
    print("=" * 60)
    
    # Test scenarios
    scenarios = [
        {"name": "High-res", "tr": 0.5, "n_timepoints": 300},
        {"name": "Standard", "tr": 2.0, "n_timepoints": 200},
        {"name": "Clinical", "tr": 3.0, "n_timepoints": 150}
    ]
    
    system = AdaptiveTemporalSystem()
    results = {}
    
    for scenario in scenarios:
        print(f"\n📊 {scenario['name']} fMRI (TR={scenario['tr']}s)")
        
        # Test parameter optimization
        params = system.optimize_temporal_parameters(
            scenario['tr'], scenario['n_timepoints']
        )
        
        print(f"   Optimized max_lag: {params.max_lag} samples ({params.max_lag * params.tr:.1f}s)")
        print(f"   Optimal lags: {params.optimal_lags}")
        print(f"   Symbolization: {params.n_symbols} symbols, order {params.ordinal_order}")
        print(f"   Confidence: {params.confidence_score:.3f}")
        
        # Generate test data
        np.random.seed(42)
        n_rois = 5  # Smaller for speed
        data = np.random.randn(n_rois, scenario['n_timepoints'])
        
        # Add connection at optimal hemodynamic lag
        hrf_lag = max(1, min(params.max_lag, int(6.0 / scenario['tr'])))
        if hrf_lag < scenario['n_timepoints']:
            data[1, hrf_lag:] += 0.6 * data[0, :-hrf_lag]
            print(f"   Added connection at lag {hrf_lag} ({hrf_lag * scenario['tr']:.1f}s)")
        
        # Test with reduced permutations for speed
        try:
            adaptive_smte = AdaptiveSMTEConnectivity(
                tr=scenario['tr'], 
                n_permutations=20  # Reduced for speed
            )
            
            config_results = adaptive_smte.auto_configure(data)
            
            # Quick SMTE test
            adaptive_smte.base_smte.fmri_data = data
            adaptive_smte.base_smte.mask = np.ones(n_rois, dtype=bool)
            
            symbolic_data = adaptive_smte.base_smte.symbolize_timeseries(data)
            adaptive_smte.base_smte.symbolic_data = symbolic_data
            
            connectivity_matrix, _ = adaptive_smte.base_smte.compute_voxel_connectivity_matrix()
            p_values = adaptive_smte.base_smte.statistical_testing(connectivity_matrix)
            
            # Check test connection
            test_p_value = p_values[0, 1]
            connection_strength = connectivity_matrix[0, 1]
            
            print(f"   Connection strength: {connection_strength:.4f}")
            print(f"   Connection p-value: {test_p_value:.4f}")
            print(f"   Detected (p<0.05): {'✅' if test_p_value < 0.05 else '❌'}")
            
            results[scenario['name']] = {
                'max_lag': params.max_lag,
                'confidence': params.confidence_score,
                'connection_p': test_p_value,
                'connection_strength': connection_strength,
                'detected': test_p_value < 0.05
            }
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[scenario['name']] = {'error': str(e)}
    
    # Summary
    print(f"\n📋 QUICK TEST SUMMARY")
    print("=" * 30)
    
    detected_count = sum(1 for r in results.values() if r.get('detected', False))
    total_count = len([r for r in results.values() if 'error' not in r])
    
    print(f"Successful optimizations: {len(results) - sum(1 for r in results.values() if 'error' in r)}/{len(results)}")
    print(f"Successful detections: {detected_count}/{total_count}")
    
    for name, result in results.items():
        if 'error' not in result:
            status = "✅" if result['detected'] else "❌"
            print(f"{name}: {status} (p={result['connection_p']:.4f}, lag={result['max_lag']})")
    
    # Test improvement over fixed parameters
    print(f"\n🔄 COMPARISON WITH FIXED PARAMETERS")
    print("-" * 40)
    
    # Test same scenarios with fixed max_lag=3
    for scenario in scenarios[:2]:  # Just test first two
        print(f"\n{scenario['name']} - Fixed vs Adaptive:")
        
        np.random.seed(42)  # Same data
        data = np.random.randn(5, scenario['n_timepoints'])
        hrf_lag = max(1, int(6.0 / scenario['tr']))
        if hrf_lag < scenario['n_timepoints']:
            data[1, hrf_lag:] += 0.6 * data[0, :-hrf_lag]
        
        # Fixed parameters (original approach)
        try:
            from voxel_smte_connectivity_corrected import VoxelSMTEConnectivity
            fixed_smte = VoxelSMTEConnectivity(
                n_symbols=2, ordinal_order=2, max_lag=3, n_permutations=20
            )
            fixed_smte.fmri_data = data
            fixed_smte.mask = np.ones(5, dtype=bool)
            
            symbolic_data = fixed_smte.symbolize_timeseries(data)
            fixed_smte.symbolic_data = symbolic_data
            connectivity_matrix, _ = fixed_smte.compute_voxel_connectivity_matrix()
            p_values = fixed_smte.statistical_testing(connectivity_matrix)
            
            fixed_p = p_values[0, 1]
            adaptive_p = results[scenario['name']]['connection_p']
            
            print(f"  Fixed (lag=3): p={fixed_p:.4f} {'✅' if fixed_p < 0.05 else '❌'}")
            print(f"  Adaptive (lag={results[scenario['name']]['max_lag']}): p={adaptive_p:.4f} {'✅' if adaptive_p < 0.05 else '❌'}")
            
            if adaptive_p < fixed_p:
                improvement = (fixed_p - adaptive_p) / fixed_p * 100
                print(f"  🎯 Adaptive improvement: {improvement:.1f}% better p-value")
            
        except Exception as e:
            print(f"  Fixed test failed: {e}")
    
    return results

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Run quick test
    test_results = quick_test_adaptive_system()