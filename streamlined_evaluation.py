#!/usr/bin/env python3
"""
Streamlined Research-Grade Evaluation for Paper Generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

# Import key implementations for comparison
from voxel_smte_connectivity_corrected import VoxelSMTEConnectivity
from validation_framework import SMTEValidationFramework

warnings.filterwarnings('ignore')

def run_streamlined_evaluation():
    """
    Run streamlined evaluation for paper results.
    """
    
    print("🔬 STREAMLINED RESEARCH EVALUATION")
    print("=" * 60)
    
    # Use validation framework for consistent testing
    validator = SMTEValidationFramework(random_state=42)
    
    # Test implementations with realistic parameters
    implementations = {
        'baseline': {
            'class': VoxelSMTEConnectivity,
            'params': {
                'n_symbols': 6,
                'ordinal_order': 3,
                'max_lag': 5,
                'n_permutations': 100,
                'random_state': 42
            }
        }
    }
    
    # Import and test enhanced implementations
    try:
        from adaptive_smte_v1 import AdaptiveSMTE
        implementations['adaptive'] = {
            'class': AdaptiveSMTE,
            'params': {
                'adaptive_mode': 'heuristic',
                'n_permutations': 100,
                'random_state': 42
            }
        }
    except ImportError:
        pass
    
    try:
        from network_aware_smte_v1 import NetworkAwareSMTE
        implementations['network_aware'] = {
            'class': NetworkAwareSMTE,
            'params': {
                'adaptive_mode': 'heuristic',
                'use_network_correction': True,
                'n_permutations': 100,
                'random_state': 42
            }
        }
    except ImportError:
        pass
    
    try:
        from physiological_smte_v1 import PhysiologicalSMTE
        implementations['physiological'] = {
            'class': PhysiologicalSMTE,
            'params': {
                'adaptive_mode': 'heuristic',
                'use_network_correction': True,
                'use_physiological_constraints': True,
                'n_permutations': 100,
                'random_state': 42
            }
        }
    except ImportError:
        pass
    
    try:
        from multiscale_smte_v1 import MultiScaleSMTE
        implementations['multiscale'] = {
            'class': MultiScaleSMTE,
            'params': {
                'use_multiscale_analysis': True,
                'scales_to_analyze': ['fast', 'intermediate'],
                'adaptive_mode': 'heuristic',
                'use_network_correction': True,
                'use_physiological_constraints': True,
                'n_permutations': 100,
                'random_state': 42
            }
        }
    except ImportError:
        pass
    
    try:
        from ensemble_smte_v1 import EnsembleSMTE
        implementations['ensemble'] = {
            'class': EnsembleSMTE,
            'params': {
                'use_ensemble_testing': True,
                'surrogate_methods': ['aaft'],
                'n_surrogates_per_method': 20,
                'use_multiscale_analysis': True,
                'scales_to_analyze': ['fast'],
                'adaptive_mode': 'heuristic',
                'use_network_correction': True,
                'use_physiological_constraints': True,
                'n_permutations': 100,
                'random_state': 42
            }
        }
    except ImportError:
        pass
    
    try:
        from hierarchical_smte_v1 import HierarchicalSMTE
        implementations['hierarchical'] = {
            'class': HierarchicalSMTE,
            'params': {
                'use_hierarchical_analysis': True,
                'hierarchy_methods': ['agglomerative'],
                'hierarchy_levels': [2, 4],
                'distance_metrics': ['correlation'],
                'use_ensemble_testing': True,
                'surrogate_methods': ['aaft'],
                'n_surrogates_per_method': 15,
                'use_multiscale_analysis': True,
                'scales_to_analyze': ['fast'],
                'adaptive_mode': 'heuristic',
                'use_network_correction': True,
                'use_physiological_constraints': True,
                'n_permutations': 100,
                'random_state': 42
            }
        }
    except ImportError:
        pass
    
    # Run evaluations
    results = {}
    
    for impl_name, impl_config in implementations.items():
        print(f"\\nTesting {impl_name}...")
        
        # Create implementation instance
        if impl_name == 'baseline':
            impl = impl_config['class'](**impl_config['params'])
        else:
            impl = impl_config['class'](**impl_config['params'])
        
        # Run validation
        start_time = time.time()
        validation_results = validator.validate_implementation(impl, f"{impl_name}_implementation")
        end_time = time.time()
        
        # Store results
        results[impl_name] = {
            'validation_results': validation_results,
            'total_evaluation_time': end_time - start_time,
            'implementation_class': impl_config['class'].__name__,
            'parameters': impl_config['params']
        }
        
        # Print summary
        summary = validation_results['summary']
        print(f"  Performance: {summary['mean_performance_improvement']:.2%}")
        print(f"  Speed: {summary['mean_speedup']:.2f}x")
        
        # Count successful tests
        regression_check = validation_results['regression_check']
        successful_tests = sum(1 for success in regression_check.values() if success)
        total_tests = len(regression_check)
        print(f"  Success: {successful_tests}/{total_tests}")
    
    # Generate summary comparison
    summary_df = create_summary_comparison(results)
    
    # Save results
    import pickle
    with open('streamlined_evaluation_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    summary_df.to_csv('evaluation_summary.csv', index=False)
    
    print("\\n📊 EVALUATION COMPLETE")
    print(f"Tested {len(implementations)} implementations")
    print("Results saved to: streamlined_evaluation_results.pkl")
    print("Summary saved to: evaluation_summary.csv")
    
    return results, summary_df

def create_summary_comparison(results):
    """
    Create summary comparison table.
    """
    
    summary_data = []
    
    for impl_name, impl_results in results.items():
        validation = impl_results['validation_results']
        summary = validation['summary']
        
        # Count successful tests
        regression_check = validation['regression_check']
        successful_tests = sum(1 for success in regression_check.values() if success)
        total_tests = len(regression_check)
        
        row = {
            'Implementation': impl_name,
            'Class': impl_results['implementation_class'],
            'Mean Performance Improvement': f"{summary['mean_performance_improvement']:.2%}",
            'Mean Speedup': f"{summary['mean_speedup']:.2f}x",
            'Success Rate': f"{successful_tests}/{total_tests}",
            'Regression Checks Passed': all(validation['regression_check'].values()),
            'Total Evaluation Time (s)': f"{impl_results['total_evaluation_time']:.1f}"
        }
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)

if __name__ == "__main__":
    results, summary_df = run_streamlined_evaluation()
    print("\\n📋 SUMMARY TABLE")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print("=" * 100)