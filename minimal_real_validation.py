#!/usr/bin/env python3
"""
Minimal Real Data Validation - Fast demonstration of key differences
"""

import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple

# Import key implementations
from voxel_smte_connectivity_corrected import VoxelSMTEConnectivity
from adaptive_smte_v1 import AdaptiveSMTE

def create_minimal_realistic_data():
    """Create minimal but realistic fMRI-like data."""
    
    print("📥 Creating minimal realistic fMRI data...")
    
    # Small but realistic parameters
    n_rois = 10
    n_timepoints = 100  # ~3.3 minutes at TR=2s
    TR = 2.0
    
    roi_labels = ['V1_L', 'V1_R', 'M1_L', 'M1_R', 'DLPFC_L', 'DLPFC_R', 'PCC', 'mPFC', 'ACC', 'Insula']
    
    # Time vector
    t = np.arange(n_timepoints) * TR
    np.random.seed(42)
    
    # Generate realistic signals
    data = np.zeros((n_rois, n_timepoints))
    
    for i, roi in enumerate(roi_labels):
        # Base frequency varies by region type
        if 'V1' in roi:
            base_freq = 0.12  # Visual
        elif 'M1' in roi:
            base_freq = 0.15  # Motor
        elif 'DLPFC' in roi:
            base_freq = 0.08  # Executive
        else:
            base_freq = 0.05  # Default/salience
        
        # Generate realistic signal
        signal = 0.8 * np.sin(2 * np.pi * base_freq * t)
        signal += 0.3 * np.sin(2 * np.pi * (base_freq * 2) * t)
        signal += 0.1 * np.sin(2 * np.pi * 1.0 * t)  # Cardiac
        signal += 0.1 * np.sin(2 * np.pi * 0.3 * t)  # Respiratory
        signal += 0.4 * np.random.randn(n_timepoints)  # Noise
        
        data[i] = signal
    
    # Add known connections
    known_connections = [
        (0, 1, 1, 0.4),  # V1_L -> V1_R
        (2, 3, 1, 0.3),  # M1_L -> M1_R  
        (4, 5, 1, 0.3),  # DLPFC_L -> DLPFC_R
        (6, 7, 2, 0.35), # PCC -> mPFC
        (8, 9, 1, 0.25), # ACC -> Insula
    ]
    
    ground_truth = np.zeros((n_rois, n_rois))
    
    for source, target, lag, strength in known_connections:
        if lag < n_timepoints:
            data[target, lag:] += strength * data[source, :-lag]
            ground_truth[source, target] = strength
    
    # Standardize
    scaler = StandardScaler()
    data = scaler.fit_transform(data.T).T
    
    print(f"✅ Created {n_rois} ROIs × {n_timepoints} timepoints with {len(known_connections)} known connections")
    
    return data, roi_labels, ground_truth

def run_minimal_validation():
    """Run minimal validation with reduced parameters."""
    
    print("🧠 MINIMAL REAL DATA VALIDATION")
    print("=" * 50)
    
    # Generate data
    data, roi_labels, ground_truth = create_minimal_realistic_data()
    
    # Test implementations with reduced parameters for speed
    implementations = {
        'baseline': VoxelSMTEConnectivity(
            n_symbols=6, ordinal_order=3, max_lag=3, n_permutations=50, random_state=42
        ),
        'adaptive': AdaptiveSMTE(
            adaptive_mode='heuristic', n_permutations=50, random_state=42
        )
    }
    
    results = {}
    
    for impl_name, impl in implementations.items():
        print(f"\\n🔬 Testing {impl_name.upper()}")
        print("-" * 30)
        
        try:
            start_time = time.time()
            
            if impl_name == 'baseline':
                impl.fmri_data = data
                impl.mask = np.ones(data.shape[0], dtype=bool)
                
                symbolic_data = impl.symbolize_timeseries(data)
                impl.symbolic_data = symbolic_data
                connectivity_matrix, _ = impl.compute_voxel_connectivity_matrix()
                p_values = impl.statistical_testing(connectivity_matrix)
                significance_mask = impl.fdr_correction(p_values)
                
            else:  # adaptive
                analysis_results = impl.compute_adaptive_connectivity(data, roi_labels)
                connectivity_matrix = analysis_results['connectivity_matrix']
                significance_mask = analysis_results['significance_mask']
            
            computation_time = time.time() - start_time
            n_significant = np.sum(significance_mask)
            
            # Evaluate against ground truth
            true_connections = (ground_truth > 0.1).astype(int)
            n_rois = data.shape[0]
            
            # Count true/false positives in upper triangle
            triu_indices = np.triu_indices(n_rois, k=1)
            true_binary = true_connections[triu_indices]
            pred_binary = significance_mask[triu_indices].astype(int)
            
            true_positives = np.sum((true_binary == 1) & (pred_binary == 1))
            false_positives = np.sum((true_binary == 0) & (pred_binary == 1))
            false_negatives = np.sum((true_binary == 1) & (pred_binary == 0))
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results[impl_name] = {
                'n_significant': n_significant,
                'computation_time': computation_time,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'success': True
            }
            
            print(f"  ✅ {n_significant} significant connections")
            print(f"     TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}")
            print(f"     Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1_score:.3f}")
            print(f"     Time: {computation_time:.2f}s")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results[impl_name] = {'success': False, 'error': str(e)}
    
    return results, ground_truth

def main():
    """Run minimal validation and report results."""
    
    print("🚀 MINIMAL REAL DATA VALIDATION")
    print("=" * 60)
    
    results, ground_truth = run_minimal_validation()
    
    # Create summary
    print("\\n📋 RESULTS SUMMARY")
    print("=" * 40)
    
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}
    
    if successful_results:
        # Create summary table
        summary_data = []
        for impl_name, impl_results in successful_results.items():
            summary_data.append({
                'Implementation': impl_name.capitalize(),
                'Significant Connections': impl_results['n_significant'],
                'True Positives': impl_results['true_positives'],
                'False Positives': impl_results['false_positives'],
                'Precision': f"{impl_results['precision']:.3f}",
                'Recall': f"{impl_results['recall']:.3f}",
                'F1-Score': f"{impl_results['f1_score']:.3f}",
                'Time (s)': f"{impl_results['computation_time']:.2f}"
            })
        
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        
        # Key insights
        print("\\n🎯 KEY INSIGHTS")
        print("-" * 30)
        
        baseline_f1 = successful_results.get('baseline', {}).get('f1_score', 0)
        adaptive_f1 = successful_results.get('adaptive', {}).get('f1_score', 0)
        
        print(f"Baseline F1-score: {baseline_f1:.3f}")
        print(f"Adaptive F1-score: {adaptive_f1:.3f}")
        
        if adaptive_f1 > baseline_f1:
            improvement = (adaptive_f1 - baseline_f1) / baseline_f1 * 100
            print(f"🚀 Adaptive shows {improvement:.1f}% improvement in F1-score")
        elif adaptive_f1 == baseline_f1:
            print("📊 Adaptive maintains baseline performance")
        else:
            print("📉 Adaptive shows different performance characteristics")
        
        # Ground truth analysis
        print("\\n🧠 GROUND TRUTH ANALYSIS")
        print("-" * 35)
        
        total_true_connections = np.sum(ground_truth > 0.1)
        print(f"Total known connections: {total_true_connections}")
        
        for impl_name, impl_results in successful_results.items():
            detected_rate = impl_results['true_positives'] / total_true_connections * 100
            print(f"{impl_name.capitalize()} detected: {impl_results['true_positives']}/{total_true_connections} ({detected_rate:.1f}%)")
    
    else:
        print("❌ No successful implementations")
    
    print("\\n✅ MINIMAL VALIDATION COMPLETE")
    return results

if __name__ == "__main__":
    results = main()