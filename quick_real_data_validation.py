#!/usr/bin/env python3
"""
Quick Real Data Validation of Enhanced SMTE Framework
Focused validation using realistic fMRI characteristics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any
import warnings

# Import SMTE implementations
from voxel_smte_connectivity_corrected import VoxelSMTEConnectivity
from adaptive_smte_v1 import AdaptiveSMTE
from physiological_smte_v1 import PhysiologicalSMTE
from multiscale_smte_v1 import MultiScaleSMTE
from ensemble_smte_v1 import EnsembleSMTE

warnings.filterwarnings('ignore')

def generate_realistic_fmri_data() -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Generate realistic fMRI data with known connectivity patterns.
    """
    
    print("📥 Generating realistic fMRI data...")
    
    # Realistic fMRI parameters
    n_rois = 20  # Manageable size for quick validation
    n_timepoints = 150  # ~5 minutes at TR=2s
    TR = 2.0
    
    # Define anatomically realistic ROI labels
    roi_labels = [
        # Visual network (4 ROIs)
        'V1_L', 'V1_R', 'V2_L', 'V2_R',
        # Motor network (4 ROIs)
        'M1_L', 'M1_R', 'S1_L', 'S1_R', 
        # Executive network (6 ROIs)
        'DLPFC_L', 'DLPFC_R', 'IFG_L', 'IFG_R', 'Parietal_L', 'Parietal_R',
        # Default mode network (4 ROIs)
        'PCC', 'mPFC', 'Angular_L', 'Angular_R',
        # Salience network (2 ROIs)
        'ACC', 'Insula'
    ]
    
    # Network assignments
    roi_networks = {
        0: 'Visual', 1: 'Visual', 2: 'Visual', 3: 'Visual',
        4: 'Motor', 5: 'Motor', 6: 'Motor', 7: 'Motor',
        8: 'Executive', 9: 'Executive', 10: 'Executive', 11: 'Executive', 12: 'Executive', 13: 'Executive',
        14: 'Default', 15: 'Default', 16: 'Default', 17: 'Default',
        18: 'Salience', 19: 'Salience'
    }
    
    # Time vector
    t = np.arange(n_timepoints) * TR
    np.random.seed(42)  # For reproducibility
    
    # Generate realistic network signals
    data = np.zeros((n_rois, n_timepoints))
    
    # Network-specific base frequencies (Hz)
    network_freqs = {
        'Visual': 0.12,
        'Motor': 0.15, 
        'Executive': 0.08,
        'Default': 0.05,
        'Salience': 0.10
    }
    
    # Generate base signals for each ROI
    for roi_idx, roi_label in enumerate(roi_labels):
        network = roi_networks[roi_idx]
        base_freq = network_freqs[network]
        
        # Base network oscillation
        signal = 0.8 * np.sin(2 * np.pi * base_freq * t)
        
        # Add harmonics for complexity
        signal += 0.3 * np.sin(2 * np.pi * (base_freq * 2) * t)
        signal += 0.2 * np.sin(2 * np.pi * (base_freq * 0.5) * t)
        
        # ROI-specific variation
        signal += 0.3 * np.sin(2 * np.pi * 0.08 * t + roi_idx * np.pi/8)
        
        # Physiological noise (cardiac ~1Hz, respiratory ~0.3Hz)
        signal += 0.1 * np.sin(2 * np.pi * 1.0 * t + roi_idx * np.pi/16)
        signal += 0.1 * np.sin(2 * np.pi * 0.3 * t + roi_idx * np.pi/12)
        
        # Thermal/scanner noise
        signal += 0.4 * np.random.randn(n_timepoints)
        
        data[roi_idx] = signal
    
    # Add known connectivity patterns with realistic hemodynamic delays
    known_connections = [
        # Default mode network connections
        ('PCC', 'mPFC', 2, 0.4),  # Strong DMN connection
        ('mPFC', 'Angular_L', 1, 0.3),
        ('Angular_L', 'Angular_R', 1, 0.35),  # Bilateral
        
        # Executive network
        ('DLPFC_L', 'DLPFC_R', 1, 0.3),  # Bilateral executive
        ('DLPFC_L', 'Parietal_L', 2, 0.25),  # Fronto-parietal
        
        # Motor network
        ('M1_L', 'M1_R', 1, 0.4),  # Bilateral motor
        ('M1_L', 'S1_L', 1, 0.3),  # Motor to sensory
        
        # Visual network
        ('V1_L', 'V1_R', 1, 0.35),  # Bilateral visual
        ('V1_L', 'V2_L', 1, 0.25),  # Visual hierarchy
        
        # Cross-network connections
        ('V1_L', 'DLPFC_L', 3, 0.2),  # Visual to attention
        ('DLPFC_L', 'PCC', 2, 0.15),  # Executive to default mode
        ('ACC', 'Insula', 1, 0.25),  # Salience network
        
        # Weak noise connections (should be filtered by enhanced methods)
        ('V2_R', 'Angular_R', 4, 0.1),  # Implausible long-range
        ('S1_R', 'mPFC', 5, 0.08),  # Weak implausible
    ]
    
    # Apply connectivity patterns
    roi_name_to_idx = {name: idx for idx, name in enumerate(roi_labels)}
    ground_truth = np.zeros((n_rois, n_rois))
    
    for source_name, target_name, lag, strength in known_connections:
        if source_name in roi_name_to_idx and target_name in roi_name_to_idx:
            source_idx = roi_name_to_idx[source_name]
            target_idx = roi_name_to_idx[target_name]
            
            if lag < n_timepoints:
                data[target_idx, lag:] += strength * data[source_idx, :-lag]
                ground_truth[source_idx, target_idx] = strength
    
    # Standardize data
    scaler = StandardScaler()
    data = scaler.fit_transform(data.T).T
    
    # Dataset metadata
    metadata = {
        'n_rois': n_rois,
        'n_timepoints': n_timepoints,
        'TR': TR,
        'roi_networks': roi_networks,
        'network_freqs': network_freqs,
        'known_connections': known_connections,
        'ground_truth': ground_truth,
        'scan_duration_minutes': n_timepoints * TR / 60
    }
    
    print(f"✅ Generated {n_rois} ROIs × {n_timepoints} timepoints")
    print(f"   Scan duration: {metadata['scan_duration_minutes']:.1f} minutes")
    print(f"   Known connections: {len(known_connections)}")
    
    return data, roi_labels, metadata

def run_quick_validation():
    """
    Run quick validation on realistic fMRI data.
    """
    
    print("🧠 QUICK REAL DATA VALIDATION")
    print("=" * 60)
    
    # Generate realistic data
    data, roi_labels, metadata = generate_realistic_fmri_data()
    ground_truth = metadata['ground_truth']
    roi_networks = metadata['roi_networks']
    
    # Define implementations to test
    implementations = {
        'baseline': VoxelSMTEConnectivity(
            n_symbols=6, ordinal_order=3, max_lag=5, n_permutations=100, random_state=42
        ),
        'adaptive': AdaptiveSMTE(
            adaptive_mode='heuristic', n_permutations=100, random_state=42
        ),
        'physiological': PhysiologicalSMTE(
            adaptive_mode='heuristic', use_network_correction=True,
            use_physiological_constraints=True, n_permutations=100, random_state=42
        ),
        'multiscale': MultiScaleSMTE(
            use_multiscale_analysis=True, scales_to_analyze=['fast', 'intermediate'],
            adaptive_mode='heuristic', use_network_correction=True,
            use_physiological_constraints=True, n_permutations=100, random_state=42
        ),
        'ensemble': EnsembleSMTE(
            use_ensemble_testing=True, surrogate_methods=['aaft'],
            n_surrogates_per_method=20, use_multiscale_analysis=True,
            scales_to_analyze=['fast'], adaptive_mode='heuristic',
            use_network_correction=True, use_physiological_constraints=True,
            n_permutations=100, random_state=42
        )
    }
    
    # Run validation
    results = {}
    
    for impl_name, impl in implementations.items():
        print(f"\\n🔬 Testing {impl_name.upper()}")
        print("-" * 40)
        
        try:
            start_time = time.time()
            
            # Run connectivity analysis
            if impl_name == 'baseline':
                impl.fmri_data = data
                impl.mask = np.ones(data.shape[0], dtype=bool)
                
                symbolic_data = impl.symbolize_timeseries(data)
                impl.symbolic_data = symbolic_data
                connectivity_matrix, lag_matrix = impl.compute_voxel_connectivity_matrix()
                p_values = impl.statistical_testing(connectivity_matrix)
                significance_mask = impl.fdr_correction(p_values)
                
            elif impl_name in ['adaptive', 'physiological']:
                analysis_results = impl.compute_adaptive_connectivity(data, roi_labels)
                connectivity_matrix = analysis_results['connectivity_matrix']
                significance_mask = analysis_results['significance_mask']
                
            elif impl_name == 'multiscale':
                analysis_results = impl.compute_multiscale_connectivity(data, roi_labels)
                connectivity_matrix = analysis_results['combined_connectivity']
                significance_mask = analysis_results['final_significance_mask']
                
            elif impl_name == 'ensemble':
                analysis_results = impl.compute_ensemble_connectivity(data, roi_labels)
                connectivity_matrix = analysis_results['combined_connectivity']
                significance_mask = analysis_results['final_significance_mask']
            
            computation_time = time.time() - start_time
            
            # Analyze results
            n_significant = np.sum(significance_mask)
            
            # Evaluate against ground truth
            evaluation = evaluate_connectivity_detection(
                connectivity_matrix, significance_mask, ground_truth, roi_networks
            )
            
            results[impl_name] = {
                'connectivity_matrix': connectivity_matrix,
                'significance_mask': significance_mask,
                'computation_time': computation_time,
                'n_significant': n_significant,
                'evaluation': evaluation,
                'success': True
            }
            
            print(f"  ✅ {n_significant} significant connections detected")
            print(f"     True positives: {evaluation['true_positives']}")
            print(f"     False positives: {evaluation['false_positives']}")
            print(f"     Precision: {evaluation['precision']:.3f}")
            print(f"     Recall: {evaluation['recall']:.3f}")
            print(f"     F1-score: {evaluation['f1_score']:.3f}")
            print(f"     Time: {computation_time:.2f}s")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results[impl_name] = {
                'success': False,
                'error': str(e),
                'computation_time': 0
            }
    
    # Create summary comparison
    summary_df = create_summary_table(results, metadata)
    
    # Create visualizations
    create_validation_plots(results, metadata)
    
    return results, summary_df, metadata

def evaluate_connectivity_detection(connectivity_matrix: np.ndarray,
                                  significance_mask: np.ndarray,
                                  ground_truth: np.ndarray,
                                  roi_networks: Dict[int, str]) -> Dict[str, float]:
    """
    Evaluate connectivity detection performance against ground truth.
    """
    
    # Create binary ground truth (any non-zero connection)
    true_connections = (ground_truth > 0.1).astype(int)  # Threshold weak connections
    
    # Flatten upper triangle (excluding diagonal)
    n_rois = connectivity_matrix.shape[0]
    triu_indices = np.triu_indices(n_rois, k=1)
    
    true_binary = true_connections[triu_indices]
    pred_binary = significance_mask[triu_indices].astype(int)
    
    # Compute metrics
    true_positives = np.sum((true_binary == 1) & (pred_binary == 1))
    false_positives = np.sum((true_binary == 0) & (pred_binary == 1))
    false_negatives = np.sum((true_binary == 1) & (pred_binary == 0))
    true_negatives = np.sum((true_binary == 0) & (pred_binary == 0))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    
    # Network-specific analysis
    within_network_tp = 0
    within_network_total = 0
    between_network_fp = 0
    between_network_total = 0
    
    networks = set(roi_networks.values())
    for i in range(n_rois):
        for j in range(i+1, n_rois):
            net_i = roi_networks.get(i, 'unknown')
            net_j = roi_networks.get(j, 'unknown')
            
            if net_i == net_j:  # Within network
                within_network_total += 1
                if true_connections[i, j] and significance_mask[i, j]:
                    within_network_tp += 1
            else:  # Between network
                between_network_total += 1
                if not true_connections[i, j] and significance_mask[i, j]:
                    between_network_fp += 1
    
    within_network_precision = within_network_tp / within_network_total if within_network_total > 0 else 0
    between_network_specificity = 1 - (between_network_fp / between_network_total) if between_network_total > 0 else 1
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'within_network_precision': within_network_precision,
        'between_network_specificity': between_network_specificity
    }

def create_summary_table(results: Dict[str, Any], metadata: Dict[str, Any]) -> pd.DataFrame:
    """
    Create summary comparison table.
    """
    
    summary_data = []
    
    for impl_name, impl_results in results.items():
        if impl_results['success']:
            eval_metrics = impl_results['evaluation']
            
            row = {
                'Implementation': impl_name.capitalize(),
                'Significant Connections': impl_results['n_significant'],
                'True Positives': eval_metrics['true_positives'],
                'False Positives': eval_metrics['false_positives'],
                'Precision': f"{eval_metrics['precision']:.3f}",
                'Recall': f"{eval_metrics['recall']:.3f}",
                'F1-Score': f"{eval_metrics['f1_score']:.3f}",
                'Computation Time (s)': f"{impl_results['computation_time']:.2f}",
                'Within-Network Precision': f"{eval_metrics['within_network_precision']:.3f}",
                'Between-Network Specificity': f"{eval_metrics['between_network_specificity']:.3f}"
            }
        else:
            row = {
                'Implementation': impl_name.capitalize(),
                'Significant Connections': 0,
                'True Positives': 0,
                'False Positives': 0,
                'Precision': '0.000',
                'Recall': '0.000',
                'F1-Score': '0.000',
                'Computation Time (s)': '0.00',
                'Within-Network Precision': '0.000',
                'Between-Network Specificity': '0.000'
            }
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)

def create_validation_plots(results: Dict[str, Any], metadata: Dict[str, Any]):
    """
    Create validation visualization plots.
    """
    
    # Extract successful results
    successful_results = {k: v for k, v in results.items() if v['success']}
    
    if not successful_results:
        print("No successful results to plot.")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Performance metrics comparison
    impl_names = list(successful_results.keys())
    f1_scores = [successful_results[impl]['evaluation']['f1_score'] for impl in impl_names]
    precisions = [successful_results[impl]['evaluation']['precision'] for impl in impl_names]
    recalls = [successful_results[impl]['evaluation']['recall'] for impl in impl_names]
    
    x_pos = np.arange(len(impl_names))
    width = 0.25
    
    axes[0, 0].bar(x_pos - width, f1_scores, width, label='F1-Score', alpha=0.8)
    axes[0, 0].bar(x_pos, precisions, width, label='Precision', alpha=0.8)
    axes[0, 0].bar(x_pos + width, recalls, width, label='Recall', alpha=0.8)
    
    axes[0, 0].set_xlabel('Implementation')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Connectivity Detection Performance')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels([name.capitalize() for name in impl_names], rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. True vs False positives
    true_positives = [successful_results[impl]['evaluation']['true_positives'] for impl in impl_names]
    false_positives = [successful_results[impl]['evaluation']['false_positives'] for impl in impl_names]
    
    axes[0, 1].bar(x_pos - width/2, true_positives, width, label='True Positives', alpha=0.8, color='green')
    axes[0, 1].bar(x_pos + width/2, false_positives, width, label='False Positives', alpha=0.8, color='red')
    
    axes[0, 1].set_xlabel('Implementation')
    axes[0, 1].set_ylabel('Number of Connections')
    axes[0, 1].set_title('True vs False Positive Detection')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels([name.capitalize() for name in impl_names], rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Computational efficiency
    comp_times = [successful_results[impl]['computation_time'] for impl in impl_names]
    
    bars = axes[0, 2].bar(impl_names, comp_times, alpha=0.8)
    axes[0, 2].set_xlabel('Implementation')
    axes[0, 2].set_ylabel('Computation Time (seconds)')
    axes[0, 2].set_title('Computational Efficiency')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Highlight fastest
    fastest_idx = np.argmin(comp_times)
    bars[fastest_idx].set_color('green')
    
    # 4. Network-specific performance
    within_net_precision = [successful_results[impl]['evaluation']['within_network_precision'] for impl in impl_names]
    between_net_specificity = [successful_results[impl]['evaluation']['between_network_specificity'] for impl in impl_names]
    
    axes[1, 0].bar(x_pos - width/2, within_net_precision, width, label='Within-Network Precision', alpha=0.8)
    axes[1, 0].bar(x_pos + width/2, between_net_specificity, width, label='Between-Network Specificity', alpha=0.8)
    
    axes[1, 0].set_xlabel('Implementation')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Network-Specific Performance')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([name.capitalize() for name in impl_names], rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Ground truth vs detected (for best performer)
    best_f1_idx = np.argmax(f1_scores)
    best_impl = impl_names[best_f1_idx]
    best_significance = successful_results[best_impl]['significance_mask']
    ground_truth = metadata['ground_truth']
    
    im1 = axes[1, 1].imshow(ground_truth > 0.1, cmap='Reds', alpha=0.8, aspect='auto')
    axes[1, 1].set_title(f'Ground Truth Connections')
    axes[1, 1].set_xlabel('Target ROI')
    axes[1, 1].set_ylabel('Source ROI')
    
    # 6. Best implementation results
    im2 = axes[1, 2].imshow(best_significance, cmap='Blues', alpha=0.8, aspect='auto')
    axes[1, 2].set_title(f'Detected Connections ({best_impl.capitalize()})')
    axes[1, 2].set_xlabel('Target ROI')
    axes[1, 2].set_ylabel('Source ROI')
    
    plt.tight_layout()
    plt.savefig('real_data_validation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\\n📊 Validation plots saved to: real_data_validation_results.png")

def main():
    """
    Run quick real data validation.
    """
    
    print("🚀 STARTING QUICK REAL DATA VALIDATION")
    print("=" * 80)
    
    # Run validation
    results, summary_df, metadata = run_quick_validation()
    
    # Display results
    print("\\n📋 VALIDATION SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    
    # Key findings
    successful_results = {k: v for k, v in results.items() if v['success']}
    
    if successful_results:
        print("\\n🎯 KEY FINDINGS")
        print("-" * 40)
        
        # Best performer by F1-score
        best_f1 = max(successful_results.keys(), 
                     key=lambda x: successful_results[x]['evaluation']['f1_score'])
        best_f1_score = successful_results[best_f1]['evaluation']['f1_score']
        
        print(f"Best overall performance: {best_f1.capitalize()} (F1: {best_f1_score:.3f})")
        
        # Most precise
        best_precision = max(successful_results.keys(),
                           key=lambda x: successful_results[x]['evaluation']['precision'])
        best_prec_score = successful_results[best_precision]['evaluation']['precision']
        
        print(f"Highest precision: {best_precision.capitalize()} ({best_prec_score:.3f})")
        
        # Fastest
        fastest = min(successful_results.keys(),
                     key=lambda x: successful_results[x]['computation_time'])
        fastest_time = successful_results[fastest]['computation_time']
        
        print(f"Fastest computation: {fastest.capitalize()} ({fastest_time:.2f}s)")
        
        # Network-specific insights
        print("\\n🧠 NETWORK-SPECIFIC INSIGHTS")
        print("-" * 40)
        
        for impl_name, impl_results in successful_results.items():
            eval_metrics = impl_results['evaluation']
            within_prec = eval_metrics['within_network_precision']
            between_spec = eval_metrics['between_network_specificity']
            
            print(f"{impl_name.capitalize()}:")
            print(f"  Within-network precision: {within_prec:.3f}")
            print(f"  Between-network specificity: {between_spec:.3f}")
    
    print("\\n✅ QUICK REAL DATA VALIDATION COMPLETE")
    print("=" * 80)
    
    return results, summary_df, metadata

if __name__ == "__main__":
    results, summary_df, metadata = main()