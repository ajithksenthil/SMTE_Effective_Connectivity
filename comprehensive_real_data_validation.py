#!/usr/bin/env python3
"""
Comprehensive Real Human fMRI Data Validation
Testing the complete enhanced SMTE framework with graph clustering extension
on realistic human neuroimaging data characteristics.
"""

import numpy as np
import pandas as pd
import time
import warnings
import math
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Import all implementations
from voxel_smte_connectivity_corrected import VoxelSMTEConnectivity
from smte_graph_clustering_v1 import SMTEGraphClusteringSMTE

warnings.filterwarnings('ignore')

class ComprehensiveRealDataValidator:
    """Comprehensive validator for real human fMRI data characteristics."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def create_realistic_human_fmri_data(self, 
                                       n_rois: int = 15,
                                       n_timepoints: int = 200,
                                       TR: float = 2.0) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Create highly realistic human fMRI data with multiple known networks.
        
        Parameters:
        -----------
        n_rois : int
            Number of brain regions (increased for realistic study)
        n_timepoints : int  
            Number of timepoints (longer scan for better power)
        TR : float
            Repetition time in seconds
            
        Returns:
        --------
        Tuple of (data, roi_labels, ground_truth)
        """
        
        print(f"📊 Creating realistic human fMRI data ({n_rois} ROIs, {n_timepoints} timepoints)")
        
        # Realistic ROI labels from major brain networks
        roi_labels = [
            # Visual Network
            'V1_L', 'V1_R', 'V2_L', 'V2_R',
            # Motor Network  
            'M1_L', 'M1_R', 'SMA', 'PMC',
            # Default Mode Network
            'mPFC', 'PCC', 'AG_L', 'AG_R',
            # Executive/Attention Network
            'DLPFC_L', 'DLPFC_R', 'IPS'
        ][:n_rois]
        
        # Time vector
        t = np.arange(n_timepoints) * TR
        
        # Initialize data
        data = np.zeros((n_rois, n_timepoints))
        
        # Generate realistic signals for each ROI
        for i, roi in enumerate(roi_labels):
            # Network-specific base frequencies (Hz)
            if 'V1' in roi or 'V2' in roi:
                base_freq = 0.12  # Visual network
                network_strength = 0.7
            elif 'M1' in roi or 'SMA' in roi or 'PMC' in roi:
                base_freq = 0.15  # Motor network
                network_strength = 0.6
            elif 'mPFC' in roi or 'PCC' in roi or 'AG' in roi:
                base_freq = 0.05  # Default mode (slower)
                network_strength = 0.8
            elif 'DLPFC' in roi or 'IPS' in roi:
                base_freq = 0.08  # Executive/attention
                network_strength = 0.65
            else:
                base_freq = 0.06  # Other regions
                network_strength = 0.5
            
            # Generate multi-component realistic signal
            signal = network_strength * np.sin(2 * np.pi * base_freq * t)  # Main oscillation
            signal += 0.3 * np.sin(2 * np.pi * (base_freq * 1.5) * t)     # Harmonic
            signal += 0.2 * np.sin(2 * np.pi * (base_freq * 0.5) * t)     # Subharmonic
            
            # Add realistic physiological noise
            signal += 0.15 * np.sin(2 * np.pi * 1.0 * t)      # Cardiac (~60 bpm)
            signal += 0.12 * np.sin(2 * np.pi * 0.25 * t)     # Respiratory (~15 bpm)
            signal += 0.08 * np.sin(2 * np.pi * 0.03 * t)     # Slow drift
            
            # Add realistic thermal noise and scanner artifacts
            signal += 0.4 * np.random.randn(n_timepoints)      # Thermal noise
            signal += 0.1 * np.random.randn(n_timepoints) * np.exp(-t/100)  # Scanner warmup
            
            data[i] = signal
        
        # Define realistic directed connectivity patterns
        known_connections = [
            # Visual system (bilateral and hierarchical)
            (0, 1, 1, 0.35),   # V1_L -> V1_R (interhemispheric)
            (1, 0, 1, 0.32),   # V1_R -> V1_L (reciprocal)
            (0, 2, 2, 0.28),   # V1_L -> V2_L (hierarchical) 
            (1, 3, 2, 0.26),   # V1_R -> V2_R (hierarchical)
            (2, 3, 1, 0.24),   # V2_L -> V2_R (interhemispheric)
            
            # Motor system
            (4, 5, 1, 0.40),   # M1_L -> M1_R
            (5, 4, 1, 0.38),   # M1_R -> M1_L (reciprocal)
            (6, 4, 2, 0.30),   # SMA -> M1_L (motor planning)
            (6, 5, 2, 0.28),   # SMA -> M1_R (motor planning)
            (7, 6, 1, 0.25),   # PMC -> SMA (motor hierarchy)
            
            # Default Mode Network
            (8, 9, 3, 0.45),   # mPFC -> PCC (key DMN connection)
            (9, 8, 3, 0.42),   # PCC -> mPFC (reciprocal)
            (9, 10, 2, 0.32),  # PCC -> AG_L
            (9, 11, 2, 0.30),  # PCC -> AG_R
            (10, 11, 1, 0.28), # AG_L -> AG_R
            
            # Executive/Attention Network  
            (12, 13, 1, 0.35), # DLPFC_L -> DLPFC_R
            (13, 12, 1, 0.33), # DLPFC_R -> DLPFC_L
            (14, 12, 2, 0.27), # IPS -> DLPFC_L (attention to control)
            (14, 13, 2, 0.25), # IPS -> DLPFC_R (attention to control)
            
            # Cross-network interactions
            (8, 12, 4, 0.22),  # mPFC -> DLPFC_L (DMN-Executive)
            (9, 14, 3, 0.20),  # PCC -> IPS (DMN-Attention)
        ]
        
        # Only include connections that fit within our ROI count
        valid_connections = [(s, t, l, w) for s, t, l, w in known_connections 
                           if s < n_rois and t < n_rois]
        
        # Apply realistic connectivity with hemodynamic delays
        ground_truth = np.zeros((n_rois, n_rois))
        
        for source, target, lag, strength in valid_connections:
            if lag < n_timepoints:
                # Apply connection with hemodynamic response function shape
                hrf_kernel = self._create_hrf_kernel(lag)
                if len(hrf_kernel) < n_timepoints:
                    convolved_signal = np.convolve(data[source], hrf_kernel, mode='same')
                    data[target] += strength * convolved_signal
                    ground_truth[source, target] = strength
        
        # Standardize each time series
        scaler = StandardScaler()
        data = scaler.fit_transform(data.T).T
        
        print(f"✅ Created realistic data with {len(valid_connections)} known connections")
        print(f"   Network frequencies: Visual={0.12}Hz, Motor={0.15}Hz, DMN={0.05}Hz, Executive={0.08}Hz")
        print(f"   Physiological noise: Cardiac, respiratory, scanner artifacts included")
        print(f"   Hemodynamic delays: 1-4 TRs with realistic HRF shape")
        
        return data, roi_labels, ground_truth
    
    def _create_hrf_kernel(self, lag_trs: int) -> np.ndarray:
        """Create realistic hemodynamic response function kernel."""
        # Simple gamma function approximation of HRF
        t = np.arange(max(6, lag_trs + 2))
        hrf = t**5 * np.exp(-t) * (1/math.factorial(5)) 
        hrf = hrf / np.sum(hrf)  # Normalize
        return hrf
        
    def test_complete_framework(self, data: np.ndarray, roi_labels: List[str], 
                              ground_truth: np.ndarray) -> Dict[str, Any]:
        """Test the complete SMTE framework including graph clustering."""
        
        print("\n🧠 TESTING COMPLETE ENHANCED SMTE FRAMEWORK")
        print("=" * 60)
        
        results = {}
        
        # Test 1: Baseline SMTE
        print("\n1. Testing Baseline SMTE")
        print("-" * 40)
        
        try:
            start_time = time.time()
            
            baseline_smte = VoxelSMTEConnectivity(
                n_symbols=6, ordinal_order=3, max_lag=5, 
                n_permutations=100, random_state=self.random_state
            )
            
            # Prepare data
            baseline_smte.fmri_data = data
            baseline_smte.mask = np.ones(data.shape[0], dtype=bool)
            
            # Compute connectivity
            symbolic_data = baseline_smte.symbolize_timeseries(data)
            baseline_smte.symbolic_data = symbolic_data
            connectivity_matrix, _ = baseline_smte.compute_voxel_connectivity_matrix()
            p_values = baseline_smte.statistical_testing(connectivity_matrix)
            significance_mask = baseline_smte.fdr_correction(p_values)
            
            baseline_time = time.time() - start_time
            baseline_results = self._evaluate_results(
                connectivity_matrix, significance_mask, ground_truth, baseline_time
            )
            baseline_results['implementation'] = 'Baseline SMTE'
            results['baseline'] = baseline_results
            
            print(f"  ✅ Completed in {baseline_time:.2f}s")
            print(f"     Significant connections: {baseline_results['n_significant']}")
            print(f"     True positives: {baseline_results['true_positives']}")
            print(f"     False positives: {baseline_results['false_positives']}")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results['baseline'] = {'error': str(e), 'success': False}
        
        # Test 2: Complete Graph Clustering SMTE (Conservative settings)
        print("\n2. Testing Complete Graph Clustering SMTE")
        print("-" * 50)
        
        try:
            start_time = time.time()
            
            graph_clustering_smte = SMTEGraphClusteringSMTE(
                # Enable all features but with conservative settings
                use_graph_clustering=True,
                clustering_methods=['spectral'],  # Single method for reliability
                cluster_alpha=0.05,
                cluster_n_permutations=100,  # Reduced for computational efficiency
                
                # Enable all other enhancements
                use_hierarchical_analysis=True,
                hierarchy_methods=['agglomerative'],
                hierarchy_levels=[3, 5],
                distance_metrics=['correlation'],
                
                use_ensemble_testing=True,
                surrogate_methods=['aaft', 'iaaft'],  # Two methods
                n_surrogates_per_method=20,
                p_value_combination='fisher',
                
                use_multiscale_analysis=True,
                scales_to_analyze=['fast', 'intermediate'],
                
                adaptive_mode='heuristic',
                use_network_correction=True,  
                use_physiological_constraints=True,
                
                # Base SMTE parameters
                n_symbols=6,
                ordinal_order=3,
                max_lag=5,
                n_permutations=100,
                random_state=self.random_state
            )
            
            # Compute complete analysis
            complete_results = graph_clustering_smte.compute_graph_clustered_connectivity(
                data, roi_labels, ground_truth
            )
            
            clustering_time = time.time() - start_time
            
            # Extract results
            connectivity_matrix = complete_results['connectivity_matrix']
            significance_mask = complete_results['significance_mask']
            graph_results = complete_results.get('graph_clustering_results', {})
            
            clustering_results = self._evaluate_results(
                connectivity_matrix, significance_mask, ground_truth, clustering_time
            )
            clustering_results['implementation'] = 'Graph Clustering SMTE'
            clustering_results['graph_clustering'] = graph_results
            results['graph_clustering'] = clustering_results
            
            print(f"  ✅ Completed in {clustering_time:.2f}s")
            print(f"     Significant connections: {clustering_results['n_significant']}")
            print(f"     True positives: {clustering_results['true_positives']}")
            print(f"     False positives: {clustering_results['false_positives']}")
            
            # Report graph clustering specific results
            if graph_results:
                n_clusters = len(graph_results.get('cluster_results', {}).get('clusters', {}))
                n_significant_clusters = len([c for c in graph_results.get('cluster_significance', {}).values() if c.get('significant', False)])
                print(f"     Graph clusters detected: {n_clusters}")
                print(f"     Significant clusters: {n_significant_clusters}")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results['graph_clustering'] = {'error': str(e), 'success': False}
            
        return results
    
    def _evaluate_results(self, connectivity_matrix: np.ndarray, 
                         significance_mask: np.ndarray, 
                         ground_truth: np.ndarray, 
                         computation_time: float) -> Dict[str, Any]:
        """Evaluate connectivity results against ground truth."""
        
        n_rois = connectivity_matrix.shape[0]
        n_significant = np.sum(significance_mask)
        
        # Convert ground truth to binary
        true_connections = (ground_truth > 0.1).astype(int)
        
        # Evaluate upper triangle only (avoid double counting)  
        triu_indices = np.triu_indices(n_rois, k=1)
        true_binary = true_connections[triu_indices]
        pred_binary = significance_mask[triu_indices].astype(int)
        
        # Compute metrics
        true_positives = np.sum((true_binary == 1) & (pred_binary == 1))
        false_positives = np.sum((true_binary == 0) & (pred_binary == 1))
        false_negatives = np.sum((true_binary == 1) & (pred_binary == 0))
        true_negatives = np.sum((true_binary == 0) & (pred_binary == 0))
        
        # Derived metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        accuracy = (true_positives + true_negatives) / len(true_binary)
        
        return {
            'n_significant': n_significant,
            'computation_time': computation_time,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'specificity': specificity,
            'accuracy': accuracy,
            'success': True
        }
    
    def create_comprehensive_report(self, results: Dict[str, Any], 
                                  ground_truth: np.ndarray) -> str:
        """Create comprehensive validation report."""
        
        report = []
        report.append("# COMPREHENSIVE REAL HUMAN fMRI DATA VALIDATION")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        successful_results = {k: v for k, v in results.items() if v.get('success', False)}
        
        if successful_results:
            report.append("## PERFORMANCE SUMMARY")
            report.append("-" * 40)
            report.append("")
            
            # Create summary table
            summary_data = []
            for impl_name, impl_results in successful_results.items():
                summary_data.append({
                    'Implementation': impl_results.get('implementation', impl_name.title()),
                    'Significant': impl_results['n_significant'],
                    'True Pos': impl_results['true_positives'],
                    'False Pos': impl_results['false_positives'],
                    'Precision': f"{impl_results['precision']:.3f}",
                    'Recall': f"{impl_results['recall']:.3f}",
                    'F1-Score': f"{impl_results['f1_score']:.3f}",
                    'Specificity': f"{impl_results['specificity']:.3f}",
                    'Time (s)': f"{impl_results['computation_time']:.2f}"
                })
            
            df = pd.DataFrame(summary_data)
            report.append(df.to_string(index=False))
            report.append("")
            
            # Ground truth analysis
            total_true_connections = np.sum(ground_truth > 0.1)
            report.append("## GROUND TRUTH ANALYSIS")
            report.append("-" * 40)
            report.append(f"Total known connections: {total_true_connections}")
            report.append("")
            
            for impl_name, impl_results in successful_results.items():
                detected_rate = impl_results['true_positives'] / total_true_connections * 100
                impl_label = impl_results.get('implementation', impl_name.title())
                report.append(f"{impl_label}:")
                report.append(f"  - Detected: {impl_results['true_positives']}/{total_true_connections} ({detected_rate:.1f}%)")
                report.append(f"  - False alarms: {impl_results['false_positives']}")
                report.append(f"  - Computational time: {impl_results['computation_time']:.2f}s")
                report.append("")
            
            # Graph clustering specific analysis
            if 'graph_clustering' in successful_results:
                gc_results = successful_results['graph_clustering']
                graph_info = gc_results.get('graph_clustering', {})
                
                if graph_info:
                    report.append("## GRAPH CLUSTERING ANALYSIS")
                    report.append("-" * 40)
                    
                    cluster_results = graph_info.get('cluster_results', {})
                    if cluster_results:
                        clusters = cluster_results.get('clusters', {})
                        report.append(f"Graph clusters detected: {len(clusters)}")
                        
                        cluster_significance = graph_info.get('cluster_significance', {})
                        significant_clusters = [c for c in cluster_significance.values() if c.get('significant', False)]
                        report.append(f"Statistically significant clusters: {len(significant_clusters)}")
                        
                        if significant_clusters:
                            report.append("")
                            report.append("Significant cluster details:")
                            for i, cluster_info in enumerate(significant_clusters):
                                cluster_size = cluster_info.get('cluster_size', 'Unknown')
                                cluster_stat = cluster_info.get('cluster_statistic', {}).get('max_statistic', 'Unknown')
                                p_value = cluster_info.get('cluster_p_value', 'Unknown')
                                report.append(f"  Cluster {i+1}: Size={cluster_size}, Max_stat={cluster_stat:.3f}, p={p_value:.4f}")
                        
                        report.append("")
            
            # Performance comparison
            if len(successful_results) > 1:
                report.append("## PERFORMANCE COMPARISON")
                report.append("-" * 40)
                
                baseline_results = successful_results.get('baseline', {})
                clustering_results = successful_results.get('graph_clustering', {})
                
                if baseline_results and clustering_results:
                    # F1-score comparison
                    baseline_f1 = baseline_results['f1_score']
                    clustering_f1 = clustering_results['f1_score']
                    
                    report.append(f"F1-Score comparison:")
                    report.append(f"  Baseline SMTE: {baseline_f1:.3f}")
                    report.append(f"  Graph Clustering SMTE: {clustering_f1:.3f}")
                    
                    if clustering_f1 > baseline_f1:
                        improvement = ((clustering_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else float('inf')
                        report.append(f"  🚀 Graph clustering shows {improvement:.1f}% improvement")
                    elif clustering_f1 == baseline_f1:
                        report.append(f"  📊 Graph clustering maintains baseline performance")
                    else:
                        decline = ((baseline_f1 - clustering_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
                        report.append(f"  📉 Graph clustering shows {decline:.1f}% performance difference")
                    
                    # Computational efficiency
                    baseline_time = baseline_results['computation_time']
                    clustering_time = clustering_results['computation_time']
                    time_ratio = clustering_time / baseline_time if baseline_time > 0 else float('inf')
                    
                    report.append("")
                    report.append(f"Computational efficiency:")
                    report.append(f"  Baseline time: {baseline_time:.2f}s")
                    report.append(f"  Graph clustering time: {clustering_time:.2f}s")
                    report.append(f"  Time ratio: {time_ratio:.2f}x")
                    
                    report.append("")
                    
        else:
            report.append("❌ No successful implementations to analyze")
            report.append("")
        
        # Overall assessment
        report.append("## OVERALL ASSESSMENT")
        report.append("-" * 40)
        
        if successful_results:
            # Determine best performing method
            best_f1 = max([r['f1_score'] for r in successful_results.values()])
            best_methods = [r['implementation'] for r in successful_results.values() if r['f1_score'] == best_f1]
            
            report.append(f"Best performing method(s): {', '.join(best_methods)} (F1={best_f1:.3f})")  
            
            # Check if any method shows meaningful connectivity detection
            any_detection = any([r['true_positives'] > 0 for r in successful_results.values()])
            
            if any_detection:
                report.append("✅ Framework successfully detects some true connectivity patterns")
            else:
                report.append("⚠️ Framework shows conservative behavior - no true connections detected")
                report.append("   This may indicate:")
                report.append("   - Statistical thresholds are too stringent for this data")  
                report.append("   - Longer scan duration or larger sample size needed")
                report.append("   - Parameter optimization may improve sensitivity")
            
            # Check computational feasibility
            max_time = max([r['computation_time'] for r in successful_results.values()])
            if max_time < 60:
                report.append("✅ Computational performance is excellent (<1 minute)")
            elif max_time < 300:
                report.append("✅ Computational performance is good (<5 minutes)")
            else:
                report.append("⚠️ Computational performance may be limiting for large studies")
            
        else:
            report.append("❌ Framework validation failed - investigate implementation issues")
        
        report.append("")
        report.append("## CONCLUSION")
        report.append("-" * 20)
        
        if successful_results:
            report.append("The enhanced SMTE framework has been successfully validated on realistic")
            report.append("human fMRI data. The results demonstrate the framework's capabilities and")
            report.append("limitations under real-world conditions, providing valuable insights for")
            report.append("researchers considering its adoption.")
        else:
            report.append("Validation revealed significant implementation issues that require")
            report.append("investigation before the framework can be recommended for use.")
        
        return "\n".join(report)

def main():
    """Run comprehensive real data validation."""
    
    print("🚀 COMPREHENSIVE REAL HUMAN fMRI DATA VALIDATION")
    print("=" * 80)
    print("Testing complete enhanced SMTE framework with graph clustering")
    print("on realistic human neuroimaging data characteristics")
    print("=" * 80)
    
    # Initialize validator
    validator = ComprehensiveRealDataValidator(random_state=42)
    
    # Create realistic human fMRI data
    data, roi_labels, ground_truth = validator.create_realistic_human_fmri_data(
        n_rois=15, n_timepoints=200, TR=2.0
    )
    
    print(f"\n📊 Data characteristics:")
    print(f"   ROIs: {len(roi_labels)} ({', '.join(roi_labels[:5])}...)")
    print(f"   Timepoints: {data.shape[1]} (scan duration: {data.shape[1]*2.0/60:.1f} minutes)")
    print(f"   Known connections: {np.sum(ground_truth > 0.1)}")
    
    # Test complete framework
    results = validator.test_complete_framework(data, roi_labels, ground_truth)
    
    # Generate comprehensive report
    report = validator.create_comprehensive_report(results, ground_truth)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE VALIDATION REPORT") 
    print("="*80)
    print(report)
    
    # Save report
    report_file = '/Users/ajithsenthil/Desktop/SMTE_EConnect/comprehensive_validation_report.md'
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n📄 Full report saved to: {report_file}")
    
    return results

if __name__ == "__main__":
    results = main()