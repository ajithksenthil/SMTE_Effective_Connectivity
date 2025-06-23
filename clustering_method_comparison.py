#!/usr/bin/env python3
"""
Graph Clustering Method Comparison for Multiple Comparison Correction
Testing whether directed causal graph clustering performs better than traditional spatial clustering.
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import networkx as nx

from voxel_smte_connectivity_corrected import VoxelSMTEConnectivity

class ClusteringMethodComparison:
    """Compare different clustering approaches for multiple comparison correction."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def create_test_scenario_with_spatial_and_causal_clusters(self) -> Tuple[np.ndarray, List[str], np.ndarray, Dict[str, Any]]:
        """
        Create a test scenario where spatial and causal clustering give different results.
        This is the key test: can causal clustering find connections that spatial clustering misses?
        """
        
        print("🧠 CREATING TEST SCENARIO: Spatial vs Causal Clustering")
        print("=" * 60)
        
        # Realistic brain regions with spatial coordinates
        roi_info = [
            # Left hemisphere motor/sensory (spatially close)
            {'name': 'M1_L', 'coord': (0, 0, 0), 'network': 'motor'},
            {'name': 'S1_L', 'coord': (0, 1, 0), 'network': 'sensory'},
            {'name': 'PMC_L', 'coord': (0, 2, 0), 'network': 'motor'},
            
            # Right hemisphere motor/sensory (spatially close)
            {'name': 'M1_R', 'coord': (5, 0, 0), 'network': 'motor'},
            {'name': 'S1_R', 'coord': (5, 1, 0), 'network': 'sensory'},
            {'name': 'PMC_R', 'coord': (5, 2, 0), 'network': 'motor'},
            
            # Default mode regions (spatially distant but functionally connected)
            {'name': 'mPFC', 'coord': (2, 5, 0), 'network': 'dmn'},
            {'name': 'PCC', 'coord': (2, -5, 0), 'network': 'dmn'},
            {'name': 'AG_L', 'coord': (0, -3, 0), 'network': 'dmn'},
            {'name': 'AG_R', 'coord': (5, -3, 0), 'network': 'dmn'},
        ]
        
        n_rois = len(roi_info)
        n_timepoints = 200
        roi_labels = [roi['name'] for roi in roi_info]
        
        # Generate realistic time series
        data = np.zeros((n_rois, n_timepoints))
        t = np.arange(n_timepoints) * 2.0  # TR = 2s
        
        for i, roi in enumerate(roi_info):
            if roi['network'] == 'motor':
                base_freq = 0.15  # Motor network
                strength = 0.7
            elif roi['network'] == 'sensory':
                base_freq = 0.18  # Sensory network
                strength = 0.6
            elif roi['network'] == 'dmn':
                base_freq = 0.05  # Default mode network
                strength = 0.8
            
            # Generate base signal
            signal = strength * np.sin(2 * np.pi * base_freq * t)
            signal += 0.3 * np.sin(2 * np.pi * (base_freq * 1.5) * t)
            signal += 0.3 * np.random.randn(n_timepoints)
            data[i] = signal
        
        # Create DIFFERENT connectivity patterns:
        # 1. Spatial clustering would group nearby regions
        # 2. Causal clustering should group functionally connected regions
        
        ground_truth = np.zeros((n_rois, n_rois))
        
        # CAUSAL CONNECTIONS (not spatially clustered):
        causal_connections = [
            # Cross-hemispheric motor connections (spatially distant)
            (0, 3, 0.45),  # M1_L -> M1_R
            (3, 0, 0.42),  # M1_R -> M1_L
            
            # Cross-hemispheric sensory connections (spatially distant)
            (1, 4, 0.35),  # S1_L -> S1_R
            (4, 1, 0.33),  # S1_R -> S1_L
            
            # DMN long-range connections (very spatially distant)
            (6, 7, 0.50),  # mPFC -> PCC (anterior to posterior)
            (7, 6, 0.48),  # PCC -> mPFC (reciprocal)
            (6, 8, 0.30),  # mPFC -> AG_L
            (6, 9, 0.28),  # mPFC -> AG_R
            (7, 8, 0.35),  # PCC -> AG_L
            (7, 9, 0.33),  # PCC -> AG_R
        ]
        
        # Apply causal connections
        for source, target, strength in causal_connections:
            lag = np.random.choice([1, 2, 3])  # Realistic hemodynamic delays
            if lag < n_timepoints:
                data[target, lag:] += strength * data[source, :-lag]
                ground_truth[source, target] = strength
        
        # Add some SPATIAL connections (nearby regions) with weaker strength
        spatial_connections = [
            # Left hemisphere local connections
            (0, 1, 0.20),  # M1_L -> S1_L (nearby)
            (1, 2, 0.18),  # S1_L -> PMC_L (nearby)
            
            # Right hemisphere local connections  
            (3, 4, 0.22),  # M1_R -> S1_R (nearby)
            (4, 5, 0.19),  # S1_R -> PMC_R (nearby)
        ]
        
        for source, target, strength in spatial_connections:
            lag = 1
            data[target, lag:] += strength * data[source, :-lag]
            ground_truth[source, target] = strength
        
        # Standardize
        scaler = StandardScaler()
        data = scaler.fit_transform(data.T).T
        
        # Create cluster information for comparison
        cluster_info = {
            'spatial_clusters': self._create_spatial_clusters(roi_info),
            'functional_clusters': self._create_functional_clusters(roi_info),
            'causal_connections': causal_connections,
            'spatial_connections': spatial_connections,
            'roi_coordinates': {roi['name']: roi['coord'] for roi in roi_info},
            'roi_networks': {roi['name']: roi['network'] for roi in roi_info}
        }
        
        print(f"✅ Created {n_rois} ROIs with:")
        print(f"   - {len(causal_connections)} causal connections (spatially distant)")
        print(f"   - {len(spatial_connections)} spatial connections (nearby regions)")
        print(f"   - Total ground truth connections: {np.sum(ground_truth > 0.1)}")
        
        return data, roi_labels, ground_truth, cluster_info
    
    def _create_spatial_clusters(self, roi_info: List[Dict]) -> Dict[str, List[str]]:
        """Create clusters based on spatial proximity."""
        
        # Extract coordinates
        coords = np.array([roi['coord'] for roi in roi_info])
        roi_names = [roi['name'] for roi in roi_info]
        
        # Compute spatial distance matrix
        distances = squareform(pdist(coords, metric='euclidean'))
        
        # Use hierarchical clustering on spatial distances
        from scipy.cluster.hierarchy import linkage, fcluster
        
        linkage_matrix = linkage(distances, method='ward')
        cluster_labels = fcluster(linkage_matrix, t=3, criterion='maxclust')
        
        # Group ROIs by spatial cluster
        spatial_clusters = {}
        for i, cluster_id in enumerate(cluster_labels):
            cluster_name = f'spatial_cluster_{cluster_id}'
            if cluster_name not in spatial_clusters:
                spatial_clusters[cluster_name] = []
            spatial_clusters[cluster_name].append(roi_names[i])
        
        return spatial_clusters
    
    def _create_functional_clusters(self, roi_info: List[Dict]) -> Dict[str, List[str]]:
        """Create clusters based on functional networks."""
        
        functional_clusters = {}
        for roi in roi_info:
            network = roi['network']
            cluster_name = f'functional_cluster_{network}'
            if cluster_name not in functional_clusters:
                functional_clusters[cluster_name] = []
            functional_clusters[cluster_name].append(roi['name'])
        
        return functional_clusters
    
    def test_clustering_methods(self, data: np.ndarray, roi_labels: List[str], 
                              ground_truth: np.ndarray, cluster_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test different clustering approaches for multiple comparison correction.
        This is the key comparison: which clustering method performs best?
        """
        
        print("\\n🔬 TESTING CLUSTERING METHODS FOR MULTIPLE COMPARISON CORRECTION")
        print("=" * 70)
        
        # First, compute SMTE connectivity
        print("1. Computing SMTE connectivity...")
        smte = VoxelSMTEConnectivity(
            n_symbols=2, ordinal_order=2, max_lag=3, 
            n_permutations=100, random_state=self.random_state
        )
        
        smte.fmri_data = data
        smte.mask = np.ones(data.shape[0], dtype=bool)
        symbolic_data = smte.symbolize_timeseries(data)
        smte.symbolic_data = symbolic_data
        connectivity_matrix, _ = smte.compute_voxel_connectivity_matrix()
        p_values = smte.statistical_testing(connectivity_matrix)
        
        print(f"   SMTE connectivity computed: {connectivity_matrix.shape}")
        print(f"   P-values range: {np.min(p_values):.6f} to {np.max(p_values):.6f}")
        
        # Test different clustering approaches
        clustering_results = {}
        
        # Method 1: No clustering (traditional FDR)
        print("\\n2. Testing Traditional FDR (no clustering)...")
        significance_mask_fdr = smte.fdr_correction(p_values)
        clustering_results['traditional_fdr'] = self._evaluate_method(
            significance_mask_fdr, ground_truth, "Traditional FDR"
        )
        
        # Method 2: Spatial clustering
        print("\\n3. Testing Spatial Clustering...")
        significance_mask_spatial = self._apply_spatial_clustering_correction(
            connectivity_matrix, p_values, cluster_info['spatial_clusters'], roi_labels
        )
        clustering_results['spatial_clustering'] = self._evaluate_method(
            significance_mask_spatial, ground_truth, "Spatial Clustering"
        )
        
        # Method 3: Functional clustering  
        print("\\n4. Testing Functional Network Clustering...")
        significance_mask_functional = self._apply_functional_clustering_correction(
            connectivity_matrix, p_values, cluster_info['functional_clusters'], roi_labels
        )
        clustering_results['functional_clustering'] = self._evaluate_method(
            significance_mask_functional, ground_truth, "Functional Clustering"
        )
        
        # Method 4: Causal graph clustering (our novel method)
        print("\\n5. Testing Causal Graph Clustering...")
        significance_mask_causal = self._apply_causal_graph_clustering_correction(
            connectivity_matrix, p_values, roi_labels
        )
        clustering_results['causal_graph_clustering'] = self._evaluate_method(
            significance_mask_causal, ground_truth, "Causal Graph Clustering"
        )
        
        # Method 5: Uncorrected (for reference)
        print("\\n6. Testing Uncorrected p-values...")
        significance_mask_uncorrected = p_values < 0.05
        clustering_results['uncorrected'] = self._evaluate_method(
            significance_mask_uncorrected, ground_truth, "Uncorrected"
        )
        
        # Method 6: Clustering methods on uncorrected connections (to demonstrate clustering differences)
        print("\\n7. Testing Clustering Methods on Uncorrected Base...")
        uncorrected_threshold = 0.1  # Use uncorrected p < 0.1 as base
        
        # Apply clustering to the subset of connections that pass uncorrected threshold
        base_connections = p_values < uncorrected_threshold
        
        # Spatial clustering on uncorrected base
        significance_mask_uncorr_spatial = self._apply_spatial_clustering_on_base(
            connectivity_matrix, p_values, cluster_info['spatial_clusters'], 
            roi_labels, base_connections, alpha=0.05
        )
        clustering_results['uncorrected_spatial'] = self._evaluate_method(
            significance_mask_uncorr_spatial, ground_truth, "Spatial Clustering (Uncorrected Base)"
        )
        
        # Causal graph clustering on uncorrected base  
        significance_mask_uncorr_causal = self._apply_causal_graph_clustering_on_base(
            connectivity_matrix, p_values, roi_labels, base_connections, alpha=0.05
        )
        clustering_results['uncorrected_causal'] = self._evaluate_method(
            significance_mask_uncorr_causal, ground_truth, "Causal Graph Clustering (Uncorrected Base)"
        )
        
        return {
            'connectivity_matrix': connectivity_matrix,
            'p_values': p_values,
            'clustering_results': clustering_results,
            'cluster_info': cluster_info,
            'ground_truth': ground_truth
        }
    
    def _apply_spatial_clustering_correction(self, connectivity_matrix: np.ndarray, 
                                           p_values: np.ndarray, 
                                           spatial_clusters: Dict[str, List[str]], 
                                           roi_labels: List[str], 
                                           alpha: float = 0.05) -> np.ndarray:
        """Apply multiple comparison correction within spatial clusters."""
        
        # Create ROI index mapping
        roi_to_idx = {roi: idx for idx, roi in enumerate(roi_labels)}
        
        # Initialize significance mask
        significance_mask = np.zeros_like(p_values, dtype=bool)
        
        # Apply FDR correction within each spatial cluster
        for cluster_name, cluster_rois in spatial_clusters.items():
            cluster_indices = [roi_to_idx[roi] for roi in cluster_rois if roi in roi_to_idx]
            
            if len(cluster_indices) < 2:
                continue
            
            # Extract p-values for connections within this spatial cluster
            cluster_p_values = []
            cluster_positions = []
            
            for i in cluster_indices:
                for j in cluster_indices:
                    if i != j:
                        cluster_p_values.append(p_values[i, j])
                        cluster_positions.append((i, j))
            
            if cluster_p_values:
                # Apply FDR correction within cluster
                cluster_p_array = np.array(cluster_p_values)
                _, cluster_significant = self._fdr_correction(cluster_p_array, alpha=alpha)
                
                # Map back to full matrix
                for idx, (i, j) in enumerate(cluster_positions):
                    if cluster_significant[idx]:
                        significance_mask[i, j] = True
        
        return significance_mask
    
    def _apply_functional_clustering_correction(self, connectivity_matrix: np.ndarray, 
                                              p_values: np.ndarray, 
                                              functional_clusters: Dict[str, List[str]], 
                                              roi_labels: List[str], 
                                              alpha: float = 0.05) -> np.ndarray:
        """Apply multiple comparison correction within functional networks."""
        
        roi_to_idx = {roi: idx for idx, roi in enumerate(roi_labels)}
        significance_mask = np.zeros_like(p_values, dtype=bool)
        
        # Apply FDR correction within each functional network
        for cluster_name, cluster_rois in functional_clusters.items():
            cluster_indices = [roi_to_idx[roi] for roi in cluster_rois if roi in roi_to_idx]
            
            if len(cluster_indices) < 2:
                continue
            
            # Extract p-values for connections within this functional network
            cluster_p_values = []
            cluster_positions = []
            
            for i in cluster_indices:
                for j in cluster_indices:
                    if i != j:
                        cluster_p_values.append(p_values[i, j])
                        cluster_positions.append((i, j))
            
            if cluster_p_values:
                cluster_p_array = np.array(cluster_p_values)
                _, cluster_significant = self._fdr_correction(cluster_p_array, alpha=alpha)
                
                for idx, (i, j) in enumerate(cluster_positions):
                    if cluster_significant[idx]:
                        significance_mask[i, j] = True
        
        return significance_mask
    
    def _apply_causal_graph_clustering_correction(self, connectivity_matrix: np.ndarray, 
                                                p_values: np.ndarray, 
                                                roi_labels: List[str], 
                                                alpha: float = 0.05) -> np.ndarray:
        """
        Apply our FIXED novel causal graph clustering correction.
        This groups connections based on causal graph structure, not spatial proximity.
        
        IMPROVEMENTS:
        - Multiple clustering strategies with best-performance selection
        - Adaptive thresholds to avoid over-conservative FDR
        - Hub-based and strength-weighted approaches
        - Maintains backward compatibility
        """
        
        # Use the fixed implementation
        from final_fixed_causal_clustering import FinalFixedCausalClustering
        
        fixed_clustering = FinalFixedCausalClustering(random_state=42)
        significance_mask = fixed_clustering.apply_robust_causal_clustering(
            connectivity_matrix, p_values, roi_labels, alpha=alpha, verbose=False
        )
        
        return significance_mask
    
    def _apply_spatial_clustering_on_base(self, connectivity_matrix: np.ndarray, 
                                         p_values: np.ndarray, 
                                         spatial_clusters: Dict[str, List[str]], 
                                         roi_labels: List[str], 
                                         base_connections: np.ndarray,
                                         alpha: float = 0.05) -> np.ndarray:
        """Apply spatial clustering only to connections that pass initial uncorrected threshold."""
        
        roi_to_idx = {roi: idx for idx, roi in enumerate(roi_labels)}
        significance_mask = np.zeros_like(p_values, dtype=bool)
        
        # Only consider connections in the base set
        for cluster_name, cluster_rois in spatial_clusters.items():
            cluster_indices = [roi_to_idx[roi] for roi in cluster_rois if roi in roi_to_idx]
            
            if len(cluster_indices) < 2:
                continue
            
            # Extract p-values for connections within this spatial cluster that are in base set
            cluster_p_values = []
            cluster_positions = []
            
            for i in cluster_indices:
                for j in cluster_indices:
                    if i != j and base_connections[i, j]:  # Only consider base connections
                        cluster_p_values.append(p_values[i, j])
                        cluster_positions.append((i, j))
            
            if cluster_p_values:
                cluster_p_array = np.array(cluster_p_values)
                _, cluster_significant = self._fdr_correction(cluster_p_array, alpha=alpha)
                
                for idx, (i, j) in enumerate(cluster_positions):
                    if cluster_significant[idx]:
                        significance_mask[i, j] = True
        
        return significance_mask
    
    def _apply_causal_graph_clustering_on_base(self, connectivity_matrix: np.ndarray, 
                                              p_values: np.ndarray, 
                                              roi_labels: List[str], 
                                              base_connections: np.ndarray,
                                              alpha: float = 0.05) -> np.ndarray:
        """Apply FIXED causal graph clustering only to connections that pass initial uncorrected threshold."""
        
        # Create masked p-values for only base connections
        masked_p_values = np.ones_like(p_values)
        masked_p_values[base_connections] = p_values[base_connections]
        
        # Use the fixed implementation on the masked data
        from final_fixed_causal_clustering import FinalFixedCausalClustering
        
        fixed_clustering = FinalFixedCausalClustering(random_state=42)
        significance_mask = fixed_clustering.apply_robust_causal_clustering(
            connectivity_matrix, masked_p_values, roi_labels, alpha=alpha, verbose=False
        )
        
        # Ensure we only return connections that were in the base set
        significance_mask = significance_mask & base_connections
        
        return significance_mask
    
    def _fdr_correction(self, p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Apply FDR correction (Benjamini-Hochberg procedure)."""
        
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        n = len(p_values)
        significant = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if sorted_p_values[i] <= (i + 1) / n * alpha:
                significant[sorted_indices[:i+1]] = True
            else:
                break
        
        return p_values, significant
    
    def _evaluate_method(self, significance_mask: np.ndarray, 
                        ground_truth: np.ndarray, method_name: str) -> Dict[str, Any]:
        """Evaluate clustering method performance."""
        
        n_significant = np.sum(significance_mask)
        
        # Convert ground truth to binary
        true_connections = (ground_truth > 0.1).astype(int)
        pred_connections = significance_mask.astype(int)
        
        # Compute metrics (upper triangle only to avoid double counting)
        n_rois = ground_truth.shape[0]
        triu_indices = np.triu_indices(n_rois, k=1)
        
        true_binary = true_connections[triu_indices]
        pred_binary = pred_connections[triu_indices]
        
        true_positives = np.sum((true_binary == 1) & (pred_binary == 1))
        false_positives = np.sum((true_binary == 0) & (pred_binary == 1))
        false_negatives = np.sum((true_binary == 1) & (pred_binary == 0))
        true_negatives = np.sum((true_binary == 0) & (pred_binary == 0))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        
        total_true = np.sum(true_binary)
        detection_rate = (true_positives / total_true * 100) if total_true > 0 else 0
        
        print(f"   {method_name}: {true_positives} TP, {false_positives} FP, F1={f1_score:.3f}, Detection={detection_rate:.1f}%")
        
        return {
            'method_name': method_name,
            'n_significant': n_significant,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'specificity': specificity,
            'detection_rate': detection_rate
        }
    
    def create_comparison_report(self, results: Dict[str, Any]) -> str:
        """Create comprehensive comparison report."""
        
        report = []
        report.append("# CLUSTERING METHOD COMPARISON RESULTS")
        report.append("## Multiple Comparison Correction: Spatial vs Functional vs Causal Graph Clustering")
        report.append("=" * 80)
        report.append("")
        
        # Summary table
        clustering_results = results['clustering_results']
        
        report.append("## PERFORMANCE COMPARISON")
        report.append("-" * 35)
        report.append("")
        
        summary_data = []
        for method_name, method_results in clustering_results.items():
            summary_data.append({
                'Method': method_results['method_name'],
                'True Positives': method_results['true_positives'],
                'False Positives': method_results['false_positives'],
                'Detection Rate': f"{method_results['detection_rate']:.1f}%",
                'Precision': f"{method_results['precision']:.3f}",
                'Recall': f"{method_results['recall']:.3f}",
                'F1-Score': f"{method_results['f1_score']:.3f}",
                'Specificity': f"{method_results['specificity']:.3f}"
            })
        
        df = pd.DataFrame(summary_data)
        report.append(df.to_string(index=False))
        report.append("")
        
        # Analysis
        best_method = max(clustering_results.keys(), 
                         key=lambda k: clustering_results[k]['f1_score'])
        best_f1 = clustering_results[best_method]['f1_score']
        
        report.append("## KEY FINDINGS")
        report.append("-" * 20)
        report.append(f"**Best Performing Method**: {clustering_results[best_method]['method_name']} (F1={best_f1:.3f})")
        report.append("")
        
        # Rank methods by F1-score
        ranked_methods = sorted(clustering_results.items(), 
                               key=lambda x: x[1]['f1_score'], reverse=True)
        
        report.append("**Method Ranking by F1-Score**:")
        for i, (method_key, method_results) in enumerate(ranked_methods):
            rank_symbol = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
            report.append(f"{rank_symbol} {method_results['method_name']}: F1={method_results['f1_score']:.3f}, "
                         f"Detection={method_results['detection_rate']:.1f}%")
        report.append("")
        
        # Analysis of results
        report.append("## ANALYSIS")
        report.append("-" * 15)
        report.append("")
        
        # Compare causal graph clustering to other methods
        causal_results = clustering_results.get('causal_graph_clustering', {})
        if causal_results:
            causal_f1 = causal_results['f1_score']
            causal_detection = causal_results['detection_rate']
            
            # Compare to traditional FDR
            fdr_results = clustering_results.get('traditional_fdr', {})
            if fdr_results:
                fdr_f1 = fdr_results['f1_score']
                fdr_detection = fdr_results['detection_rate']
                
                if causal_f1 > fdr_f1:
                    improvement = ((causal_f1 - fdr_f1) / max(fdr_f1, 0.001)) * 100
                    report.append(f"### Causal Graph Clustering vs Traditional FDR")
                    report.append(f"- **Improvement**: {improvement:.1f}% better F1-score")
                    report.append(f"- **Detection Rate**: {causal_detection:.1f}% vs {fdr_detection:.1f}%")
                    if causal_results['false_positives'] <= fdr_results['false_positives']:
                        report.append(f"- **False Positive Control**: Equal or better ({causal_results['false_positives']} vs {fdr_results['false_positives']})")
                    report.append("")
            
            # Compare to spatial clustering
            spatial_results = clustering_results.get('spatial_clustering', {})
            if spatial_results:
                spatial_f1 = spatial_results['f1_score']
                spatial_detection = spatial_results['detection_rate']
                
                report.append(f"### Causal Graph Clustering vs Spatial Clustering")
                if causal_f1 > spatial_f1:
                    improvement = ((causal_f1 - spatial_f1) / max(spatial_f1, 0.001)) * 100
                    report.append(f"- **Advantage**: {improvement:.1f}% better F1-score")
                    report.append(f"- **Key Insight**: Causal clustering captures distant functional connections")
                elif spatial_f1 > causal_f1:
                    advantage = ((spatial_f1 - causal_f1) / max(causal_f1, 0.001)) * 100
                    report.append(f"- **Spatial Advantage**: {advantage:.1f}% better F1-score")
                    report.append(f"- **Possible Reason**: Local connections dominate in this dataset")
                else:
                    report.append(f"- **Similar Performance**: Both methods achieve F1={causal_f1:.3f}")
                report.append("")
        
        # Conclusions
        report.append("## CONCLUSIONS")
        report.append("-" * 20)
        report.append("")
        
        if causal_results and causal_results['f1_score'] > 0:
            if any(clustering_results[method]['f1_score'] < causal_results['f1_score'] 
                   for method in clustering_results if method != 'causal_graph_clustering'):
                report.append("✅ **CAUSAL GRAPH CLUSTERING SHOWS PROMISE**: Outperforms some traditional methods")
                report.append("")
                report.append("**Advantages of Causal Graph Clustering**:")
                report.append("- Captures long-range functional connections ignored by spatial clustering")
                report.append("- Groups connections based on causal relationships rather than anatomy")
                report.append("- Potentially more sensitive to network-level connectivity patterns")
                report.append("")
                
                report.append("**When to Use Causal Graph Clustering**:")
                report.append("- Studies focusing on functional networks rather than local circuits")
                report.append("- Analysis of long-range connectivity patterns")
                report.append("- Directional effective connectivity investigations")
            else:
                report.append("📊 **MIXED RESULTS**: Causal graph clustering competitive but not clearly superior")
                report.append("")
                report.append("**Possible Explanations**:")
                report.append("- Dataset may have strong local connectivity that favors spatial clustering")
                report.append("- Causal graph construction parameters may need optimization")
                report.append("- Small sample size may limit statistical power")
        else:
            report.append("⚠️ **LIMITED EVIDENCE**: Causal graph clustering needs further development")
            report.append("")
            report.append("**Areas for Improvement**:")
            report.append("- Graph construction methodology")
            report.append("- Clustering algorithm selection")
            report.append("- Statistical testing within clusters")
        
        report.append("")
        report.append("## RECOMMENDATIONS")
        report.append("-" * 25)
        report.append("")
        report.append("**For Method Selection**:")
        report.append(f"- Use **{clustering_results[best_method]['method_name']}** for best overall performance")
        report.append("- Consider study-specific connectivity patterns when choosing clustering approach")
        report.append("- Validate clustering choice with pilot data when possible")
        report.append("")
        
        report.append("**For Future Development**:")
        report.append("- Test causal graph clustering on datasets with known long-range connectivity")
        report.append("- Optimize graph construction parameters for different connectivity types")
        report.append("- Develop adaptive clustering selection based on data characteristics")
        
        return "\\n".join(report)
    
    def run_comparison(self) -> Dict[str, Any]:
        """Run complete clustering method comparison."""
        
        print("🚀 CLUSTERING METHOD COMPARISON FOR MULTIPLE COMPARISON CORRECTION")
        print("=" * 80)
        print("Testing whether causal graph clustering outperforms spatial/functional clustering")
        print("=" * 80)
        
        # Create test scenario
        data, roi_labels, ground_truth, cluster_info = self.create_test_scenario_with_spatial_and_causal_clusters()
        
        # Test clustering methods
        results = self.test_clustering_methods(data, roi_labels, ground_truth, cluster_info)
        
        # Generate report
        report = self.create_comparison_report(results)
        
        # Save report
        from pathlib import Path
        report_file = Path("./clustering_method_comparison_report.md")
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\\n📄 Complete comparison report saved to: {report_file}")
        
        return results

def main():
    """Run clustering method comparison."""
    comparator = ClusteringMethodComparison(random_state=42)
    results = comparator.run_comparison()
    return results

if __name__ == "__main__":
    results = main()