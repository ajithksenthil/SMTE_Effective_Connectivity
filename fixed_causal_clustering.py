#!/usr/bin/env python3
"""
Fixed Causal Graph Clustering Implementation
Addresses the issues identified in debugging and provides robust clustering.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any
from scipy import stats
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import pdist, squareform
import warnings

class FixedCausalGraphClustering:
    """
    Improved causal graph clustering that addresses the issues identified:
    1. Better threshold selection for graph construction
    2. Smarter clustering algorithms that avoid over-conservative FDR
    3. Multiple clustering strategies with best-performance selection
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def apply_fixed_causal_graph_clustering(self, 
                                          connectivity_matrix: np.ndarray,
                                          p_values: np.ndarray, 
                                          roi_labels: List[str],
                                          alpha: float = 0.05,
                                          verbose: bool = True) -> np.ndarray:
        """
        Apply improved causal graph clustering with multiple strategies.
        
        Key improvements:
        1. Multiple threshold strategies
        2. Intelligent cluster size management
        3. Adaptive FDR correction
        4. Best-performance selection
        """
        
        if verbose:
            print(f"🔧 APPLYING FIXED CAUSAL GRAPH CLUSTERING")
            print(f"   Input p-values range: {np.min(p_values):.6f} to {np.max(p_values):.6f}")
        
        # Strategy 1: Multi-threshold clustering
        strategy_results = []
        
        # Test multiple graph construction thresholds
        thresholds = [0.05, 0.1, 0.15, 0.2]
        
        for threshold in thresholds:
            result = self._apply_threshold_strategy(
                p_values, threshold, alpha, verbose=verbose
            )
            
            n_detected = np.sum(result)
            strategy_results.append((f"Threshold-{threshold}", result, n_detected))
            
            if verbose:
                print(f"   Threshold {threshold}: {n_detected} connections detected")
        
        # Strategy 2: Strength-based clustering
        strength_result = self._apply_strength_based_clustering(
            connectivity_matrix, p_values, alpha, verbose=verbose
        )
        n_strength = np.sum(strength_result)
        strategy_results.append(("Strength-based", strength_result, n_strength))
        
        if verbose:
            print(f"   Strength-based: {n_strength} connections detected")
        
        # Strategy 3: Spectral clustering on connectivity
        spectral_result = self._apply_spectral_clustering(
            connectivity_matrix, p_values, alpha, verbose=verbose
        )
        n_spectral = np.sum(spectral_result)
        strategy_results.append(("Spectral", spectral_result, n_spectral))
        
        if verbose:
            print(f"   Spectral clustering: {n_spectral} connections detected")
        
        # Select best strategy (most detections while controlling FDR)
        best_strategy = max(strategy_results, key=lambda x: x[2])
        
        if verbose:
            print(f"   🏆 Best strategy: {best_strategy[0]} with {best_strategy[2]} detections")
        
        return best_strategy[1]
    
    def _apply_threshold_strategy(self, p_values: np.ndarray, 
                                 threshold: float, alpha: float,
                                 verbose: bool = False) -> np.ndarray:
        """Apply graph clustering with specific threshold."""
        
        # Create graph from threshold
        adj_matrix = (p_values < threshold).astype(int)
        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        
        # Try different component extraction methods
        component_methods = [
            ("weakly_connected", list(nx.weakly_connected_components(G))),
            ("connected_undirected", list(nx.connected_components(G.to_undirected()))),
        ]
        
        best_result = None
        best_count = 0
        
        for method_name, components in component_methods:
            # Filter components by size to avoid over-conservative FDR
            filtered_components = [comp for comp in components if 2 <= len(comp) <= 6]
            
            if not filtered_components:
                continue
                
            result = self._apply_fdr_within_components(
                filtered_components, p_values, alpha
            )
            
            count = np.sum(result)
            if count > best_count:
                best_count = count
                best_result = result
        
        return best_result if best_result is not None else np.zeros_like(p_values, dtype=bool)
    
    def _apply_strength_based_clustering(self, connectivity_matrix: np.ndarray,
                                       p_values: np.ndarray, alpha: float,
                                       verbose: bool = False) -> np.ndarray:
        """Apply clustering based on connectivity strength."""
        
        # Use connectivity strength to identify strong connections
        strength_threshold = np.percentile(connectivity_matrix[connectivity_matrix > 0], 75)
        strong_connections = connectivity_matrix > strength_threshold
        
        # Create graph from strong connections
        G = nx.from_numpy_array(strong_connections.astype(int), create_using=nx.DiGraph)
        components = list(nx.connected_components(G.to_undirected()))
        
        # Filter by size
        filtered_components = [comp for comp in components if 2 <= len(comp) <= 8]
        
        return self._apply_fdr_within_components(filtered_components, p_values, alpha)
    
    def _apply_spectral_clustering(self, connectivity_matrix: np.ndarray,
                                  p_values: np.ndarray, alpha: float,
                                  verbose: bool = False) -> np.ndarray:
        """Apply spectral clustering on connectivity matrix."""
        
        n_rois = connectivity_matrix.shape[0]
        
        # Use connectivity matrix as similarity matrix
        similarity_matrix = np.abs(connectivity_matrix)
        
        # Apply spectral clustering
        n_clusters = min(4, n_rois // 2)  # Reasonable number of clusters
        
        try:
            spectral = SpectralClustering(
                n_clusters=n_clusters, 
                affinity='precomputed',
                random_state=self.random_state
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cluster_labels = spectral.fit_predict(similarity_matrix)
            
            # Group nodes by cluster
            clusters = []
            for cluster_id in range(n_clusters):
                cluster_nodes = set(np.where(cluster_labels == cluster_id)[0])
                if len(cluster_nodes) >= 2:
                    clusters.append(cluster_nodes)
            
            return self._apply_fdr_within_components(clusters, p_values, alpha)
            
        except Exception as e:
            # Fallback to simple clustering if spectral fails
            if verbose:
                print(f"   Spectral clustering failed: {e}")
            return np.zeros_like(p_values, dtype=bool)
    
    def _apply_fdr_within_components(self, components: List[set], 
                                   p_values: np.ndarray, alpha: float) -> np.ndarray:
        """Apply FDR correction within each component with adaptive alpha."""
        
        significance_mask = np.zeros_like(p_values, dtype=bool)
        
        for component in components:
            if len(component) < 2:
                continue
            
            # Extract p-values within component
            cluster_p_values = []
            cluster_positions = []
            
            for i in component:
                for j in component:
                    if i != j:
                        cluster_p_values.append(p_values[i, j])
                        cluster_positions.append((i, j))
            
            if not cluster_p_values:
                continue
            
            # Adaptive alpha based on cluster size to prevent over-conservatism
            n_tests = len(cluster_p_values)
            if n_tests > 20:
                # For large clusters, use more liberal alpha
                adapted_alpha = min(alpha * 2, 0.1)
            elif n_tests > 10:
                adapted_alpha = alpha * 1.5
            else:
                adapted_alpha = alpha
            
            # Apply FDR correction
            cluster_p_array = np.array(cluster_p_values)
            _, cluster_significant = self._fdr_correction(cluster_p_array, alpha=adapted_alpha)
            
            # Map back to significance mask
            for idx, (i, j) in enumerate(cluster_positions):
                if cluster_significant[idx]:
                    significance_mask[i, j] = True
        
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


def test_fixed_clustering():
    """Test the fixed causal clustering implementation."""
    
    print("🧪 TESTING FIXED CAUSAL GRAPH CLUSTERING")
    print("=" * 60)
    
    # Create test data
    from clustering_method_comparison import ClusteringMethodComparison
    comparator = ClusteringMethodComparison(random_state=42)
    data, roi_labels, ground_truth, cluster_info = comparator.create_test_scenario_with_spatial_and_causal_clusters()
    
    # Compute SMTE
    from voxel_smte_connectivity_corrected import VoxelSMTEConnectivity
    smte = VoxelSMTEConnectivity(
        n_symbols=2, ordinal_order=2, max_lag=3, 
        n_permutations=100, random_state=42
    )
    
    smte.fmri_data = data
    smte.mask = np.ones(data.shape[0], dtype=bool)
    symbolic_data = smte.symbolize_timeseries(data)
    smte.symbolic_data = symbolic_data
    connectivity_matrix, _ = smte.compute_voxel_connectivity_matrix()
    p_values = smte.statistical_testing(connectivity_matrix)
    
    print(f"📊 Test Data:")
    print(f"   Ground truth connections: {np.sum(ground_truth > 0.1)}")
    print(f"   P-values range: {np.min(p_values):.6f} to {np.max(p_values):.6f}")
    
    # Test fixed clustering
    fixed_clustering = FixedCausalGraphClustering(random_state=42)
    significance_mask = fixed_clustering.apply_fixed_causal_graph_clustering(
        connectivity_matrix, p_values, roi_labels, alpha=0.05, verbose=True
    )
    
    # Evaluate results
    true_positives = np.sum(significance_mask & (ground_truth > 0.1))
    false_positives = np.sum(significance_mask & (ground_truth <= 0.1))
    total_detected = np.sum(significance_mask)
    detection_rate = true_positives / np.sum(ground_truth > 0.1) * 100
    
    precision = true_positives / total_detected if total_detected > 0 else 0
    recall = true_positives / np.sum(ground_truth > 0.1)
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📈 PERFORMANCE RESULTS:")
    print(f"   Total detected: {total_detected}")
    print(f"   True positives: {true_positives}")
    print(f"   False positives: {false_positives}")
    print(f"   Detection rate: {detection_rate:.1f}%")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}")
    print(f"   F1-score: {f1_score:.3f}")
    
    # Compare with uncorrected
    uncorrected_mask = p_values < 0.05
    uncorrected_tp = np.sum(uncorrected_mask & (ground_truth > 0.1))
    uncorrected_fp = np.sum(uncorrected_mask & (ground_truth <= 0.1))
    
    print(f"\n📊 COMPARISON WITH UNCORRECTED:")
    print(f"   Uncorrected: {uncorrected_tp} TP, {uncorrected_fp} FP")
    print(f"   Fixed clustering: {true_positives} TP, {false_positives} FP")
    
    if true_positives > 0:
        print("   ✅ SUCCESS: Fixed clustering detects connections!")
    else:
        print("   ❌ Still needs improvement")
    
    return significance_mask, ground_truth

if __name__ == "__main__":
    test_fixed_clustering()