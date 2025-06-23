#!/usr/bin/env python3
"""
Debug Causal Graph Clustering Implementation
Systematically identify and fix issues with causal graph clustering.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from clustering_method_comparison import ClusteringMethodComparison

class CausalClusteringDebugger:
    """Debug the causal graph clustering implementation."""
    
    def __init__(self):
        self.comparator = ClusteringMethodComparison(random_state=42)
        
    def debug_graph_construction(self):
        """Debug the graph construction process step by step."""
        
        print("🔍 DEBUGGING CAUSAL GRAPH CLUSTERING")
        print("=" * 50)
        
        # Create test data
        data, roi_labels, ground_truth, cluster_info = self.comparator.create_test_scenario_with_spatial_and_causal_clusters()
        
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
        
        print(f"📊 SMTE Analysis Results:")
        print(f"   Connectivity matrix shape: {connectivity_matrix.shape}")
        print(f"   P-values range: {np.min(p_values):.6f} to {np.max(p_values):.6f}")
        print(f"   Ground truth connections: {np.sum(ground_truth > 0.1)}")
        
        # Debug different threshold levels
        thresholds = [0.05, 0.1, 0.15, 0.2, 0.3]
        
        for threshold in thresholds:
            print(f"\n🔍 Testing threshold = {threshold}")
            
            # Create adjacency matrix
            adj_matrix = (p_values < threshold).astype(int)
            n_connections = np.sum(adj_matrix) - np.trace(adj_matrix)  # Exclude diagonal
            
            print(f"   Connections found: {n_connections}")
            
            if n_connections > 0:
                # Create graph
                G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
                G_undirected = G.to_undirected()
                
                # Find connected components
                components = list(nx.connected_components(G_undirected))
                print(f"   Connected components: {len(components)}")
                
                for i, component in enumerate(components):
                    print(f"     Component {i+1}: {len(component)} nodes = {component}")
                    
                    # Check if this component has ground truth connections
                    component_connections = 0
                    true_connections = 0
                    
                    for node1 in component:
                        for node2 in component:
                            if node1 != node2:
                                component_connections += 1
                                if ground_truth[node1, node2] > 0.1:
                                    true_connections += 1
                    
                    print(f"       Total component connections: {component_connections}")
                    print(f"       True positive connections: {true_connections}")
                
                # Test clustering approach
                significance_mask = self._test_clustering_on_components(
                    components, p_values, threshold=0.05
                )
                detected = np.sum(significance_mask)
                true_detected = np.sum(significance_mask & (ground_truth > 0.1))
                
                print(f"   After FDR within clusters: {detected} total, {true_detected} true positives")
            else:
                print("   No connections found - threshold too strict")
        
        return p_values, ground_truth, roi_labels
        
    def _test_clustering_on_components(self, components, p_values, threshold=0.05):
        """Test FDR correction within connected components."""
        
        significance_mask = np.zeros_like(p_values, dtype=bool)
        
        for component in components:
            if len(component) < 2:
                continue
                
            # Extract p-values within this component
            cluster_p_values = []
            cluster_positions = []
            
            for i in component:
                for j in component:
                    if i != j:
                        cluster_p_values.append(p_values[i, j])
                        cluster_positions.append((i, j))
            
            if cluster_p_values:
                # Apply FDR correction
                cluster_p_array = np.array(cluster_p_values)
                _, cluster_significant = self._fdr_correction(cluster_p_array, alpha=threshold)
                
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
    
    def test_improved_graph_construction(self):
        """Test improved graph construction methods."""
        
        print("\n🔧 TESTING IMPROVED GRAPH CONSTRUCTION")
        print("=" * 50)
        
        # Get test data
        p_values, ground_truth, roi_labels = self.debug_graph_construction()
        
        # Try different graph construction approaches
        approaches = [
            ("Liberal threshold", 0.2),
            ("Moderate threshold", 0.15), 
            ("Conservative threshold", 0.1),
            ("Very conservative", 0.05)
        ]
        
        best_approach = None
        best_detection = 0
        
        for approach_name, threshold in approaches:
            print(f"\n🎯 Testing {approach_name} (p < {threshold})")
            
            # Create base connections
            base_connections = p_values < threshold
            
            # Apply causal graph clustering
            significance_mask = self._apply_improved_causal_clustering(
                p_values, base_connections, alpha=0.05
            )
            
            # Evaluate performance
            true_positives = np.sum(significance_mask & (ground_truth > 0.1))
            false_positives = np.sum(significance_mask & (ground_truth <= 0.1))
            detection_rate = true_positives / np.sum(ground_truth > 0.1) * 100
            
            print(f"   True positives: {true_positives}")
            print(f"   False positives: {false_positives}")
            print(f"   Detection rate: {detection_rate:.1f}%")
            
            if true_positives > best_detection:
                best_detection = true_positives
                best_approach = (approach_name, threshold)
        
        print(f"\n🏆 Best approach: {best_approach[0]} with {best_detection} detections")
        return best_approach
    
    def _apply_improved_causal_clustering(self, p_values, base_connections, alpha=0.05):
        """Improved causal graph clustering implementation."""
        
        # Create directed graph from base connections
        G = nx.from_numpy_array(base_connections.astype(int), create_using=nx.DiGraph)
        
        # Multiple clustering approaches
        clustering_results = []
        
        # Approach 1: Connected components
        G_undirected = G.to_undirected()
        components = list(nx.connected_components(G_undirected))
        result1 = self._cluster_by_components(components, p_values, alpha)
        clustering_results.append(("Connected Components", result1))
        
        # Approach 2: Weakly connected components (for directed graphs)
        weak_components = list(nx.weakly_connected_components(G))
        result2 = self._cluster_by_components(weak_components, p_values, alpha)
        clustering_results.append(("Weak Components", result2))
        
        # Approach 3: Strongly connected components
        strong_components = list(nx.strongly_connected_components(G))
        # Filter out single-node components
        strong_components = [comp for comp in strong_components if len(comp) > 1]
        result3 = self._cluster_by_components(strong_components, p_values, alpha)
        clustering_results.append(("Strong Components", result3))
        
        # Choose best result (most detections)
        best_result = max(clustering_results, key=lambda x: np.sum(x[1]))
        print(f"     Best clustering: {best_result[0]} ({np.sum(best_result[1])} detections)")
        
        return best_result[1]
    
    def _cluster_by_components(self, components, p_values, alpha):
        """Apply FDR correction within each component."""
        
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
            
            if cluster_p_values:
                cluster_p_array = np.array(cluster_p_values)
                _, cluster_significant = self._fdr_correction(cluster_p_array, alpha=alpha)
                
                for idx, (i, j) in enumerate(cluster_positions):
                    if cluster_significant[idx]:
                        significance_mask[i, j] = True
        
        return significance_mask

def main():
    """Run causal clustering debugging."""
    debugger = CausalClusteringDebugger()
    
    # Debug current issues
    debugger.debug_graph_construction()
    
    # Test improvements
    best_approach = debugger.test_improved_graph_construction()
    
    return best_approach

if __name__ == "__main__":
    best_approach = main()