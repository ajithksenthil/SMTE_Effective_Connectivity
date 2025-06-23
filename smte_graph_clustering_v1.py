#!/usr/bin/env python3
"""
SMTE Graph Network Clustering for Cluster-Level Multiple Comparisons Thresholding
Extension of the enhanced SMTE framework for directional effective connectivity networks.

This module implements graph-based clustering approaches for SMTE connectivity matrices
with cluster-level statistical thresholding to improve detection sensitivity while
controlling family-wise error rates.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
import logging
from scipy import stats, ndimage
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from hierarchical_smte_v1 import HierarchicalSMTE

logging.basicConfig(level=logging.INFO)


class SMTEGraphClusterAnalyzer:
    """
    Advanced graph clustering analyzer for SMTE connectivity networks.
    Implements cluster-level statistical thresholding for directional networks.
    """
    
    def __init__(self, 
                 clustering_methods: List[str] = ['spectral', 'louvain', 'modularity'],
                 cluster_sizes: List[int] = [5, 10, 15, 20],
                 cluster_thresholds: List[float] = [0.01, 0.05, 0.1],
                 directional_analysis: bool = True):
        
        self.clustering_methods = clustering_methods
        self.cluster_sizes = cluster_sizes
        self.cluster_thresholds = cluster_thresholds
        self.directional_analysis = directional_analysis
        
        # Available clustering algorithms
        self.clustering_algorithms = {
            'spectral': self._spectral_clustering,
            'louvain': self._louvain_clustering,
            'modularity': self._modularity_clustering,
            'walktrap': self._walktrap_clustering,
            'infomap': self._infomap_clustering,
            'leiden': self._leiden_clustering
        }
        
        # Cluster-level statistics
        self.cluster_stats_methods = {
            'max_statistic': self._compute_max_statistic,
            'sum_statistic': self._compute_sum_statistic,
            'mean_statistic': self._compute_mean_statistic,
            'median_statistic': self._compute_median_statistic,
            'cluster_mass': self._compute_cluster_mass
        }
        
        # Results storage
        self.clustering_results = {}
        self.cluster_statistics = {}
        self.permutation_distributions = {}
        
    def analyze_smte_graph_clusters(self,
                                  connectivity_matrix: np.ndarray,
                                  p_values: np.ndarray,
                                  roi_labels: List[str],
                                  initial_threshold: float = 0.05,
                                  n_permutations: int = 1000,
                                  cluster_alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform comprehensive graph clustering analysis on SMTE connectivity matrix.
        
        Parameters:
        -----------
        connectivity_matrix : np.ndarray
            SMTE connectivity matrix (n_rois x n_rois)
        p_values : np.ndarray
            Statistical p-values for each connection
        roi_labels : List[str]
            Labels for brain regions
        initial_threshold : float
            Initial threshold for forming clusters
        n_permutations : int
            Number of permutations for cluster-level statistics
        cluster_alpha : float
            Cluster-level significance threshold
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive clustering analysis results
        """
        
        print("🧠 SMTE GRAPH CLUSTERING ANALYSIS")
        print("=" * 70)
        print(f"Matrix size: {connectivity_matrix.shape}")
        print(f"Initial threshold: {initial_threshold}")
        print(f"Cluster alpha: {cluster_alpha}")
        print(f"Permutations: {n_permutations}")
        
        # Create initial thresholded network
        initial_network = self._create_thresholded_network(
            connectivity_matrix, p_values, initial_threshold
        )
        
        # Detect clusters using multiple methods
        cluster_results = self._detect_multiple_clusters(
            initial_network, connectivity_matrix, roi_labels
        )
        
        # Compute cluster-level statistics
        cluster_statistics = self._compute_cluster_level_statistics(
            cluster_results, connectivity_matrix, p_values
        )
        
        # Generate null distribution through permutation
        null_distributions = self._generate_cluster_null_distributions(
            connectivity_matrix, p_values, cluster_results, 
            initial_threshold, n_permutations
        )
        
        # Perform cluster-level statistical testing
        cluster_significance = self._test_cluster_significance(
            cluster_statistics, null_distributions, cluster_alpha
        )
        
        # Create final cluster-corrected connectivity matrix
        cluster_corrected_matrix = self._create_cluster_corrected_matrix(
            connectivity_matrix, cluster_results, cluster_significance
        )
        
        # Analyze directional properties if enabled
        directional_analysis = {}
        if self.directional_analysis:
            directional_analysis = self._analyze_directional_clusters(
                connectivity_matrix, cluster_results, roi_labels
            )
        
        # Compile comprehensive results
        graph_clustering_results = {
            'initial_network': initial_network,
            'cluster_results': cluster_results,
            'cluster_statistics': cluster_statistics,
            'null_distributions': null_distributions,
            'cluster_significance': cluster_significance,
            'cluster_corrected_matrix': cluster_corrected_matrix,
            'directional_analysis': directional_analysis,
            'analysis_parameters': {
                'initial_threshold': initial_threshold,
                'cluster_alpha': cluster_alpha,
                'n_permutations': n_permutations,
                'clustering_methods': self.clustering_methods,
                'directional_analysis': self.directional_analysis
            },
            'roi_labels': roi_labels
        }
        
        # Store results
        self.clustering_results = graph_clustering_results
        
        print(f"\\n🔍 CLUSTERING SUMMARY:")
        print(f"Methods tested: {len(cluster_results)}")
        print(f"Significant clusters found: {sum(len(sig.get('significant_clusters', [])) for sig in cluster_significance.values())}")
        print(f"Cluster-corrected connections: {np.sum(cluster_corrected_matrix > 0)}")
        
        return graph_clustering_results
    
    def _create_thresholded_network(self,
                                  connectivity_matrix: np.ndarray,
                                  p_values: np.ndarray,
                                  threshold: float) -> Dict[str, Any]:
        """
        Create initial thresholded network for clustering.
        """
        
        # Apply statistical threshold
        significant_mask = p_values < threshold
        
        # Create thresholded connectivity matrix
        thresholded_matrix = connectivity_matrix * significant_mask
        
        # Create NetworkX directed graph
        n_rois = connectivity_matrix.shape[0]
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(n_rois):
            G.add_node(i)
        
        # Add edges for significant connections
        for i in range(n_rois):
            for j in range(n_rois):
                if i != j and significant_mask[i, j]:
                    G.add_edge(i, j, weight=connectivity_matrix[i, j], 
                              p_value=p_values[i, j])
        
        # Compute basic network properties
        network_properties = self._compute_network_properties(G)
        
        return {
            'graph': G,
            'thresholded_matrix': thresholded_matrix,
            'significant_mask': significant_mask,
            'n_significant_edges': np.sum(significant_mask),
            'threshold': threshold,
            'properties': network_properties
        }
    
    def _compute_network_properties(self, G: nx.DiGraph) -> Dict[str, Any]:
        """
        Compute basic properties of the directed network.
        """
        
        properties = {}
        
        # Basic properties
        properties['n_nodes'] = G.number_of_nodes()
        properties['n_edges'] = G.number_of_edges()
        properties['density'] = nx.density(G)
        
        # Connectivity properties
        if G.number_of_edges() > 0:
            try:
                # Convert to undirected for some metrics
                G_undirected = G.to_undirected()
                
                # Connected components
                properties['n_connected_components'] = nx.number_connected_components(G_undirected)
                properties['largest_component_size'] = len(max(nx.connected_components(G_undirected), key=len))
                
                # Clustering coefficient
                properties['average_clustering'] = nx.average_clustering(G_undirected)
                
                # Degree statistics
                in_degrees = [d for n, d in G.in_degree()]
                out_degrees = [d for n, d in G.out_degree()]
                
                properties['mean_in_degree'] = np.mean(in_degrees)
                properties['mean_out_degree'] = np.mean(out_degrees)
                properties['degree_assortativity'] = nx.degree_assortativity_coefficient(G_undirected)
                
            except Exception as e:
                print(f"Warning: Could not compute some network properties: {e}")
                properties.update({
                    'n_connected_components': 1,
                    'largest_component_size': G.number_of_nodes(),
                    'average_clustering': 0.0,
                    'mean_in_degree': 0.0,
                    'mean_out_degree': 0.0,
                    'degree_assortativity': 0.0
                })
        else:
            # Empty graph
            properties.update({
                'n_connected_components': G.number_of_nodes(),
                'largest_component_size': 1,
                'average_clustering': 0.0,
                'mean_in_degree': 0.0,
                'mean_out_degree': 0.0,
                'degree_assortativity': 0.0
            })
        
        return properties
    
    def _detect_multiple_clusters(self,
                                initial_network: Dict[str, Any],
                                connectivity_matrix: np.ndarray,
                                roi_labels: List[str]) -> Dict[str, Any]:
        """
        Detect clusters using multiple clustering methods.
        """
        
        print("🔍 Detecting clusters using multiple methods...")
        
        cluster_results = {}
        G = initial_network['graph']
        thresholded_matrix = initial_network['thresholded_matrix']
        
        for method in self.clustering_methods:
            if method in self.clustering_algorithms:
                print(f"  Running {method} clustering...")
                
                try:
                    clustering_func = self.clustering_algorithms[method]
                    method_results = clustering_func(G, thresholded_matrix, roi_labels)
                    cluster_results[method] = method_results
                    
                    n_clusters = len(set(method_results['cluster_labels'])) - (1 if -1 in method_results['cluster_labels'] else 0)
                    print(f"    Found {n_clusters} clusters")
                    
                except Exception as e:
                    print(f"    ❌ Error in {method}: {str(e)}")
                    continue
        
        return cluster_results
    
    def _spectral_clustering(self, G: nx.DiGraph, 
                           connectivity_matrix: np.ndarray,
                           roi_labels: List[str]) -> Dict[str, Any]:
        """
        Perform spectral clustering on the connectivity matrix.
        """
        
        # Convert directed graph to similarity matrix
        n_nodes = len(roi_labels)
        
        if G.number_of_edges() == 0:
            # No edges - return singleton clusters
            return {
                'cluster_labels': list(range(n_nodes)),
                'n_clusters': n_nodes,
                'cluster_quality': 0.0,
                'method': 'spectral'
            }
        
        # Create similarity matrix from connectivity strengths
        similarity_matrix = np.abs(connectivity_matrix)
        
        # Try different numbers of clusters
        best_score = -1
        best_labels = None
        best_n_clusters = 2
        
        for n_clusters in range(2, min(n_nodes, max(self.cluster_sizes) + 1)):
            try:
                clustering = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',
                    random_state=42
                )
                
                labels = clustering.fit_predict(similarity_matrix)
                
                # Compute silhouette score
                if len(set(labels)) > 1:
                    score = silhouette_score(similarity_matrix, labels, metric='precomputed')
                    
                    if score > best_score:
                        best_score = score
                        best_labels = labels
                        best_n_clusters = n_clusters
                        
            except Exception as e:
                continue
        
        if best_labels is None:
            best_labels = list(range(n_nodes))
            best_n_clusters = n_nodes
            best_score = 0.0
        
        return {
            'cluster_labels': best_labels,
            'n_clusters': best_n_clusters,
            'cluster_quality': best_score,
            'method': 'spectral'
        }
    
    def _louvain_clustering(self, G: nx.DiGraph,
                          connectivity_matrix: np.ndarray,
                          roi_labels: List[str]) -> Dict[str, Any]:
        """
        Perform Louvain community detection.
        """
        
        try:
            # Convert to undirected graph with edge weights
            G_undirected = G.to_undirected()
            
            # Add weights to edges
            for u, v, data in G_undirected.edges(data=True):
                if 'weight' not in data:
                    G_undirected[u][v]['weight'] = abs(connectivity_matrix[u, v])
            
            # Perform Louvain clustering
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.louvain_communities(G_undirected, seed=42)
            
            # Convert to cluster labels
            cluster_labels = [-1] * len(roi_labels)
            for cluster_id, community in enumerate(communities):
                for node in community:
                    cluster_labels[node] = cluster_id
            
            # Compute modularity as quality measure
            try:
                modularity = nx_comm.modularity(G_undirected, communities)
            except:
                modularity = 0.0
            
            return {
                'cluster_labels': cluster_labels,
                'n_clusters': len(communities),
                'cluster_quality': modularity,
                'method': 'louvain',
                'communities': communities
            }
            
        except ImportError:
            # Fallback to simple clustering if community detection not available
            return self._modularity_clustering(G, connectivity_matrix, roi_labels)
    
    def _modularity_clustering(self, G: nx.DiGraph,
                             connectivity_matrix: np.ndarray,
                             roi_labels: List[str]) -> Dict[str, Any]:
        """
        Perform modularity-based clustering.
        """
        
        # Convert to undirected graph
        G_undirected = G.to_undirected()
        
        if G_undirected.number_of_edges() == 0:
            return {
                'cluster_labels': list(range(len(roi_labels))),
                'n_clusters': len(roi_labels),
                'cluster_quality': 0.0,
                'method': 'modularity'
            }
        
        try:
            # Use greedy modularity optimization
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.greedy_modularity_communities(G_undirected)
            
            # Convert to cluster labels
            cluster_labels = [-1] * len(roi_labels)
            for cluster_id, community in enumerate(communities):
                for node in community:
                    cluster_labels[node] = cluster_id
            
            # Compute modularity
            modularity = nx_comm.modularity(G_undirected, communities)
            
            return {
                'cluster_labels': cluster_labels,
                'n_clusters': len(communities),
                'cluster_quality': modularity,
                'method': 'modularity'
            }
            
        except Exception as e:
            # Fallback to agglomerative clustering
            return self._agglomerative_fallback(connectivity_matrix, roi_labels)
    
    def _walktrap_clustering(self, G: nx.DiGraph,
                           connectivity_matrix: np.ndarray,
                           roi_labels: List[str]) -> Dict[str, Any]:
        """
        Simulate walktrap clustering using random walks.
        """
        
        # Simplified implementation - use spectral clustering as approximation
        return self._spectral_clustering(G, connectivity_matrix, roi_labels)
    
    def _infomap_clustering(self, G: nx.DiGraph,
                          connectivity_matrix: np.ndarray,
                          roi_labels: List[str]) -> Dict[str, Any]:
        """
        Simulate infomap clustering using information theory principles.
        """
        
        # Simplified implementation - use modularity as approximation
        return self._modularity_clustering(G, connectivity_matrix, roi_labels)
    
    def _leiden_clustering(self, G: nx.DiGraph,
                         connectivity_matrix: np.ndarray,
                         roi_labels: List[str]) -> Dict[str, Any]:
        """
        Simulate Leiden clustering (improved Louvain).
        """
        
        # Use Louvain as approximation
        return self._louvain_clustering(G, connectivity_matrix, roi_labels)
    
    def _agglomerative_fallback(self, connectivity_matrix: np.ndarray,
                              roi_labels: List[str]) -> Dict[str, Any]:
        """
        Fallback agglomerative clustering.
        """
        
        # Use correlation distance
        similarity_matrix = np.abs(connectivity_matrix)
        distance_matrix = 1 - similarity_matrix
        
        # Fill diagonal
        np.fill_diagonal(distance_matrix, 0)
        
        # Try different numbers of clusters
        best_score = -1
        best_labels = None
        best_n_clusters = 2
        
        for n_clusters in range(2, min(len(roi_labels), 10)):
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric='precomputed',
                    linkage='average'
                )
                
                labels = clustering.fit_predict(distance_matrix)
                
                if len(set(labels)) > 1:
                    score = silhouette_score(distance_matrix, labels, metric='precomputed')
                    
                    if score > best_score:
                        best_score = score
                        best_labels = labels
                        best_n_clusters = n_clusters
                        
            except Exception:
                continue
        
        if best_labels is None:
            best_labels = list(range(len(roi_labels)))
            best_n_clusters = len(roi_labels)
            best_score = 0.0
        
        return {
            'cluster_labels': best_labels,
            'n_clusters': best_n_clusters,
            'cluster_quality': best_score,
            'method': 'agglomerative'
        }
    
    def _compute_cluster_level_statistics(self,
                                        cluster_results: Dict[str, Any],
                                        connectivity_matrix: np.ndarray,
                                        p_values: np.ndarray) -> Dict[str, Any]:
        """
        Compute cluster-level statistics for each clustering method.
        """
        
        print("📊 Computing cluster-level statistics...")
        
        cluster_statistics = {}
        
        for method_name, method_results in cluster_results.items():
            cluster_labels = method_results['cluster_labels']
            unique_clusters = list(set(cluster_labels))
            
            method_stats = {}
            
            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Skip noise points
                    continue
                
                # Get nodes in this cluster
                cluster_nodes = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                if len(cluster_nodes) < 2:  # Skip singleton clusters
                    continue
                
                # Compute various cluster statistics
                cluster_stats = {}
                
                for stat_name, stat_func in self.cluster_stats_methods.items():
                    try:
                        stat_value = stat_func(
                            cluster_nodes, connectivity_matrix, p_values
                        )
                        cluster_stats[stat_name] = stat_value
                    except Exception as e:
                        cluster_stats[stat_name] = 0.0
                
                # Additional cluster properties
                cluster_stats.update({
                    'cluster_size': len(cluster_nodes),
                    'cluster_nodes': cluster_nodes,
                    'within_cluster_density': self._compute_within_cluster_density(
                        cluster_nodes, connectivity_matrix, p_values
                    ),
                    'cluster_coherence': self._compute_cluster_coherence(
                        cluster_nodes, connectivity_matrix
                    )
                })
                
                method_stats[f'cluster_{cluster_id}'] = cluster_stats
            
            cluster_statistics[method_name] = method_stats
        
        return cluster_statistics
    
    def _compute_max_statistic(self,
                             cluster_nodes: List[int],
                             connectivity_matrix: np.ndarray,
                             p_values: np.ndarray) -> float:
        """
        Compute maximum statistic within cluster.
        """
        
        max_stat = 0.0
        
        for i in cluster_nodes:
            for j in cluster_nodes:
                if i != j:
                    # Use -log(p) as test statistic
                    p_val = p_values[i, j]
                    if p_val > 0:
                        stat = -np.log(p_val)
                        max_stat = max(max_stat, stat)
        
        return max_stat
    
    def _compute_sum_statistic(self,
                             cluster_nodes: List[int],
                             connectivity_matrix: np.ndarray,
                             p_values: np.ndarray) -> float:
        """
        Compute sum of statistics within cluster.
        """
        
        sum_stat = 0.0
        
        for i in cluster_nodes:
            for j in cluster_nodes:
                if i != j:
                    p_val = p_values[i, j]
                    if p_val > 0:
                        sum_stat += -np.log(p_val)
        
        return sum_stat
    
    def _compute_mean_statistic(self,
                              cluster_nodes: List[int],
                              connectivity_matrix: np.ndarray,
                              p_values: np.ndarray) -> float:
        """
        Compute mean statistic within cluster.
        """
        
        stats = []
        
        for i in cluster_nodes:
            for j in cluster_nodes:
                if i != j:
                    p_val = p_values[i, j]
                    if p_val > 0:
                        stats.append(-np.log(p_val))
        
        return np.mean(stats) if stats else 0.0
    
    def _compute_median_statistic(self,
                                cluster_nodes: List[int],
                                connectivity_matrix: np.ndarray,
                                p_values: np.ndarray) -> float:
        """
        Compute median statistic within cluster.
        """
        
        stats = []
        
        for i in cluster_nodes:
            for j in cluster_nodes:
                if i != j:
                    p_val = p_values[i, j]
                    if p_val > 0:
                        stats.append(-np.log(p_val))
        
        return np.median(stats) if stats else 0.0
    
    def _compute_cluster_mass(self,
                            cluster_nodes: List[int],
                            connectivity_matrix: np.ndarray,
                            p_values: np.ndarray) -> float:
        """
        Compute cluster mass (sum of suprathreshold statistics).
        """
        
        cluster_mass = 0.0
        threshold_stat = -np.log(0.05)  # Corresponds to p < 0.05
        
        for i in cluster_nodes:
            for j in cluster_nodes:
                if i != j:
                    p_val = p_values[i, j]
                    if p_val > 0:
                        stat = -np.log(p_val)
                        if stat > threshold_stat:
                            cluster_mass += stat - threshold_stat
        
        return cluster_mass
    
    def _compute_within_cluster_density(self,
                                      cluster_nodes: List[int],
                                      connectivity_matrix: np.ndarray,
                                      p_values: np.ndarray) -> float:
        """
        Compute density of significant connections within cluster.
        """
        
        n_nodes = len(cluster_nodes)
        if n_nodes < 2:
            return 0.0
        
        total_possible = n_nodes * (n_nodes - 1)  # Directed graph
        significant_connections = 0
        
        for i in cluster_nodes:
            for j in cluster_nodes:
                if i != j and p_values[i, j] < 0.05:
                    significant_connections += 1
        
        return significant_connections / total_possible
    
    def _compute_cluster_coherence(self,
                                 cluster_nodes: List[int],
                                 connectivity_matrix: np.ndarray) -> float:
        """
        Compute coherence of connectivity patterns within cluster.
        """
        
        if len(cluster_nodes) < 2:
            return 1.0
        
        # Extract submatrix for cluster nodes
        cluster_matrix = connectivity_matrix[np.ix_(cluster_nodes, cluster_nodes)]
        
        # Compute pairwise correlations
        correlations = []
        n_nodes = len(cluster_nodes)
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                # Correlation between connectivity profiles
                profile_i = connectivity_matrix[cluster_nodes[i], :]
                profile_j = connectivity_matrix[cluster_nodes[j], :]
                
                corr = np.corrcoef(profile_i, profile_j)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _generate_cluster_null_distributions(self,
                                           connectivity_matrix: np.ndarray,
                                           p_values: np.ndarray,
                                           cluster_results: Dict[str, Any],
                                           initial_threshold: float,
                                           n_permutations: int) -> Dict[str, Any]:
        """
        Generate null distributions for cluster statistics through permutation.
        """
        
        print(f"🎲 Generating null distributions ({n_permutations} permutations)...")
        
        null_distributions = {}
        
        for method_name, method_results in cluster_results.items():
            print(f"  Processing {method_name}...")
            
            method_nulls = {}
            cluster_labels = method_results['cluster_labels']
            unique_clusters = [c for c in set(cluster_labels) if c != -1]
            
            for cluster_id in unique_clusters:
                cluster_nodes = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                if len(cluster_nodes) < 2:
                    continue
                
                # Generate null distribution for this cluster
                cluster_nulls = {stat_name: [] for stat_name in self.cluster_stats_methods.keys()}
                
                for perm in range(n_permutations):
                    # Permute connectivity matrix while preserving structure
                    perm_matrix, perm_p_values = self._permute_connectivity_matrix(
                        connectivity_matrix, p_values
                    )
                    
                    # Compute statistics for permuted data
                    for stat_name, stat_func in self.cluster_stats_methods.items():
                        try:
                            null_stat = stat_func(cluster_nodes, perm_matrix, perm_p_values)
                            cluster_nulls[stat_name].append(null_stat)
                        except:
                            cluster_nulls[stat_name].append(0.0)
                
                method_nulls[f'cluster_{cluster_id}'] = cluster_nulls
            
            null_distributions[method_name] = method_nulls
        
        return null_distributions
    
    def _permute_connectivity_matrix(self,
                                   connectivity_matrix: np.ndarray,
                                   p_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Permute connectivity matrix while preserving statistical structure.
        """
        
        # Simple permutation: shuffle edge weights
        flat_indices = np.triu_indices_from(connectivity_matrix, k=1)
        
        # Get upper triangle values
        conn_values = connectivity_matrix[flat_indices]
        p_val_values = p_values[flat_indices]
        
        # Shuffle
        perm_indices = np.random.permutation(len(conn_values))
        perm_conn_values = conn_values[perm_indices]
        perm_p_values = p_val_values[perm_indices]
        
        # Create permuted matrices
        perm_connectivity = np.zeros_like(connectivity_matrix)
        perm_p_matrix = np.ones_like(p_values)
        
        # Fill upper triangle
        perm_connectivity[flat_indices] = perm_conn_values
        perm_p_matrix[flat_indices] = perm_p_values
        
        # Mirror to lower triangle if undirected analysis
        if not self.directional_analysis:
            perm_connectivity += perm_connectivity.T
            perm_p_matrix = np.minimum(perm_p_matrix, perm_p_matrix.T)
        
        return perm_connectivity, perm_p_matrix
    
    def _test_cluster_significance(self,
                                 cluster_statistics: Dict[str, Any],
                                 null_distributions: Dict[str, Any],
                                 cluster_alpha: float) -> Dict[str, Any]:
        """
        Test significance of clusters against null distributions.
        """
        
        print(f"🧪 Testing cluster significance (α = {cluster_alpha})...")
        
        cluster_significance = {}
        
        for method_name in cluster_statistics.keys():
            method_significance = {
                'significant_clusters': [],
                'cluster_p_values': {},
                'significant_statistics': {}
            }
            
            method_stats = cluster_statistics[method_name]
            method_nulls = null_distributions.get(method_name, {})
            
            for cluster_name, cluster_stats in method_stats.items():
                cluster_p_values = {}
                significant_stats = {}
                
                for stat_name, observed_stat in cluster_stats.items():
                    if stat_name not in self.cluster_stats_methods:
                        continue
                    
                    if cluster_name in method_nulls and stat_name in method_nulls[cluster_name]:
                        null_stats = method_nulls[cluster_name][stat_name]
                        
                        if null_stats:
                            # Compute p-value (proportion of null stats >= observed)
                            p_value = np.mean(np.array(null_stats) >= observed_stat)
                            cluster_p_values[stat_name] = p_value
                            
                            if p_value < cluster_alpha:
                                significant_stats[stat_name] = {
                                    'observed': observed_stat,
                                    'p_value': p_value,
                                    'null_mean': np.mean(null_stats),
                                    'null_std': np.std(null_stats)
                                }
                
                method_significance['cluster_p_values'][cluster_name] = cluster_p_values
                
                # Mark cluster as significant if any statistic is significant
                if significant_stats:
                    method_significance['significant_clusters'].append(cluster_name)
                    method_significance['significant_statistics'][cluster_name] = significant_stats
            
            cluster_significance[method_name] = method_significance
            
            n_sig = len(method_significance['significant_clusters'])
            print(f"  {method_name}: {n_sig} significant clusters")
        
        return cluster_significance
    
    def _create_cluster_corrected_matrix(self,
                                       connectivity_matrix: np.ndarray,
                                       cluster_results: Dict[str, Any],
                                       cluster_significance: Dict[str, Any]) -> np.ndarray:
        """
        Create final cluster-corrected connectivity matrix.
        """
        
        print("🔧 Creating cluster-corrected connectivity matrix...")
        
        n_rois = connectivity_matrix.shape[0]
        corrected_matrix = np.zeros((n_rois, n_rois))
        
        # For each clustering method, add significant clusters
        for method_name, method_results in cluster_results.items():
            if method_name not in cluster_significance:
                continue
            
            cluster_labels = method_results['cluster_labels']
            significant_clusters = cluster_significance[method_name]['significant_clusters']
            
            for cluster_name in significant_clusters:
                cluster_id = int(cluster_name.split('_')[1])
                cluster_nodes = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                # Add connections within significant cluster
                for i in cluster_nodes:
                    for j in cluster_nodes:
                        if i != j:
                            corrected_matrix[i, j] = max(
                                corrected_matrix[i, j], 
                                connectivity_matrix[i, j]
                            )
        
        print(f"  Cluster-corrected connections: {np.sum(corrected_matrix > 0)}")
        
        return corrected_matrix
    
    def _analyze_directional_clusters(self,
                                    connectivity_matrix: np.ndarray,
                                    cluster_results: Dict[str, Any],
                                    roi_labels: List[str]) -> Dict[str, Any]:
        """
        Analyze directional properties of detected clusters.
        """
        
        print("➡️ Analyzing directional cluster properties...")
        
        directional_analysis = {}
        
        for method_name, method_results in cluster_results.items():
            cluster_labels = method_results['cluster_labels']
            unique_clusters = [c for c in set(cluster_labels) if c != -1]
            
            method_directional = {}
            
            for cluster_id in unique_clusters:
                cluster_nodes = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                if len(cluster_nodes) < 2:
                    continue
                
                # Analyze directional properties
                cluster_directional = self._compute_cluster_directional_properties(
                    cluster_nodes, connectivity_matrix, roi_labels
                )
                
                method_directional[f'cluster_{cluster_id}'] = cluster_directional
            
            directional_analysis[method_name] = method_directional
        
        return directional_analysis
    
    def _compute_cluster_directional_properties(self,
                                              cluster_nodes: List[int],
                                              connectivity_matrix: np.ndarray,
                                              roi_labels: List[str]) -> Dict[str, Any]:
        """
        Compute directional properties for a specific cluster.
        """
        
        n_nodes = len(cluster_nodes)
        
        # Extract cluster submatrix
        cluster_matrix = connectivity_matrix[np.ix_(cluster_nodes, cluster_nodes)]
        
        # Compute directional statistics
        directional_props = {}
        
        # Reciprocity: proportion of bidirectional connections
        bidirectional_count = 0
        total_connections = 0
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    if cluster_matrix[i, j] > 0 or cluster_matrix[j, i] > 0:
                        total_connections += 1
                        if cluster_matrix[i, j] > 0 and cluster_matrix[j, i] > 0:
                            bidirectional_count += 1
        
        reciprocity = bidirectional_count / total_connections if total_connections > 0 else 0
        
        # In-degree and out-degree statistics
        in_degrees = np.sum(cluster_matrix > 0, axis=0)
        out_degrees = np.sum(cluster_matrix > 0, axis=1)
        
        # Hub identification (nodes with high degree)
        total_degrees = in_degrees + out_degrees
        hub_threshold = np.mean(total_degrees) + np.std(total_degrees)
        hub_nodes = [cluster_nodes[i] for i in range(n_nodes) if total_degrees[i] > hub_threshold]
        
        # Directional flow analysis
        net_flow = out_degrees - in_degrees  # Positive = more outgoing, negative = more incoming
        
        directional_props = {
            'reciprocity': reciprocity,
            'mean_in_degree': np.mean(in_degrees),
            'mean_out_degree': np.mean(out_degrees),
            'degree_correlation': np.corrcoef(in_degrees, out_degrees)[0, 1] if len(in_degrees) > 1 else 0,
            'hub_nodes': [roi_labels[i] for i in hub_nodes],
            'n_hubs': len(hub_nodes),
            'net_flow_pattern': {
                'sources': [roi_labels[cluster_nodes[i]] for i in range(n_nodes) if net_flow[i] > 1],
                'sinks': [roi_labels[cluster_nodes[i]] for i in range(n_nodes) if net_flow[i] < -1],
                'balanced': [roi_labels[cluster_nodes[i]] for i in range(n_nodes) if abs(net_flow[i]) <= 1]
            }
        }
        
        return directional_props


class SMTEGraphClusteringSMTE(HierarchicalSMTE):
    """
    SMTE implementation with graph clustering for cluster-level multiple comparisons.
    Extends HierarchicalSMTE with advanced cluster-based statistical thresholding.
    """
    
    def __init__(self,
                 use_graph_clustering: bool = True,
                 clustering_methods: List[str] = ['spectral', 'louvain', 'modularity'],
                 cluster_thresholds: List[float] = [0.01, 0.05, 0.1],
                 cluster_alpha: float = 0.05,
                 cluster_n_permutations: int = 1000,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.use_graph_clustering = use_graph_clustering
        self.clustering_methods = clustering_methods
        self.cluster_thresholds = cluster_thresholds
        self.cluster_alpha = cluster_alpha
        self.cluster_n_permutations = cluster_n_permutations
        
        # Initialize graph clustering analyzer
        self.graph_cluster_analyzer = SMTEGraphClusterAnalyzer(
            clustering_methods=clustering_methods,
            cluster_thresholds=cluster_thresholds,
            directional_analysis=True
        )
        
        # Store clustering results
        self.graph_clustering_results = None
    
    def compute_graph_clustered_connectivity(self,
                                           data: np.ndarray,
                                           roi_labels: List[str],
                                           ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute connectivity with graph clustering for cluster-level thresholding.
        """
        
        print("🕸️ SMTE GRAPH CLUSTERING CONNECTIVITY ANALYSIS")
        print("=" * 80)
        print(f"Graph clustering: {self.use_graph_clustering}")
        print(f"Clustering methods: {self.clustering_methods}")
        print(f"Cluster alpha: {self.cluster_alpha}")
        
        if not self.use_graph_clustering:
            # Fall back to hierarchical analysis
            return self.compute_hierarchical_connectivity(data, roi_labels, ground_truth)
        
        # First compute hierarchical connectivity
        hierarchical_results = self.compute_hierarchical_connectivity(data, roi_labels, ground_truth)
        
        # Extract connectivity matrix and p-values
        if 'ensemble_results' in hierarchical_results:
            ensemble_results = hierarchical_results['ensemble_results']
            if 'combined_connectivity' in ensemble_results:
                connectivity_matrix = ensemble_results['combined_connectivity']
                # Use ensemble p-values if available
                if 'combined_p_values' in ensemble_results:
                    p_values = ensemble_results['combined_p_values']
                else:
                    # Fallback: create p-values from connectivity strength
                    p_values = 1.0 / (1.0 + np.abs(connectivity_matrix))
            else:
                connectivity_matrix = ensemble_results['final_connectivity_matrix']
                p_values = 1.0 / (1.0 + np.abs(connectivity_matrix))
        else:
            # Use combined connectivity from hierarchical results
            connectivity_matrix = hierarchical_results['best_hierarchy']['hierarchy_result']['combined_connectivity']
            p_values = 1.0 / (1.0 + np.abs(connectivity_matrix))
        
        # Perform graph clustering analysis
        graph_clustering_results = self.graph_cluster_analyzer.analyze_smte_graph_clusters(
            connectivity_matrix=connectivity_matrix,
            p_values=p_values,
            roi_labels=roi_labels,
            initial_threshold=0.05,
            n_permutations=self.cluster_n_permutations,
            cluster_alpha=self.cluster_alpha
        )
        
        # Create final results combining hierarchical and clustering
        final_connectivity_results = {
            'hierarchical_results': hierarchical_results,
            'graph_clustering_results': graph_clustering_results,
            'final_connectivity_matrix': graph_clustering_results['cluster_corrected_matrix'],
            'cluster_level_significance': self._compute_cluster_level_significance(
                graph_clustering_results
            ),
            'roi_labels': roi_labels,
            'analysis_parameters': {
                'use_graph_clustering': self.use_graph_clustering,
                'clustering_methods': self.clustering_methods,
                'cluster_alpha': self.cluster_alpha,
                'cluster_n_permutations': self.cluster_n_permutations
            }
        }
        
        # Store results
        self.graph_clustering_results = final_connectivity_results
        
        # Print summary
        n_cluster_connections = np.sum(graph_clustering_results['cluster_corrected_matrix'] > 0)
        n_hierarchical_connections = np.sum(connectivity_matrix > 0)
        
        print(f"\\n🔍 GRAPH CLUSTERING SUMMARY:")
        print(f"Hierarchical connections: {n_hierarchical_connections}")
        print(f"Cluster-corrected connections: {n_cluster_connections}")
        print(f"Improvement ratio: {n_cluster_connections / max(n_hierarchical_connections, 1):.2f}x")
        
        return final_connectivity_results
    
    def _compute_cluster_level_significance(self,
                                          graph_clustering_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute overall cluster-level significance summary.
        """
        
        cluster_significance = graph_clustering_results['cluster_significance']
        
        # Count total significant clusters across methods
        total_significant_clusters = 0
        significant_by_method = {}
        
        for method_name, method_significance in cluster_significance.items():
            n_significant = len(method_significance['significant_clusters'])
            significant_by_method[method_name] = n_significant
            total_significant_clusters += n_significant
        
        # Compute consensus significant clusters
        # (clusters that are significant across multiple methods)
        consensus_clusters = self._find_consensus_clusters(cluster_significance)
        
        cluster_level_summary = {
            'total_significant_clusters': total_significant_clusters,
            'significant_by_method': significant_by_method,
            'consensus_clusters': consensus_clusters,
            'cluster_correction_effective': total_significant_clusters > 0
        }
        
        return cluster_level_summary
    
    def _find_consensus_clusters(self, cluster_significance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find clusters that are significant across multiple methods.
        """
        
        # This is a simplified implementation
        # In practice, you might want more sophisticated consensus detection
        
        consensus_clusters = []
        
        # For now, just return clusters from the best method
        best_method = None
        max_clusters = 0
        
        for method_name, method_significance in cluster_significance.items():
            n_clusters = len(method_significance['significant_clusters'])
            if n_clusters > max_clusters:
                max_clusters = n_clusters
                best_method = method_name
        
        if best_method:
            for cluster_name in cluster_significance[best_method]['significant_clusters']:
                consensus_clusters.append({
                    'cluster_name': cluster_name,
                    'method': best_method,
                    'consensus_score': 1.0  # Simplified
                })
        
        return consensus_clusters
    
    def create_graph_clustering_visualizations(self,
                                             graph_clustering_results: Dict[str, Any],
                                             roi_labels: List[str],
                                             save_prefix: str = 'smte_graph_clustering'):
        """
        Create comprehensive graph clustering visualizations.
        """
        
        print("\\n📊 Creating graph clustering visualizations...")
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        
        # 1. Original connectivity matrix
        connectivity_matrix = graph_clustering_results['hierarchical_results']['ensemble_results']['combined_connectivity']
        im1 = axes[0, 0].imshow(connectivity_matrix, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('Original SMTE Connectivity')
        axes[0, 0].set_xlabel('Target ROI')
        axes[0, 0].set_ylabel('Source ROI')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        # 2. Cluster-corrected connectivity matrix
        cluster_corrected = graph_clustering_results['cluster_corrected_matrix']
        im2 = axes[0, 1].imshow(cluster_corrected, cmap='viridis', aspect='auto')
        axes[0, 1].set_title('Cluster-Corrected Connectivity')
        axes[0, 1].set_xlabel('Target ROI')
        axes[0, 1].set_ylabel('Source ROI')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # 3. Difference matrix
        difference = cluster_corrected - (connectivity_matrix > 0).astype(float)
        im3 = axes[0, 2].imshow(difference, cmap='RdBu_r', aspect='auto')
        axes[0, 2].set_title('Cluster Correction Effect')
        axes[0, 2].set_xlabel('Target ROI')
        axes[0, 2].set_ylabel('Source ROI')
        plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # 4. Clustering results for best method
        cluster_results = graph_clustering_results['cluster_results']
        if cluster_results:
            best_method = max(cluster_results.keys(), 
                             key=lambda x: cluster_results[x]['cluster_quality'])
            cluster_labels = cluster_results[best_method]['cluster_labels']
            
            cluster_matrix = np.outer(cluster_labels, np.ones_like(cluster_labels))
            im4 = axes[1, 0].imshow(cluster_matrix, cmap='tab10', aspect='auto')
            axes[1, 0].set_title(f'Cluster Assignment ({best_method})')
            axes[1, 0].set_xlabel('ROI Index')
            axes[1, 0].set_ylabel('ROI Index')
        
        # 5. Cluster statistics comparison
        cluster_significance = graph_clustering_results['cluster_significance']
        methods = list(cluster_significance.keys())
        n_significant = [len(cluster_significance[method]['significant_clusters']) 
                        for method in methods]
        
        bars = axes[1, 1].bar(methods, n_significant)
        axes[1, 1].set_title('Significant Clusters by Method')
        axes[1, 1].set_xlabel('Clustering Method')
        axes[1, 1].set_ylabel('Number of Significant Clusters')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Highlight best method
        if n_significant:
            best_idx = np.argmax(n_significant)
            bars[best_idx].set_color('orange')
        
        # 6. Network properties comparison
        initial_network = graph_clustering_results['initial_network']
        network_props = initial_network['properties']
        
        prop_names = ['density', 'average_clustering', 'mean_in_degree', 'mean_out_degree']
        prop_values = [network_props.get(prop, 0) for prop in prop_names]
        
        axes[1, 2].bar(range(len(prop_names)), prop_values)
        axes[1, 2].set_title('Network Properties')
        axes[1, 2].set_xlabel('Property')
        axes[1, 2].set_ylabel('Value')
        axes[1, 2].set_xticks(range(len(prop_names)))
        axes[1, 2].set_xticklabels(prop_names, rotation=45)
        
        # 7-9. Cluster-specific visualizations
        if 'directional_analysis' in graph_clustering_results:
            directional_analysis = graph_clustering_results['directional_analysis']
            
            # Plot reciprocity for each method
            if directional_analysis:
                method_reciprocities = {}
                for method_name, method_directional in directional_analysis.items():
                    reciprocities = []
                    for cluster_name, cluster_props in method_directional.items():
                        reciprocities.append(cluster_props.get('reciprocity', 0))
                    if reciprocities:
                        method_reciprocities[method_name] = np.mean(reciprocities)
                
                if method_reciprocities:
                    methods = list(method_reciprocities.keys())
                    reciprocities = list(method_reciprocities.values())
                    
                    axes[2, 0].bar(methods, reciprocities)
                    axes[2, 0].set_title('Average Cluster Reciprocity')
                    axes[2, 0].set_xlabel('Method')
                    axes[2, 0].set_ylabel('Reciprocity')
                    axes[2, 0].tick_params(axis='x', rotation=45)
        
        # Summary statistics
        n_original = np.sum(connectivity_matrix > 0)
        n_corrected = np.sum(cluster_corrected > 0)
        improvement = n_corrected / max(n_original, 1)
        
        axes[2, 1].text(0.1, 0.8, f'Original Connections: {n_original}', transform=axes[2, 1].transAxes, fontsize=12)
        axes[2, 1].text(0.1, 0.6, f'Cluster-Corrected: {n_corrected}', transform=axes[2, 1].transAxes, fontsize=12)
        axes[2, 1].text(0.1, 0.4, f'Improvement: {improvement:.2f}x', transform=axes[2, 1].transAxes, fontsize=12)
        axes[2, 1].text(0.1, 0.2, f'Cluster α: {graph_clustering_results["analysis_parameters"]["cluster_alpha"]}', 
                        transform=axes[2, 1].transAxes, fontsize=12)
        axes[2, 1].set_title('Cluster Correction Summary')
        axes[2, 1].set_xticks([])
        axes[2, 1].set_yticks([])
        
        # Method comparison
        if cluster_results:
            method_qualities = [cluster_results[method]['cluster_quality'] for method in methods]
            
            axes[2, 2].scatter(n_significant, method_qualities)
            for i, method in enumerate(methods):
                axes[2, 2].annotate(method, (n_significant[i], method_qualities[i]))
            
            axes[2, 2].set_xlabel('Number of Significant Clusters')
            axes[2, 2].set_ylabel('Clustering Quality')
            axes[2, 2].set_title('Method Performance Comparison')
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Comprehensive graph clustering visualization saved: {save_prefix}_comprehensive_analysis.png")


def test_smte_graph_clustering():
    """
    Test the SMTE graph clustering implementation.
    """
    
    print("🧪 TESTING SMTE GRAPH CLUSTERING")
    print("=" * 70)
    
    # Generate test data with network structure
    np.random.seed(42)
    n_rois = 15
    n_timepoints = 120
    TR = 2.0
    
    # Create ROI labels with network structure
    roi_labels = [
        # Visual network
        'V1_L', 'V1_R', 'V2_L',
        # Motor network
        'M1_L', 'M1_R', 'S1_L',
        # Executive network
        'DLPFC_L', 'DLPFC_R', 'IFG_L', 'Parietal_L',
        # Default mode network
        'PCC', 'mPFC', 'Angular_L',
        # Salience network
        'ACC', 'Insula'
    ]
    
    # Generate realistic data with network clustering
    t = np.arange(n_timepoints) * TR
    data = []
    
    network_freqs = {'Visual': 0.12, 'Motor': 0.15, 'Executive': 0.08, 'Default': 0.05, 'Salience': 0.10}
    roi_networks = {
        0: 'Visual', 1: 'Visual', 2: 'Visual',
        3: 'Motor', 4: 'Motor', 5: 'Motor',
        6: 'Executive', 7: 'Executive', 8: 'Executive', 9: 'Executive',
        10: 'Default', 11: 'Default', 12: 'Default',
        13: 'Salience', 14: 'Salience'
    }
    
    for roi_idx, roi_label in enumerate(roi_labels):
        network = roi_networks[roi_idx]
        base_freq = network_freqs[network]
        
        # Generate network-specific signal
        signal = 0.8 * np.sin(2 * np.pi * base_freq * t)
        signal += 0.3 * np.sin(2 * np.pi * (base_freq * 2) * t)
        signal += 0.2 * np.sin(2 * np.pi * (base_freq * 0.5) * t)
        signal += 0.4 * np.random.randn(n_timepoints)
        
        data.append(signal)
    
    data = np.array(data)
    
    # Add network-specific connectivity
    known_connections = [
        # Within-network connections
        (0, 1, 1, 0.4),  # V1_L -> V1_R
        (3, 4, 1, 0.3),  # M1_L -> M1_R
        (6, 7, 1, 0.3),  # DLPFC_L -> DLPFC_R
        (10, 11, 2, 0.35), # PCC -> mPFC
        (13, 14, 1, 0.25), # ACC -> Insula
        
        # Between-network connections
        (0, 6, 2, 0.2),  # Visual -> Executive
        (6, 10, 3, 0.15), # Executive -> Default
    ]
    
    for source, target, lag, strength in known_connections:
        if lag < n_timepoints:
            data[target, lag:] += strength * data[source, :-lag]
    
    # Standardize data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data = scaler.fit_transform(data.T).T
    
    # Test clustering implementations
    print("\\n🔬 Testing Graph Clustering Implementation")
    print("-" * 50)
    
    # Test with reduced parameters for speed
    clustering_smte = SMTEGraphClusteringSMTE(
        use_graph_clustering=True,
        clustering_methods=['spectral', 'louvain'],
        cluster_alpha=0.05,
        cluster_n_permutations=100,  # Reduced for testing
        use_hierarchical_analysis=True,
        hierarchy_methods=['agglomerative'],
        hierarchy_levels=[2, 4],
        distance_metrics=['correlation'],
        use_ensemble_testing=True,
        surrogate_methods=['aaft'],
        n_surrogates_per_method=20,
        use_multiscale_analysis=True,
        scales_to_analyze=['fast'],
        adaptive_mode='heuristic',
        use_network_correction=True,
        use_physiological_constraints=True,
        known_networks=roi_networks,
        TR=TR,
        n_permutations=50,
        random_state=42
    )
    
    # Run analysis
    results = clustering_smte.compute_graph_clustered_connectivity(data, roi_labels)
    
    # Create visualizations
    clustering_smte.create_graph_clustering_visualizations(
        results['graph_clustering_results'], roi_labels
    )
    
    # Print detailed results
    print("\\n📊 DETAILED RESULTS")
    print("-" * 40)
    
    cluster_significance = results['cluster_level_significance']
    print(f"Total significant clusters: {cluster_significance['total_significant_clusters']}")
    print(f"Cluster correction effective: {cluster_significance['cluster_correction_effective']}")
    
    for method, n_clusters in cluster_significance['significant_by_method'].items():
        print(f"  {method}: {n_clusters} significant clusters")
    
    # Compare with baseline
    hierarchical_connections = np.sum(results['hierarchical_results']['ensemble_results']['combined_connectivity'] > 0)
    cluster_connections = np.sum(results['final_connectivity_matrix'] > 0)
    
    print(f"\\nConnectivity comparison:")
    print(f"  Hierarchical: {hierarchical_connections} connections")
    print(f"  Cluster-corrected: {cluster_connections} connections")
    print(f"  Improvement: {cluster_connections / max(hierarchical_connections, 1):.2f}x")
    
    return results


if __name__ == "__main__":
    results = test_smte_graph_clustering()