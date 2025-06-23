#!/usr/bin/env python3
"""
Phase 2.3: Hierarchical Connectivity Analysis for SMTE
This module implements hierarchical decomposition of brain connectivity patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
import logging
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import seaborn as sns

from ensemble_smte_v1 import EnsembleSMTE

logging.basicConfig(level=logging.INFO)


class HierarchicalAnalyzer:
    """
    Implements hierarchical decomposition of brain connectivity patterns.
    """
    
    def __init__(self, 
                 clustering_methods: List[str] = ['agglomerative', 'spectral', 'modularity'],
                 hierarchy_levels: List[int] = [2, 4, 8],
                 distance_metrics: List[str] = ['correlation', 'euclidean', 'cosine']):
        
        self.clustering_methods = clustering_methods
        self.hierarchy_levels = hierarchy_levels
        self.distance_metrics = distance_metrics
        
        # Available clustering methods
        self.available_methods = {
            'agglomerative': self._agglomerative_clustering,
            'spectral': self._spectral_clustering,
            'modularity': self._modularity_clustering,
            'kmeans': self._kmeans_clustering
        }
        
        # Available distance metrics
        self.distance_functions = {
            'correlation': self._correlation_distance,
            'euclidean': self._euclidean_distance,
            'cosine': self._cosine_distance,
            'connectivity': self._connectivity_distance
        }
    
    def build_connectivity_hierarchy(self,
                                   connectivity_matrix: np.ndarray,
                                   roi_labels: List[str],
                                   method: str = 'agglomerative',
                                   distance_metric: str = 'correlation') -> Dict[str, Any]:
        """
        Build hierarchical decomposition of connectivity patterns.
        
        Parameters:
        -----------
        connectivity_matrix : np.ndarray
            Connectivity matrix (n_rois x n_rois)
        roi_labels : List[str]
            Labels for ROIs
        method : str
            Clustering method to use
        distance_metric : str
            Distance metric for clustering
            
        Returns:
        --------
        Dict[str, Any]
            Hierarchical decomposition results
        """
        
        print(f"Building connectivity hierarchy using {method} clustering with {distance_metric} distance...")
        
        if method not in self.available_methods:
            raise ValueError(f"Unknown clustering method: {method}. Available: {list(self.available_methods.keys())}")
        
        n_rois = len(roi_labels)
        
        # Compute distance matrix
        distance_matrix = self._compute_distance_matrix(connectivity_matrix, distance_metric)
        
        # Build hierarchy for different levels
        hierarchy_results = {}
        
        for n_clusters in self.hierarchy_levels:
            if n_clusters <= n_rois:
                level_result = self._cluster_at_level(
                    connectivity_matrix, distance_matrix, roi_labels, 
                    method, n_clusters
                )
                hierarchy_results[f'level_{n_clusters}'] = level_result
        
        # Compute overall hierarchy structure
        if method == 'agglomerative':
            linkage_matrix = linkage(squareform(distance_matrix), method='ward')
            dendrogram_data = self._create_dendrogram_data(linkage_matrix, roi_labels)
        else:
            linkage_matrix = None
            dendrogram_data = None
        
        # Analyze hierarchy stability
        stability_analysis = self._analyze_hierarchy_stability(
            connectivity_matrix, distance_matrix, roi_labels, method
        )
        
        # Compute inter-level consistency
        consistency_analysis = self._analyze_level_consistency(hierarchy_results)
        
        hierarchy_summary = {
            'method': method,
            'distance_metric': distance_metric,
            'n_rois': n_rois,
            'hierarchy_levels': list(hierarchy_results.keys()),
            'level_results': hierarchy_results,
            'linkage_matrix': linkage_matrix,
            'dendrogram_data': dendrogram_data,
            'stability_analysis': stability_analysis,
            'consistency_analysis': consistency_analysis,
            'distance_matrix': distance_matrix
        }
        
        return hierarchy_summary
    
    def _compute_distance_matrix(self,
                               connectivity_matrix: np.ndarray,
                               distance_metric: str) -> np.ndarray:
        """Compute distance matrix using specified metric."""
        
        if distance_metric not in self.distance_functions:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
        
        distance_func = self.distance_functions[distance_metric]
        return distance_func(connectivity_matrix)
    
    def _correlation_distance(self, connectivity_matrix: np.ndarray) -> np.ndarray:
        """Compute correlation-based distance matrix."""
        
        # Use connectivity profiles as features
        correlation_matrix = np.corrcoef(connectivity_matrix)
        
        # Handle NaN values
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Convert correlation to distance (1 - |correlation|)
        distance_matrix = 1.0 - np.abs(correlation_matrix)
        
        # Ensure diagonal is zero
        np.fill_diagonal(distance_matrix, 0.0)
        
        return distance_matrix
    
    def _euclidean_distance(self, connectivity_matrix: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance matrix."""
        
        # Use connectivity profiles as features
        distances = pdist(connectivity_matrix, metric='euclidean')
        distance_matrix = squareform(distances)
        
        return distance_matrix
    
    def _cosine_distance(self, connectivity_matrix: np.ndarray) -> np.ndarray:
        """Compute cosine distance matrix."""
        
        # Use connectivity profiles as features
        distances = pdist(connectivity_matrix, metric='cosine')
        distance_matrix = squareform(distances)
        
        # Handle NaN values
        distance_matrix = np.nan_to_num(distance_matrix, nan=1.0)
        
        return distance_matrix
    
    def _connectivity_distance(self, connectivity_matrix: np.ndarray) -> np.ndarray:
        """Compute connectivity-based distance matrix."""
        
        # Use negative connectivity strength as distance
        # (higher connectivity = lower distance)
        distance_matrix = 1.0 / (1.0 + np.abs(connectivity_matrix))
        
        # Ensure diagonal is zero
        np.fill_diagonal(distance_matrix, 0.0)
        
        return distance_matrix
    
    def _cluster_at_level(self,
                        connectivity_matrix: np.ndarray,
                        distance_matrix: np.ndarray,
                        roi_labels: List[str],
                        method: str,
                        n_clusters: int) -> Dict[str, Any]:
        """Perform clustering at specific hierarchy level."""
        
        clustering_func = self.available_methods[method]
        cluster_labels = clustering_func(distance_matrix, n_clusters)
        
        # Compute clustering quality metrics
        if n_clusters > 1 and n_clusters < len(roi_labels):
            silhouette = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
        else:
            silhouette = 0.0
        
        # Analyze cluster characteristics
        cluster_analysis = self._analyze_clusters(
            connectivity_matrix, cluster_labels, roi_labels, n_clusters
        )
        
        # Compute within-cluster and between-cluster connectivity
        connectivity_analysis = self._analyze_cluster_connectivity(
            connectivity_matrix, cluster_labels, n_clusters
        )
        
        level_result = {
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels,
            'silhouette_score': silhouette,
            'cluster_analysis': cluster_analysis,
            'connectivity_analysis': connectivity_analysis
        }
        
        return level_result
    
    def _agglomerative_clustering(self, distance_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform agglomerative clustering."""
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(distance_matrix)
        return cluster_labels
    
    def _spectral_clustering(self, distance_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform spectral clustering."""
        
        from sklearn.cluster import SpectralClustering
        
        # Convert distance to similarity
        sigma = np.median(distance_matrix[distance_matrix > 0])
        similarity_matrix = np.exp(-distance_matrix / (2 * sigma**2))
        
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        
        cluster_labels = clustering.fit_predict(similarity_matrix)
        return cluster_labels
    
    def _modularity_clustering(self, distance_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform modularity-based clustering using network analysis."""
        
        # Convert distance to similarity
        similarity_matrix = 1.0 - distance_matrix
        
        # Create NetworkX graph
        G = nx.from_numpy_array(similarity_matrix)
        
        # Use community detection algorithms
        try:
            import networkx.algorithms.community as nx_comm
            
            # Use Louvain algorithm and then merge communities to target number
            communities = nx_comm.louvain_communities(G, seed=42)
            
            # If we have more communities than desired, merge smallest ones
            while len(communities) > n_clusters:
                # Find two smallest communities
                sizes = [len(c) for c in communities]
                smallest_idx = np.argmin(sizes)
                remaining_sizes = [sizes[i] for i in range(len(sizes)) if i != smallest_idx]
                second_smallest_idx = sizes.index(min(remaining_sizes))
                
                # Merge communities
                merged_community = communities[smallest_idx] | communities[second_smallest_idx]
                new_communities = []
                for i, c in enumerate(communities):
                    if i != smallest_idx and i != second_smallest_idx:
                        new_communities.append(c)
                new_communities.append(merged_community)
                communities = new_communities
            
            # Convert to cluster labels
            cluster_labels = np.zeros(distance_matrix.shape[0], dtype=int)
            for cluster_id, community in enumerate(communities):
                for node in community:
                    cluster_labels[node] = cluster_id
                    
        except ImportError:
            # Fall back to simple clustering if networkx community detection not available
            cluster_labels = self._kmeans_clustering(distance_matrix, n_clusters)
        
        return cluster_labels
    
    def _kmeans_clustering(self, distance_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform K-means clustering."""
        
        # Use MDS to embed in Euclidean space, then apply K-means
        from sklearn.manifold import MDS
        
        mds = MDS(n_components=min(10, distance_matrix.shape[0]-1), 
                  dissimilarity='precomputed', random_state=42)
        embedded_data = mds.fit_transform(distance_matrix)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embedded_data)
        
        return cluster_labels
    
    def _analyze_clusters(self,
                        connectivity_matrix: np.ndarray,
                        cluster_labels: np.ndarray,
                        roi_labels: List[str],
                        n_clusters: int) -> Dict[str, Any]:
        """Analyze characteristics of identified clusters."""
        
        cluster_analysis = {}
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_rois = [roi_labels[i] for i in range(len(roi_labels)) if cluster_mask[i]]
            cluster_indices = np.where(cluster_mask)[0]
            
            # Compute cluster statistics
            if len(cluster_indices) > 1:
                # Within-cluster connectivity
                within_cluster_conn = connectivity_matrix[np.ix_(cluster_indices, cluster_indices)]
                within_cluster_mean = np.mean(within_cluster_conn[np.triu_indices_from(within_cluster_conn, k=1)])
                within_cluster_std = np.std(within_cluster_conn[np.triu_indices_from(within_cluster_conn, k=1)])
                
                # Cluster coherence (average correlation)
                cluster_coherence = np.mean(np.corrcoef(connectivity_matrix[cluster_indices]))
            else:
                within_cluster_mean = 0.0
                within_cluster_std = 0.0
                cluster_coherence = 1.0
            
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'size': len(cluster_rois),
                'roi_labels': cluster_rois,
                'roi_indices': cluster_indices.tolist(),
                'within_cluster_connectivity_mean': within_cluster_mean,
                'within_cluster_connectivity_std': within_cluster_std,
                'cluster_coherence': cluster_coherence
            }
        
        return cluster_analysis
    
    def _analyze_cluster_connectivity(self,
                                    connectivity_matrix: np.ndarray,
                                    cluster_labels: np.ndarray,
                                    n_clusters: int) -> Dict[str, Any]:
        """Analyze connectivity patterns between and within clusters."""
        
        # Compute cluster-level connectivity matrix
        cluster_connectivity = np.zeros((n_clusters, n_clusters))
        cluster_sizes = np.zeros(n_clusters)
        
        for i in range(n_clusters):
            cluster_i_mask = cluster_labels == i
            cluster_i_indices = np.where(cluster_i_mask)[0]
            cluster_sizes[i] = len(cluster_i_indices)
            
            for j in range(n_clusters):
                cluster_j_mask = cluster_labels == j
                cluster_j_indices = np.where(cluster_j_mask)[0]
                
                if i == j:
                    # Within-cluster connectivity
                    if len(cluster_i_indices) > 1:
                        within_cluster = connectivity_matrix[np.ix_(cluster_i_indices, cluster_i_indices)]
                        cluster_connectivity[i, j] = np.mean(within_cluster[np.triu_indices_from(within_cluster, k=1)])
                    else:
                        cluster_connectivity[i, j] = 0.0
                else:
                    # Between-cluster connectivity
                    between_cluster = connectivity_matrix[np.ix_(cluster_i_indices, cluster_j_indices)]
                    cluster_connectivity[i, j] = np.mean(between_cluster)
        
        # Compute modularity-like metric
        within_cluster_strength = np.sum(np.diag(cluster_connectivity) * cluster_sizes)
        total_strength = np.sum(cluster_connectivity * np.outer(cluster_sizes, cluster_sizes))
        
        if total_strength > 0:
            modularity = within_cluster_strength / total_strength
        else:
            modularity = 0.0
        
        connectivity_analysis = {
            'cluster_connectivity_matrix': cluster_connectivity,
            'cluster_sizes': cluster_sizes,
            'modularity': modularity,
            'within_cluster_total': within_cluster_strength,
            'total_connectivity': total_strength
        }
        
        return connectivity_analysis
    
    def _create_dendrogram_data(self, linkage_matrix: np.ndarray, roi_labels: List[str]) -> Dict[str, Any]:
        """Create dendrogram data structure."""
        
        dendrogram_data = {
            'linkage_matrix': linkage_matrix,
            'roi_labels': roi_labels,
            'n_merges': len(linkage_matrix)
        }
        
        return dendrogram_data
    
    def _analyze_hierarchy_stability(self,
                                   connectivity_matrix: np.ndarray,
                                   distance_matrix: np.ndarray,
                                   roi_labels: List[str],
                                   method: str,
                                   n_bootstrap: int = 50) -> Dict[str, Any]:
        """Analyze stability of hierarchical clustering through bootstrap sampling."""
        
        print("Analyzing hierarchy stability...")
        
        n_rois = len(roi_labels)
        stability_scores = {}
        
        # Bootstrap stability analysis
        for n_clusters in self.hierarchy_levels:
            if n_clusters <= n_rois:
                cluster_similarities = []
                
                # Original clustering
                original_labels = self._cluster_at_level(
                    connectivity_matrix, distance_matrix, roi_labels, method, n_clusters
                )['cluster_labels']
                
                # Bootstrap samples
                for boot_iter in range(min(n_bootstrap, 20)):  # Limit for validation speed
                    # Resample ROIs
                    boot_indices = np.random.choice(n_rois, size=n_rois, replace=True)
                    boot_connectivity = connectivity_matrix[np.ix_(boot_indices, boot_indices)]
                    boot_distance = self._compute_distance_matrix(boot_connectivity, 'correlation')
                    
                    # Cluster bootstrap sample
                    boot_labels = self._cluster_at_level(
                        boot_connectivity, boot_distance, 
                        [roi_labels[i] for i in boot_indices], method, n_clusters
                    )['cluster_labels']
                    
                    # Map back to original indices and compute similarity
                    mapped_labels = np.zeros(n_rois, dtype=int)
                    for i, orig_idx in enumerate(boot_indices):
                        mapped_labels[orig_idx] = boot_labels[i]
                    
                    # Compute adjusted rand index
                    ari = adjusted_rand_score(original_labels, mapped_labels)
                    cluster_similarities.append(ari)
                
                stability_scores[f'level_{n_clusters}'] = {
                    'mean_ari': np.mean(cluster_similarities),
                    'std_ari': np.std(cluster_similarities),
                    'min_ari': np.min(cluster_similarities),
                    'max_ari': np.max(cluster_similarities)
                }
        
        stability_analysis = {
            'method': method,
            'n_bootstrap_samples': min(n_bootstrap, 20),
            'stability_scores': stability_scores
        }
        
        return stability_analysis
    
    def _analyze_level_consistency(self, hierarchy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consistency between different hierarchy levels."""
        
        print("Analyzing inter-level consistency...")
        
        consistency_analysis = {}
        level_names = sorted(hierarchy_results.keys())
        
        # Pairwise consistency between levels
        for i, level1 in enumerate(level_names):
            for j, level2 in enumerate(level_names[i+1:], i+1):
                labels1 = hierarchy_results[level1]['cluster_labels']
                labels2 = hierarchy_results[level2]['cluster_labels']
                
                # Compute adjusted rand index
                ari = adjusted_rand_score(labels1, labels2)
                
                consistency_analysis[f'{level1}_vs_{level2}'] = {
                    'adjusted_rand_index': ari,
                    'n_clusters_1': hierarchy_results[level1]['n_clusters'],
                    'n_clusters_2': hierarchy_results[level2]['n_clusters']
                }
        
        # Overall consistency score
        if len(level_names) > 1:
            all_aris = [info['adjusted_rand_index'] for info in consistency_analysis.values()]
            overall_consistency = np.mean(all_aris)
        else:
            overall_consistency = 1.0
        
        consistency_analysis['overall_consistency'] = overall_consistency
        
        return consistency_analysis


class HierarchicalSMTE(EnsembleSMTE):
    """
    SMTE implementation with hierarchical connectivity analysis.
    """
    
    def __init__(self,
                 use_hierarchical_analysis: bool = True,
                 hierarchy_methods: List[str] = ['agglomerative', 'spectral'],
                 hierarchy_levels: List[int] = [2, 4, 6],
                 distance_metrics: List[str] = ['correlation', 'euclidean'],
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.use_hierarchical_analysis = use_hierarchical_analysis
        self.hierarchy_methods = hierarchy_methods
        self.hierarchy_levels = hierarchy_levels
        self.distance_metrics = distance_metrics
        
        # Initialize hierarchical analyzer
        self.hierarchical_analyzer = HierarchicalAnalyzer(
            clustering_methods=hierarchy_methods,
            hierarchy_levels=hierarchy_levels,
            distance_metrics=distance_metrics
        )
        
        # Store hierarchical results
        self.hierarchical_results = None
    
    def compute_hierarchical_connectivity(self,
                                        data: np.ndarray,
                                        roi_labels: List[str],
                                        ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute connectivity with hierarchical analysis.
        """
        
        print("Computing hierarchical SMTE connectivity...")
        print(f"Hierarchy methods: {self.hierarchy_methods}")
        print(f"Hierarchy levels: {self.hierarchy_levels}")
        
        if not self.use_hierarchical_analysis:
            # Fall back to ensemble analysis
            return self.compute_ensemble_connectivity(data, roi_labels, ground_truth)
        
        # First compute ensemble connectivity
        ensemble_results = self.compute_ensemble_connectivity(data, roi_labels, ground_truth)
        
        # Extract final connectivity matrix
        if 'final_connectivity_matrix' in ensemble_results:
            connectivity_matrix = ensemble_results['final_connectivity_matrix']
        elif 'combined_connectivity' in ensemble_results:
            connectivity_matrix = ensemble_results['combined_connectivity']
        else:
            raise ValueError("No connectivity matrix found in ensemble results")
        
        # Perform hierarchical analysis for each method and distance metric
        hierarchical_analyses = {}
        
        for method in self.hierarchy_methods:
            method_results = {}
            
            for distance_metric in self.distance_metrics:
                print(f"\\nAnalyzing hierarchy: {method} with {distance_metric} distance")
                
                hierarchy_result = self.hierarchical_analyzer.build_connectivity_hierarchy(
                    connectivity_matrix, roi_labels, method, distance_metric
                )
                
                method_results[distance_metric] = hierarchy_result
            
            hierarchical_analyses[method] = method_results
        
        # Compare and combine hierarchical results
        hierarchy_comparison = self._compare_hierarchical_methods(hierarchical_analyses)
        
        # Select best hierarchy
        best_hierarchy = self._select_best_hierarchy(hierarchical_analyses, hierarchy_comparison)
        
        # Create comprehensive hierarchical connectivity analysis
        hierarchical_connectivity_results = {
            'ensemble_results': ensemble_results,
            'hierarchical_analyses': hierarchical_analyses,
            'hierarchy_comparison': hierarchy_comparison,
            'best_hierarchy': best_hierarchy,
            'final_hierarchy': best_hierarchy['hierarchy_result'],
            'hierarchy_summary': self._create_hierarchy_summary(hierarchical_analyses),
            'roi_labels': roi_labels
        }
        
        # Store results
        self.hierarchical_results = hierarchical_connectivity_results
        
        print(f"\\nHierarchical analysis complete:")
        print(f"Best method: {best_hierarchy['method']} with {best_hierarchy['distance_metric']} distance")
        print(f"Hierarchy levels analyzed: {self.hierarchy_levels}")
        
        return hierarchical_connectivity_results
    
    def _compare_hierarchical_methods(self, hierarchical_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Compare different hierarchical clustering methods."""
        
        print("Comparing hierarchical methods...")
        
        comparison_results = {}
        
        # Extract stability and consistency scores for each method/distance combination
        for method, method_results in hierarchical_analyses.items():
            method_comparison = {}
            
            for distance_metric, hierarchy_result in method_results.items():
                stability_analysis = hierarchy_result['stability_analysis']
                consistency_analysis = hierarchy_result['consistency_analysis']
                
                # Compute average stability across levels
                stability_scores = []
                for level_name, level_stability in stability_analysis['stability_scores'].items():
                    stability_scores.append(level_stability['mean_ari'])
                
                avg_stability = np.mean(stability_scores) if stability_scores else 0.0
                overall_consistency = consistency_analysis['overall_consistency']
                
                # Compute silhouette scores across levels
                silhouette_scores = []
                for level_name, level_result in hierarchy_result['level_results'].items():
                    silhouette_scores.append(level_result['silhouette_score'])
                
                avg_silhouette = np.mean(silhouette_scores) if silhouette_scores else 0.0
                
                method_comparison[distance_metric] = {
                    'average_stability': avg_stability,
                    'overall_consistency': overall_consistency,
                    'average_silhouette': avg_silhouette,
                    'composite_score': (avg_stability + overall_consistency + avg_silhouette) / 3.0
                }
            
            comparison_results[method] = method_comparison
        
        # Find best method/distance combination
        best_score = -1.0
        best_method = None
        best_distance = None
        
        for method, method_results in comparison_results.items():
            for distance_metric, scores in method_results.items():
                if scores['composite_score'] > best_score:
                    best_score = scores['composite_score']
                    best_method = method
                    best_distance = distance_metric
        
        comparison_results['best_combination'] = {
            'method': best_method,
            'distance_metric': best_distance,
            'composite_score': best_score
        }
        
        return comparison_results
    
    def _select_best_hierarchy(self, 
                             hierarchical_analyses: Dict[str, Any],
                             hierarchy_comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best hierarchical clustering result."""
        
        best_combo = hierarchy_comparison['best_combination']
        best_method = best_combo['method']
        best_distance = best_combo['distance_metric']
        
        best_hierarchy_result = hierarchical_analyses[best_method][best_distance]
        
        best_hierarchy = {
            'method': best_method,
            'distance_metric': best_distance,
            'composite_score': best_combo['composite_score'],
            'hierarchy_result': best_hierarchy_result
        }
        
        return best_hierarchy
    
    def _create_hierarchy_summary(self, hierarchical_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of all hierarchical analyses."""
        
        summary = {
            'methods_tested': list(hierarchical_analyses.keys()),
            'distance_metrics_tested': [],
            'hierarchy_levels': self.hierarchy_levels,
            'method_performance': {}
        }
        
        # Extract distance metrics from first method
        if hierarchical_analyses:
            first_method = list(hierarchical_analyses.keys())[0]
            summary['distance_metrics_tested'] = list(hierarchical_analyses[first_method].keys())
        
        # Summarize performance for each method
        for method, method_results in hierarchical_analyses.items():
            method_summary = {
                'average_stability': [],
                'average_consistency': [],
                'average_silhouette': []
            }
            
            for distance_metric, hierarchy_result in method_results.items():
                # Extract stability
                stability_scores = []
                for level_stability in hierarchy_result['stability_analysis']['stability_scores'].values():
                    stability_scores.append(level_stability['mean_ari'])
                
                if stability_scores:
                    method_summary['average_stability'].append(np.mean(stability_scores))
                
                # Extract consistency
                method_summary['average_consistency'].append(
                    hierarchy_result['consistency_analysis']['overall_consistency']
                )
                
                # Extract silhouette
                silhouette_scores = []
                for level_result in hierarchy_result['level_results'].values():
                    silhouette_scores.append(level_result['silhouette_score'])
                
                if silhouette_scores:
                    method_summary['average_silhouette'].append(np.mean(silhouette_scores))
            
            # Compute overall averages
            summary['method_performance'][method] = {
                'mean_stability': np.mean(method_summary['average_stability']) if method_summary['average_stability'] else 0.0,
                'mean_consistency': np.mean(method_summary['average_consistency']) if method_summary['average_consistency'] else 0.0,
                'mean_silhouette': np.mean(method_summary['average_silhouette']) if method_summary['average_silhouette'] else 0.0
            }
        
        return summary
    
    def analyze_hierarchical_network_organization(self,
                                                hierarchical_results: Dict[str, Any],
                                                roi_labels: List[str]) -> Dict[str, Any]:
        """
        Analyze network organization at different hierarchical levels.
        """
        
        print("Analyzing hierarchical network organization...")
        
        best_hierarchy = hierarchical_results['best_hierarchy']['hierarchy_result']
        level_results = best_hierarchy['level_results']
        
        network_organization = {}
        
        # Analyze each hierarchy level
        for level_name, level_result in level_results.items():
            cluster_labels = level_result['cluster_labels']
            n_clusters = level_result['n_clusters']
            
            # Identify functional networks at this level
            level_networks = self._identify_functional_networks(
                cluster_labels, roi_labels, n_clusters
            )
            
            # Analyze network characteristics
            network_characteristics = self._analyze_network_characteristics(
                level_result, roi_labels
            )
            
            network_organization[level_name] = {
                'n_clusters': n_clusters,
                'functional_networks': level_networks,
                'network_characteristics': network_characteristics,
                'silhouette_score': level_result['silhouette_score']
            }
        
        # Cross-level network analysis
        cross_level_analysis = self._analyze_cross_level_networks(network_organization)
        
        organization_summary = {
            'level_organization': network_organization,
            'cross_level_analysis': cross_level_analysis,
            'hierarchy_method': best_hierarchy['method'],
            'distance_metric': best_hierarchy['distance_metric']
        }
        
        return organization_summary
    
    def _identify_functional_networks(self,
                                    cluster_labels: np.ndarray,
                                    roi_labels: List[str],
                                    n_clusters: int) -> Dict[str, Any]:
        """Identify functional networks from clustering results."""
        
        functional_networks = {}
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_rois = [roi_labels[i] for i in range(len(roi_labels)) if cluster_mask[i]]
            
            # Attempt to identify known functional networks
            network_type = self._classify_network_type(cluster_rois)
            
            functional_networks[f'network_{cluster_id}'] = {
                'roi_labels': cluster_rois,
                'size': len(cluster_rois),
                'predicted_network_type': network_type,
                'cluster_id': cluster_id
            }
        
        return functional_networks
    
    def _classify_network_type(self, roi_labels: List[str]) -> str:
        """Classify network type based on ROI labels."""
        
        # Simple heuristic classification based on common ROI naming patterns
        roi_text = ' '.join(roi_labels).lower()
        
        if any(keyword in roi_text for keyword in ['v1', 'v2', 'visual', 'occipital']):
            return 'visual'
        elif any(keyword in roi_text for keyword in ['m1', 'motor', 'precentral']):
            return 'motor'
        elif any(keyword in roi_text for keyword in ['s1', 'sensory', 'postcentral']):
            return 'sensory'
        elif any(keyword in roi_text for keyword in ['dlpfc', 'frontal', 'executive']):
            return 'executive'
        elif any(keyword in roi_text for keyword in ['pcc', 'mpfc', 'default', 'precuneus']):
            return 'default_mode'
        elif any(keyword in roi_text for keyword in ['acc', 'insula', 'salience']):
            return 'salience'
        elif any(keyword in roi_text for keyword in ['attention', 'parietal']):
            return 'attention'
        else:
            return 'unknown'
    
    def _analyze_network_characteristics(self,
                                       level_result: Dict[str, Any],
                                       roi_labels: List[str]) -> Dict[str, Any]:
        """Analyze characteristics of networks at a given level."""
        
        cluster_analysis = level_result['cluster_analysis']
        connectivity_analysis = level_result['connectivity_analysis']
        
        characteristics = {
            'cluster_sizes': [],
            'within_cluster_connectivity': [],
            'cluster_coherence': [],
            'modularity': connectivity_analysis['modularity']
        }
        
        for cluster_name, cluster_info in cluster_analysis.items():
            characteristics['cluster_sizes'].append(cluster_info['size'])
            characteristics['within_cluster_connectivity'].append(
                cluster_info['within_cluster_connectivity_mean']
            )
            characteristics['cluster_coherence'].append(cluster_info['cluster_coherence'])
        
        # Compute summary statistics
        characteristics['size_distribution'] = {
            'mean': np.mean(characteristics['cluster_sizes']),
            'std': np.std(characteristics['cluster_sizes']),
            'min': np.min(characteristics['cluster_sizes']),
            'max': np.max(characteristics['cluster_sizes'])
        }
        
        characteristics['connectivity_distribution'] = {
            'mean': np.mean(characteristics['within_cluster_connectivity']),
            'std': np.std(characteristics['within_cluster_connectivity'])
        }
        
        return characteristics
    
    def _analyze_cross_level_networks(self, network_organization: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how networks change across hierarchy levels."""
        
        cross_level_analysis = {
            'level_progression': {},
            'network_stability': {},
            'emergence_patterns': {}
        }
        
        level_names = sorted(network_organization.keys())
        
        # Track how modularity and silhouette change across levels
        modularities = []
        silhouettes = []
        n_clusters_list = []
        
        for level_name in level_names:
            level_org = network_organization[level_name]
            modularities.append(level_org['network_characteristics']['modularity'])
            silhouettes.append(level_org['silhouette_score'])
            n_clusters_list.append(level_org['n_clusters'])
        
        cross_level_analysis['level_progression'] = {
            'n_clusters': n_clusters_list,
            'modularity_progression': modularities,
            'silhouette_progression': silhouettes
        }
        
        # Identify optimal number of clusters
        if len(silhouettes) > 1:
            optimal_level_idx = np.argmax(silhouettes)
            optimal_level = level_names[optimal_level_idx]
            optimal_n_clusters = n_clusters_list[optimal_level_idx]
        else:
            optimal_level = level_names[0] if level_names else None
            optimal_n_clusters = n_clusters_list[0] if n_clusters_list else 0
        
        cross_level_analysis['optimal_hierarchy_level'] = {
            'level_name': optimal_level,
            'n_clusters': optimal_n_clusters,
            'silhouette_score': max(silhouettes) if silhouettes else 0.0,
            'modularity': modularities[optimal_level_idx] if modularities else 0.0
        }
        
        return cross_level_analysis
    
    def create_hierarchical_visualizations(self,
                                         hierarchical_results: Dict[str, Any],
                                         roi_labels: List[str],
                                         save_prefix: str = 'hierarchical_smte'):
        """Create comprehensive hierarchical visualizations."""
        
        best_hierarchy = hierarchical_results['best_hierarchy']['hierarchy_result']
        level_results = best_hierarchy['level_results']
        
        # 1. Dendrogram visualization (if available)
        if best_hierarchy['dendrogram_data'] is not None:
            plt.figure(figsize=(12, 8))
            dendrogram_data = best_hierarchy['dendrogram_data']
            linkage_matrix = dendrogram_data['linkage_matrix']
            
            dendrogram(linkage_matrix, labels=roi_labels, orientation='right')
            plt.title(f'Hierarchical Clustering Dendrogram\\n'
                     f'Method: {best_hierarchy["method"]}, Distance: {best_hierarchy["distance_metric"]}')
            plt.xlabel('Distance')
            plt.tight_layout()
            plt.savefig(f'{save_prefix}_dendrogram.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 2. Multi-level clustering visualization
        n_levels = len(level_results)
        fig, axes = plt.subplots(2, n_levels, figsize=(5*n_levels, 10))
        
        if n_levels == 1:
            axes = axes.reshape(2, 1)
        
        ensemble_results = hierarchical_results['ensemble_results']
        connectivity_matrix = ensemble_results.get('combined_connectivity', 
                                                  ensemble_results.get('final_connectivity_matrix'))
        
        for idx, (level_name, level_result) in enumerate(level_results.items()):
            cluster_labels = level_result['cluster_labels']
            n_clusters = level_result['n_clusters']
            
            # Reorder connectivity matrix by clusters
            cluster_order = np.argsort(cluster_labels)
            reordered_connectivity = connectivity_matrix[np.ix_(cluster_order, cluster_order)]
            
            # Plot reordered connectivity
            im1 = axes[0, idx].imshow(reordered_connectivity, cmap='viridis', aspect='auto')
            axes[0, idx].set_title(f'{level_name} ({n_clusters} clusters)\\nConnectivity Matrix')
            
            # Add cluster boundaries
            cluster_boundaries = []
            current_cluster = cluster_labels[cluster_order[0]]
            boundary_pos = 0
            
            for i, roi_idx in enumerate(cluster_order):
                if cluster_labels[roi_idx] != current_cluster:
                    cluster_boundaries.append(i)
                    current_cluster = cluster_labels[roi_idx]
            
            for boundary in cluster_boundaries:
                axes[0, idx].axhline(y=boundary-0.5, color='red', linewidth=2)
                axes[0, idx].axvline(x=boundary-0.5, color='red', linewidth=2)
            
            plt.colorbar(im1, ax=axes[0, idx], fraction=0.046, pad=0.04)
            
            # Plot cluster assignments
            cluster_matrix = np.outer(cluster_labels, np.ones_like(cluster_labels))
            im2 = axes[1, idx].imshow(cluster_matrix, cmap='tab10', aspect='auto')
            axes[1, idx].set_title(f'{level_name}\\nCluster Assignments')
            axes[1, idx].set_xlabel('ROI Index')
            axes[1, idx].set_ylabel('ROI Index')
            
            cbar2 = plt.colorbar(im2, ax=axes[1, idx], fraction=0.046, pad=0.04)
            cbar2.set_label('Cluster ID')
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_multilevel_clustering.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Hierarchy quality metrics
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Silhouette scores across levels
        level_names = list(level_results.keys())
        silhouette_scores = [level_results[level]['silhouette_score'] for level in level_names]
        n_clusters_list = [level_results[level]['n_clusters'] for level in level_names]
        
        axes[0, 0].plot(n_clusters_list, silhouette_scores, 'bo-')
        axes[0, 0].set_xlabel('Number of Clusters')
        axes[0, 0].set_ylabel('Silhouette Score')
        axes[0, 0].set_title('Clustering Quality vs Number of Clusters')
        axes[0, 0].grid(True)
        
        # Modularity scores
        modularity_scores = [level_results[level]['connectivity_analysis']['modularity'] 
                           for level in level_names]
        
        axes[0, 1].plot(n_clusters_list, modularity_scores, 'ro-')
        axes[0, 1].set_xlabel('Number of Clusters')
        axes[0, 1].set_ylabel('Modularity')
        axes[0, 1].set_title('Modularity vs Number of Clusters')
        axes[0, 1].grid(True)
        
        # Stability analysis
        stability_analysis = best_hierarchy['stability_analysis']
        stability_scores = []
        stability_stds = []
        
        for level in level_names:
            level_key = f'level_{level_results[level]["n_clusters"]}'
            if level_key in stability_analysis['stability_scores']:
                stability_info = stability_analysis['stability_scores'][level_key]
                stability_scores.append(stability_info['mean_ari'])
                stability_stds.append(stability_info['std_ari'])
            else:
                stability_scores.append(0.0)
                stability_stds.append(0.0)
        
        axes[1, 0].errorbar(n_clusters_list, stability_scores, yerr=stability_stds, 
                           fmt='go-', capsize=5)
        axes[1, 0].set_xlabel('Number of Clusters')
        axes[1, 0].set_ylabel('Stability (ARI)')
        axes[1, 0].set_title('Clustering Stability')
        axes[1, 0].grid(True)
        
        # Method comparison
        comparison_results = hierarchical_results['hierarchy_comparison']
        methods = list(comparison_results.keys())
        methods = [m for m in methods if m != 'best_combination']
        
        if methods:
            method_scores = []
            for method in methods:
                method_results = comparison_results[method]
                avg_score = np.mean([scores['composite_score'] 
                                   for scores in method_results.values()])
                method_scores.append(avg_score)
            
            bars = axes[1, 1].bar(methods, method_scores)
            axes[1, 1].set_xlabel('Clustering Method')
            axes[1, 1].set_ylabel('Composite Score')
            axes[1, 1].set_title('Method Comparison')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # Highlight best method
            best_method = hierarchical_results['best_hierarchy']['method']
            if best_method in methods:
                best_idx = methods.index(best_method)
                bars[best_idx].set_color('orange')
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_quality_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()


def test_hierarchical_smte():
    """Test the hierarchical SMTE implementation."""
    
    print("Testing Hierarchical SMTE Implementation")
    print("=" * 60)
    
    # Generate realistic test data with hierarchical structure
    np.random.seed(42)
    n_regions = 16
    n_timepoints = 150  # Moderate length for hierarchical analysis
    TR = 2.0
    
    # Create ROI labels with clear network structure
    roi_labels = [
        # Visual network (4 ROIs)
        'V1_L', 'V1_R', 'V2_L', 'V2_R',
        # Motor network (4 ROIs)  
        'M1_L', 'M1_R', 'PMC_L', 'PMC_R',
        # Executive network (4 ROIs)
        'DLPFC_L', 'DLPFC_R', 'IFG_L', 'IFG_R',
        # Default mode network (4 ROIs)
        'PCC', 'mPFC', 'AG_L', 'AG_R'
    ]
    
    # Define ground truth networks
    known_networks = {
        'visual': [0, 1, 2, 3],
        'motor': [4, 5, 6, 7],
        'executive': [8, 9, 10, 11],
        'default': [12, 13, 14, 15]
    }
    
    # Generate time vector
    t = np.arange(n_timepoints) * TR
    
    # Create hierarchical network structure
    data = []
    base_freq = 0.05  # Base frequency
    
    for i, label in enumerate(roi_labels):
        # Determine network membership
        if i < 4:  # Visual
            network_signal = np.sin(2 * np.pi * base_freq * t)
            network_noise = 0.3
        elif i < 8:  # Motor
            network_signal = np.sin(2 * np.pi * (base_freq * 1.2) * t)
            network_noise = 0.3
        elif i < 12:  # Executive
            network_signal = np.sin(2 * np.pi * (base_freq * 0.8) * t)
            network_noise = 0.4
        else:  # Default
            network_signal = np.sin(2 * np.pi * (base_freq * 0.6) * t)
            network_noise = 0.3
        
        # Add region-specific variations
        region_variation = 0.2 * np.sin(2 * np.pi * (base_freq * 2) * t + i * np.pi/8)
        noise = network_noise * np.random.randn(n_timepoints)
        
        signal = network_signal + region_variation + noise
        data.append(signal)
    
    data = np.array(data)
    
    # Add within-network connectivity
    # Visual network internal connections
    data[1, 1:] += 0.3 * data[0, :-1]  # V1_L -> V1_R
    data[2, 2:] += 0.25 * data[0, :-2]  # V1_L -> V2_L
    
    # Motor network internal connections
    data[5, 1:] += 0.4 * data[4, :-1]  # M1_L -> M1_R
    data[6, 2:] += 0.3 * data[4, :-2]  # M1_L -> PMC_L
    
    # Executive network internal connections
    data[9, 2:] += 0.35 * data[8, :-2]  # DLPFC_L -> DLPFC_R
    
    # Default network internal connections
    data[13, 3:] += 0.4 * data[12, :-3]  # PCC -> mPFC
    
    # Add between-network connections (weaker)
    data[8, 4:] += 0.2 * data[0, :-4]   # V1_L -> DLPFC_L (visual-executive)
    data[12, 5:] += 0.15 * data[8, :-5]  # DLPFC_L -> PCC (executive-default)
    
    # Standardize data
    scaler = StandardScaler()
    data = scaler.fit_transform(data.T).T
    
    # Test ensemble analysis (baseline)
    print("\\n1. Testing Ensemble Analysis (Baseline)")
    print("-" * 50)
    
    ensemble_smte = HierarchicalSMTE(
        use_hierarchical_analysis=False,  # Disable hierarchical
        use_ensemble_testing=True,
        surrogate_methods=['aaft'],
        n_surrogates_per_method=20,
        use_multiscale_analysis=True,
        scales_to_analyze=['fast', 'intermediate'],
        adaptive_mode='heuristic',
        use_network_correction=True,
        use_physiological_constraints=True,
        known_networks=known_networks,
        TR=TR,
        n_permutations=100,
        random_state=42
    )
    
    ensemble_results = ensemble_smte.compute_hierarchical_connectivity(
        data, roi_labels
    )
    
    print(f"Ensemble: {ensemble_results['n_final_significant']} significant connections")
    
    # Test hierarchical analysis
    print("\\n2. Testing Hierarchical Analysis")
    print("-" * 40)
    
    hierarchical_smte = HierarchicalSMTE(
        use_hierarchical_analysis=True,   # Enable hierarchical
        hierarchy_methods=['agglomerative', 'spectral'],
        hierarchy_levels=[2, 4, 6],
        distance_metrics=['correlation', 'euclidean'],
        use_ensemble_testing=True,
        surrogate_methods=['aaft'],
        n_surrogates_per_method=15,  # Reduced for validation speed
        use_multiscale_analysis=True,
        scales_to_analyze=['fast'],  # Single scale for speed
        adaptive_mode='heuristic',
        use_network_correction=True,
        use_physiological_constraints=True,
        known_networks=known_networks,
        TR=TR,
        n_permutations=100,
        random_state=42
    )
    
    hierarchical_results = hierarchical_smte.compute_hierarchical_connectivity(
        data, roi_labels
    )
    
    # Analyze network organization
    network_organization = hierarchical_smte.analyze_hierarchical_network_organization(
        hierarchical_results, roi_labels
    )
    
    print(f"\\nHierarchy analysis:")
    print(f"Best method: {hierarchical_results['best_hierarchy']['method']}")
    print(f"Best distance: {hierarchical_results['best_hierarchy']['distance_metric']}")
    
    optimal_level = network_organization['cross_level_analysis']['optimal_hierarchy_level']
    print(f"Optimal level: {optimal_level['n_clusters']} clusters")
    print(f"Optimal silhouette: {optimal_level['silhouette_score']:.3f}")
    
    # Create visualizations
    hierarchical_smte.create_hierarchical_visualizations(
        hierarchical_results, roi_labels
    )
    
    return hierarchical_results, network_organization


if __name__ == "__main__":
    results, organization = test_hierarchical_smte()