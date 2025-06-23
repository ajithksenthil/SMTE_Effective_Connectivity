import numpy as np
import nibabel as nib
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.cluster import KMeans
from numba import njit, prange
import warnings
from typing import Tuple, List, Optional, Dict, Union, Any
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging


class VoxelSMTEConnectivity:
    """
    Complete fMRI voxel effective connectivity analysis using Symbolic Matrix Transfer Entropy.
    
    This implementation provides:
    - Multiple symbolization methods (uniform, quantile, ordinal patterns, vector quantization)
    - Efficient voxel-wise SMTE computation with parallelization
    - Statistical significance testing with permutation and FDR correction
    - Network analysis and visualization tools
    - Memory-efficient processing for large datasets
    """
    
    def __init__(self, 
                 n_symbols: int = 5,
                 symbolizer: str = 'uniform',
                 max_lag: int = 5,
                 alpha: float = 0.05,
                 n_permutations: int = 1000,
                 n_jobs: int = -1,
                 memory_efficient: bool = True):
        """
        Initialize VoxelSMTEConnectivity analyzer.
        
        Parameters:
        -----------
        n_symbols : int
            Number of symbols for discretization
        symbolizer : str
            Symbolization method ('uniform', 'quantile', 'ordinal', 'vq')
        max_lag : int
            Maximum time lag to consider for transfer entropy
        alpha : float
            Statistical significance threshold
        n_permutations : int
            Number of permutations for statistical testing
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        memory_efficient : bool
            Use memory-efficient processing for large datasets
        """
        self.n_symbols = n_symbols
        self.symbolizer = symbolizer
        self.max_lag = max_lag
        self.alpha = alpha
        self.n_permutations = n_permutations
        self.n_jobs = n_jobs if n_jobs != -1 else os.cpu_count()
        self.memory_efficient = memory_efficient
        
        # Internal storage
        self.fmri_data = None
        self.mask = None
        self.voxel_coords = None
        self.symbolic_data = None
        self.smte_matrix = None
        self.p_values = None
        self.connectivity_graph = None
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_fmri_data(self, fmri_path: str, mask_path: Optional[str] = None) -> None:
        """
        Load fMRI data and optional brain mask.
        
        Parameters:
        -----------
        fmri_path : str
            Path to 4D fMRI NIfTI file
        mask_path : str, optional
            Path to 3D brain mask NIfTI file
        """
        self.logger.info(f"Loading fMRI data from {fmri_path}")
        
        # Load fMRI data
        fmri_img = nib.load(fmri_path)
        self.fmri_data = fmri_img.get_fdata()
        
        # Load or create mask
        if mask_path:
            self.logger.info(f"Loading mask from {mask_path}")
            mask_img = nib.load(mask_path)
            self.mask = mask_img.get_fdata().astype(bool)
        else:
            self.logger.info("Creating default mask (non-zero voxels)")
            self.mask = np.mean(self.fmri_data, axis=3) > 0
            
        # Get voxel coordinates
        self.voxel_coords = np.where(self.mask)
        n_voxels = len(self.voxel_coords[0])
        
        self.logger.info(f"Loaded {n_voxels} voxels with {self.fmri_data.shape[3]} time points")
        
    def extract_voxel_timeseries(self) -> np.ndarray:
        """
        Extract time series for all masked voxels.
        
        Returns:
        --------
        np.ndarray
            Voxel time series matrix (n_voxels, n_timepoints)
        """
        if self.fmri_data is None or self.mask is None:
            raise ValueError("fMRI data and mask must be loaded first")
            
        # Extract masked voxel time series
        voxel_ts = self.fmri_data[self.mask]
        
        # Standardize time series
        scaler = StandardScaler()
        voxel_ts = scaler.fit_transform(voxel_ts.T).T
        
        return voxel_ts
    
    def symbolize_timeseries(self, timeseries: np.ndarray) -> np.ndarray:
        """
        Convert continuous time series to symbolic sequences.
        
        Parameters:
        -----------
        timeseries : np.ndarray
            Time series data (n_voxels, n_timepoints)
            
        Returns:
        --------
        np.ndarray
            Symbolic sequences (n_voxels, n_timepoints)
        """
        self.logger.info(f"Symbolizing time series using {self.symbolizer} method")
        
        n_voxels, n_timepoints = timeseries.shape
        symbolic_data = np.zeros_like(timeseries, dtype=int)
        
        if self.symbolizer == 'uniform':
            discretizer = KBinsDiscretizer(
                n_bins=self.n_symbols, 
                encode='ordinal', 
                strategy='uniform'
            )
            for i in range(n_voxels):
                symbolic_data[i] = discretizer.fit_transform(
                    timeseries[i].reshape(-1, 1)
                ).astype(int).flatten()
                
        elif self.symbolizer == 'quantile':
            discretizer = KBinsDiscretizer(
                n_bins=self.n_symbols, 
                encode='ordinal', 
                strategy='quantile'
            )
            for i in range(n_voxels):
                symbolic_data[i] = discretizer.fit_transform(
                    timeseries[i].reshape(-1, 1)
                ).astype(int).flatten()
                
        elif self.symbolizer == 'ordinal':
            for i in range(n_voxels):
                symbolic_data[i] = self._ordinal_patterns(timeseries[i])
                
        elif self.symbolizer == 'vq':
            for i in range(n_voxels):
                kmeans = KMeans(n_clusters=self.n_symbols, random_state=42, n_init=10)
                symbolic_data[i] = kmeans.fit_predict(timeseries[i].reshape(-1, 1))
                
        else:
            raise ValueError(f"Unknown symbolizer: {self.symbolizer}")
            
        return symbolic_data
    
    def _ordinal_patterns(self, ts: np.ndarray, order: int = 3) -> np.ndarray:
        """
        Convert time series to ordinal patterns.
        
        Parameters:
        -----------
        ts : np.ndarray
            Time series
        order : int
            Order of ordinal patterns
            
        Returns:
        --------
        np.ndarray
            Ordinal pattern sequence
        """
        if len(ts) < order:
            raise ValueError("Time series too short for ordinal patterns")
            
        n_patterns = len(ts) - order + 1
        patterns = np.zeros(n_patterns, dtype=int)
        
        for i in range(n_patterns):
            window = ts[i:i+order]
            # Get ordinal pattern (rank order)
            patterns[i] = self._pattern_to_index(window.argsort())
            
        return patterns
    
    @staticmethod
    @njit
    def _pattern_to_index(pattern: np.ndarray) -> int:
        """Convert ordinal pattern to unique index."""
        index = 0
        n = len(pattern)
        for i in range(n):
            smaller_count = 0
            for j in range(i+1, n):
                if pattern[j] < pattern[i]:
                    smaller_count += 1
            index += smaller_count * np.math.factorial(n - i - 1)
        return index
    
    @staticmethod
    @njit
    def _compute_gram_matrix(symbols: np.ndarray) -> np.ndarray:
        """
        Compute Gram matrix for symbolic sequence.
        
        Parameters:
        -----------
        symbols : np.ndarray
            Symbolic sequence
            
        Returns:
        --------
        np.ndarray
            Normalized Gram matrix
        """
        n = len(symbols)
        gram = np.zeros((n, n), dtype=np.float64)
        
        for i in range(n):
            for j in range(n):
                gram[i, j] = 1.0 if symbols[i] == symbols[j] else 0.0
                
        # Normalize with regularization
        gram += 1e-10
        return gram / np.sum(gram)
    
    @staticmethod
    @njit
    def _matrix_entropy(gram_matrix: np.ndarray) -> float:
        """
        Compute von Neumann entropy of Gram matrix.
        
        Parameters:
        -----------
        gram_matrix : np.ndarray
            Gram matrix
            
        Returns:
        --------
        float
            Matrix entropy
        """
        eigenvals = np.linalg.eigvals(gram_matrix)
        eigenvals = np.maximum(eigenvals, 1e-12)  # Avoid log(0)
        entropy = -np.sum(eigenvals * np.log(eigenvals))
        return max(entropy, 0.0)  # Ensure non-negative
    
    def _compute_smte_pair(self, symbols_x: np.ndarray, symbols_y: np.ndarray) -> Tuple[float, int]:
        """
        Compute SMTE between two symbolic sequences.
        
        Parameters:
        -----------
        symbols_x : np.ndarray
            Target symbolic sequence
        symbols_y : np.ndarray
            Source symbolic sequence
            
        Returns:
        --------
        Tuple[float, int]
            (SMTE value, optimal lag)
        """
        best_smte = 0.0
        best_lag = 0
        
        for lag in range(self.max_lag + 1):
            if lag == 0:
                x_present = symbols_x[1:]
                x_past = symbols_x[:-1]
                y_past = symbols_y[:-1]
            else:
                if lag >= len(symbols_x):
                    continue
                x_present = symbols_x[lag:]
                x_past = symbols_x[:-lag]
                y_past = symbols_y[:-lag]
            
            # Compute joint distributions using histograms for efficiency
            n_bins = min(self.n_symbols, len(np.unique(x_present)))
            
            # H(X_t | X_{t-1})
            joint_x = np.column_stack([x_present, x_past])
            hist_x_joint, _ = np.histogramdd(joint_x.T, bins=n_bins)
            hist_x_joint = hist_x_joint + 1e-10
            prob_x_joint = hist_x_joint / np.sum(hist_x_joint)
            H_x_given_xpast = -np.sum(prob_x_joint * np.log(prob_x_joint))
            
            hist_x_past, _ = np.histogram(x_past, bins=n_bins)
            hist_x_past = hist_x_past + 1e-10
            prob_x_past = hist_x_past / np.sum(hist_x_past)
            H_x_past = -np.sum(prob_x_past * np.log(prob_x_past))
            
            H_x_cond_xpast = H_x_given_xpast - H_x_past
            
            # H(X_t | X_{t-1}, Y_{t-1})
            joint_xy = np.column_stack([x_present, x_past, y_past])
            hist_xy_joint, _ = np.histogramdd(joint_xy.T, bins=n_bins)
            hist_xy_joint = hist_xy_joint + 1e-10
            prob_xy_joint = hist_xy_joint / np.sum(hist_xy_joint)
            H_x_given_xy_past = -np.sum(prob_xy_joint * np.log(prob_xy_joint))
            
            joint_y_past = np.column_stack([x_past, y_past])
            hist_y_joint, _ = np.histogramdd(joint_y_past.T, bins=n_bins)
            hist_y_joint = hist_y_joint + 1e-10
            prob_y_joint = hist_y_joint / np.sum(hist_y_joint)
            H_xy_past = -np.sum(prob_y_joint * np.log(prob_y_joint))
            
            H_x_cond_xy_past = H_x_given_xy_past - H_xy_past
            
            # Transfer entropy
            te = H_x_cond_xpast - H_x_cond_xy_past
            te = max(te, 0.0)  # Ensure non-negative
            
            if te > best_smte:
                best_smte = te
                best_lag = lag
                
        return best_smte, best_lag
    
    def compute_voxel_connectivity_matrix(self, 
                                        voxel_indices: Optional[List[int]] = None,
                                        chunk_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute SMTE connectivity matrix between voxels.
        
        Parameters:
        -----------
        voxel_indices : List[int], optional
            Indices of voxels to analyze (None for all)
        chunk_size : int
            Number of voxel pairs to process in each chunk
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (SMTE matrix, optimal lag matrix)
        """
        if self.symbolic_data is None:
            raise ValueError("Symbolic data not computed. Run symbolize_timeseries first.")
            
        n_voxels = self.symbolic_data.shape[0]
        
        if voxel_indices is None:
            voxel_indices = list(range(n_voxels))
            
        n_selected = len(voxel_indices)
        smte_matrix = np.zeros((n_selected, n_selected))
        lag_matrix = np.zeros((n_selected, n_selected), dtype=int)
        
        self.logger.info(f"Computing SMTE matrix for {n_selected} voxels")
        
        if self.memory_efficient and n_selected > 1000:
            # Process in chunks for large datasets
            self._compute_matrix_chunked(voxel_indices, smte_matrix, lag_matrix, chunk_size)
        else:
            # Process all pairs
            self._compute_matrix_full(voxel_indices, smte_matrix, lag_matrix)
            
        return smte_matrix, lag_matrix
    
    def _compute_matrix_chunked(self, voxel_indices: List[int], 
                               smte_matrix: np.ndarray, 
                               lag_matrix: np.ndarray,
                               chunk_size: int) -> None:
        """Compute SMTE matrix in chunks for memory efficiency."""
        n_voxels = len(voxel_indices)
        n_chunks = (n_voxels * n_voxels - 1) // chunk_size + 1
        
        self.logger.info(f"Processing {n_chunks} chunks of size {chunk_size}")
        
        pairs = [(i, j) for i in range(n_voxels) for j in range(n_voxels) if i != j]
        
        for chunk_idx in range(0, len(pairs), chunk_size):
            chunk_pairs = pairs[chunk_idx:chunk_idx + chunk_size]
            
            if self.n_jobs > 1:
                # Parallel processing
                with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                    futures = []
                    for i, j in chunk_pairs:
                        future = executor.submit(
                            self._compute_smte_pair,
                            self.symbolic_data[voxel_indices[i]],
                            self.symbolic_data[voxel_indices[j]]
                        )
                        futures.append((future, i, j))
                    
                    for future, i, j in futures:
                        smte_val, optimal_lag = future.result()
                        smte_matrix[i, j] = smte_val
                        lag_matrix[i, j] = optimal_lag
            else:
                # Sequential processing
                for i, j in chunk_pairs:
                    smte_val, optimal_lag = self._compute_smte_pair(
                        self.symbolic_data[voxel_indices[i]],
                        self.symbolic_data[voxel_indices[j]]
                    )
                    smte_matrix[i, j] = smte_val
                    lag_matrix[i, j] = optimal_lag
                    
            if (chunk_idx // chunk_size + 1) % 10 == 0:
                self.logger.info(f"Processed {chunk_idx // chunk_size + 1}/{n_chunks} chunks")
    
    def _compute_matrix_full(self, voxel_indices: List[int], 
                            smte_matrix: np.ndarray, 
                            lag_matrix: np.ndarray) -> None:
        """Compute full SMTE matrix."""
        n_voxels = len(voxel_indices)
        
        for i in range(n_voxels):
            for j in range(n_voxels):
                if i != j:
                    smte_val, optimal_lag = self._compute_smte_pair(
                        self.symbolic_data[voxel_indices[i]],
                        self.symbolic_data[voxel_indices[j]]
                    )
                    smte_matrix[i, j] = smte_val
                    lag_matrix[i, j] = optimal_lag
                    
            if (i + 1) % 50 == 0:
                self.logger.info(f"Processed {i + 1}/{n_voxels} voxels")
    
    def statistical_testing(self, smte_matrix: np.ndarray) -> np.ndarray:
        """
        Perform statistical significance testing using permutation tests.
        
        Parameters:
        -----------
        smte_matrix : np.ndarray
            SMTE connectivity matrix
            
        Returns:
        --------
        np.ndarray
            P-values matrix
        """
        self.logger.info("Performing statistical significance testing")
        
        n_voxels = smte_matrix.shape[0]
        p_values = np.ones((n_voxels, n_voxels))
        
        # Generate null distribution for each voxel pair
        for i in range(n_voxels):
            for j in range(n_voxels):
                if i != j and smte_matrix[i, j] > 0:
                    # Permutation test
                    null_smte = []
                    symbols_j = self.symbolic_data[j].copy()
                    
                    for perm in range(self.n_permutations):
                        # Circular shift to destroy temporal structure
                        shift = np.random.randint(1, len(symbols_j))
                        symbols_j_perm = np.roll(symbols_j, shift)
                        
                        null_val, _ = self._compute_smte_pair(
                            self.symbolic_data[i], symbols_j_perm
                        )
                        null_smte.append(null_val)
                    
                    # Compute p-value
                    null_smte = np.array(null_smte)
                    p_values[i, j] = (np.sum(null_smte >= smte_matrix[i, j]) + 1) / (self.n_permutations + 1)
        
        return p_values
    
    def fdr_correction(self, p_values: np.ndarray) -> np.ndarray:
        """
        Apply Benjamini-Hochberg FDR correction.
        
        Parameters:
        -----------
        p_values : np.ndarray
            Raw p-values matrix
            
        Returns:
        --------
        np.ndarray
            Boolean mask of significant connections
        """
        # Flatten p-values (excluding diagonal)
        mask = ~np.eye(p_values.shape[0], dtype=bool)
        p_flat = p_values[mask]
        
        # Sort p-values
        sorted_indices = np.argsort(p_flat)
        sorted_p = p_flat[sorted_indices]
        
        # Apply BH correction
        m = len(sorted_p)
        bh_threshold = self.alpha * np.arange(1, m + 1) / m
        
        # Find largest k such that P(k) <= (k/m) * alpha
        significant_indices = sorted_indices[sorted_p <= bh_threshold]
        
        # Create significance mask
        significance_mask = np.zeros_like(p_values, dtype=bool)
        if len(significant_indices) > 0:
            flat_mask = np.zeros(len(p_flat), dtype=bool)
            flat_mask[significant_indices] = True
            significance_mask[mask] = flat_mask
            
        return significance_mask
    
    def build_connectivity_graph(self, 
                                smte_matrix: np.ndarray, 
                                significance_mask: np.ndarray,
                                voxel_coords: Optional[np.ndarray] = None) -> nx.DiGraph:
        """
        Build directed connectivity graph from significant SMTE connections.
        
        Parameters:
        -----------
        smte_matrix : np.ndarray
            SMTE connectivity matrix
        significance_mask : np.ndarray
            Boolean mask of significant connections
        voxel_coords : np.ndarray, optional
            3D coordinates of voxels
            
        Returns:
        --------
        nx.DiGraph
            Directed connectivity graph
        """
        G = nx.DiGraph()
        
        # Add nodes
        n_voxels = smte_matrix.shape[0]
        for i in range(n_voxels):
            node_attrs = {'voxel_id': i}
            if voxel_coords is not None:
                node_attrs.update({
                    'x': voxel_coords[0][i],
                    'y': voxel_coords[1][i], 
                    'z': voxel_coords[2][i]
                })
            G.add_node(i, **node_attrs)
        
        # Add significant edges
        for i in range(n_voxels):
            for j in range(n_voxels):
                if significance_mask[i, j]:
                    G.add_edge(j, i, weight=smte_matrix[i, j])  # j -> i (source -> target)
        
        self.logger.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G
    
    def analyze_network_properties(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Compute network properties of the connectivity graph.
        
        Parameters:
        -----------
        graph : nx.DiGraph
            Connectivity graph
            
        Returns:
        --------
        Dict[str, Any]
            Network properties
        """
        properties = {}
        
        # Basic properties
        properties['n_nodes'] = graph.number_of_nodes()
        properties['n_edges'] = graph.number_of_edges()
        properties['density'] = nx.density(graph)
        
        # Centrality measures
        properties['in_degree_centrality'] = nx.in_degree_centrality(graph)
        properties['out_degree_centrality'] = nx.out_degree_centrality(graph)
        properties['betweenness_centrality'] = nx.betweenness_centrality(graph)
        properties['closeness_centrality'] = nx.closeness_centrality(graph.to_undirected())
        
        # Hub analysis
        in_degrees = dict(graph.in_degree())
        out_degrees = dict(graph.out_degree())
        properties['hub_nodes'] = [node for node, degree in in_degrees.items() 
                                  if degree > np.mean(list(in_degrees.values())) + 2*np.std(list(in_degrees.values()))]
        
        # Connectivity strength
        edge_weights = [data['weight'] for _, _, data in graph.edges(data=True)]
        properties['mean_connection_strength'] = np.mean(edge_weights)
        properties['std_connection_strength'] = np.std(edge_weights)
        
        return properties
    
    def visualize_connectivity_matrix(self, smte_matrix: np.ndarray, 
                                    significance_mask: Optional[np.ndarray] = None,
                                    title: str = "SMTE Connectivity Matrix") -> None:
        """
        Visualize the SMTE connectivity matrix.
        
        Parameters:
        -----------
        smte_matrix : np.ndarray
            SMTE connectivity matrix
        significance_mask : np.ndarray, optional
            Significance mask to overlay
        title : str
            Plot title
        """
        plt.figure(figsize=(10, 8))
        
        # Plot matrix
        if significance_mask is not None:
            # Mask non-significant connections
            masked_matrix = smte_matrix.copy()
            masked_matrix[~significance_mask] = 0
            im = plt.imshow(masked_matrix, cmap='viridis', aspect='auto')
        else:
            im = plt.imshow(smte_matrix, cmap='viridis', aspect='auto')
        
        plt.colorbar(im, label='SMTE Value')
        plt.title(title)
        plt.xlabel('Source Voxel')
        plt.ylabel('Target Voxel')
        
        # Add significance overlay if provided
        if significance_mask is not None:
            y, x = np.where(significance_mask)
            plt.scatter(x, y, c='red', s=1, alpha=0.5, label='Significant')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, output_dir: str, 
                    smte_matrix: np.ndarray,
                    p_values: np.ndarray,
                    significance_mask: np.ndarray,
                    connectivity_graph: nx.DiGraph,
                    network_properties: Dict[str, Any]) -> None:
        """
        Save analysis results to files.
        
        Parameters:
        -----------
        output_dir : str
            Output directory
        smte_matrix : np.ndarray
            SMTE connectivity matrix
        p_values : np.ndarray
            P-values matrix
        significance_mask : np.ndarray
            Significance mask
        connectivity_graph : nx.DiGraph
            Connectivity graph
        network_properties : Dict[str, Any]
            Network properties
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save matrices
        np.save(os.path.join(output_dir, 'smte_matrix.npy'), smte_matrix)
        np.save(os.path.join(output_dir, 'p_values.npy'), p_values)
        np.save(os.path.join(output_dir, 'significance_mask.npy'), significance_mask)
        
        # Save graph
        nx.write_graphml(connectivity_graph, os.path.join(output_dir, 'connectivity_graph.graphml'))
        
        # Save network properties
        import json
        # Convert numpy arrays to lists for JSON serialization
        json_properties = {}
        for key, value in network_properties.items():
            if isinstance(value, dict):
                json_properties[key] = {str(k): float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                       for k, v in value.items()}
            elif isinstance(value, (np.integer, np.floating)):
                json_properties[key] = float(value)
            else:
                json_properties[key] = value
                
        with open(os.path.join(output_dir, 'network_properties.json'), 'w') as f:
            json.dump(json_properties, f, indent=2)
        
        self.logger.info(f"Results saved to {output_dir}")
    
    def run_complete_analysis(self, 
                            fmri_path: str,
                            output_dir: str,
                            mask_path: Optional[str] = None,
                            voxel_indices: Optional[List[int]] = None,
                            visualize: bool = True) -> Dict[str, Any]:
        """
        Run complete voxel-wise SMTE connectivity analysis.
        
        Parameters:
        -----------
        fmri_path : str
            Path to fMRI data
        output_dir : str
            Output directory
        mask_path : str, optional
            Path to brain mask
        voxel_indices : List[int], optional
            Specific voxels to analyze
        visualize : bool
            Whether to generate visualizations
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results
        """
        self.logger.info("Starting complete SMTE connectivity analysis")
        
        # 1. Load data
        self.load_fmri_data(fmri_path, mask_path)
        
        # 2. Extract voxel time series
        voxel_timeseries = self.extract_voxel_timeseries()
        
        # 3. Symbolize time series
        self.symbolic_data = self.symbolize_timeseries(voxel_timeseries)
        
        # 4. Compute connectivity matrix
        smte_matrix, lag_matrix = self.compute_voxel_connectivity_matrix(voxel_indices)
        
        # 5. Statistical testing
        p_values = self.statistical_testing(smte_matrix)
        
        # 6. FDR correction
        significance_mask = self.fdr_correction(p_values)
        
        # 7. Build connectivity graph
        connectivity_graph = self.build_connectivity_graph(
            smte_matrix, significance_mask, self.voxel_coords
        )
        
        # 8. Network analysis
        network_properties = self.analyze_network_properties(connectivity_graph)
        
        # 9. Save results
        self.save_results(
            output_dir, smte_matrix, p_values, significance_mask,
            connectivity_graph, network_properties
        )
        
        # 10. Visualization
        if visualize:
            self.visualize_connectivity_matrix(smte_matrix, significance_mask)
        
        # Store results
        self.smte_matrix = smte_matrix
        self.p_values = p_values
        self.connectivity_graph = connectivity_graph
        
        results = {
            'smte_matrix': smte_matrix,
            'p_values': p_values,
            'significance_mask': significance_mask,
            'connectivity_graph': connectivity_graph,
            'network_properties': network_properties,
            'n_significant_connections': np.sum(significance_mask)
        }
        
        self.logger.info(f"Analysis complete. Found {np.sum(significance_mask)} significant connections.")
        
        return results


def example_usage():
    """Example usage of VoxelSMTEConnectivity."""
    
    # Initialize analyzer
    analyzer = VoxelSMTEConnectivity(
        n_symbols=5,
        symbolizer='uniform',
        max_lag=3,
        alpha=0.05,
        n_permutations=500,
        n_jobs=4
    )
    
    # Run analysis
    results = analyzer.run_complete_analysis(
        fmri_path='/path/to/fmri_data.nii.gz',
        output_dir='/path/to/output/',
        mask_path='/path/to/mask.nii.gz',
        visualize=True
    )
    
    print(f"Found {results['n_significant_connections']} significant connections")
    print(f"Network density: {results['network_properties']['density']:.4f}")
    
    return results


if __name__ == "__main__":
    # Run example (uncomment and modify paths)
    # results = example_usage()
    pass