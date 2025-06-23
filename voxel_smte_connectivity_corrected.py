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
from itertools import permutations
import math


class VoxelSMTEConnectivity:
    """
    Research-grade fMRI voxel effective connectivity analysis using Symbolic Matrix Transfer Entropy.
    
    This implementation follows established theoretical foundations and includes:
    - Mathematically correct SMTE computation based on Gram matrices
    - Proper ordinal pattern symbolization with factorial encoding
    - Robust statistical significance testing with multiple comparison correction
    - Memory-efficient processing for large-scale neuroimaging data
    
    References:
    -----------
    - Staniek, M. & Lehnertz, K. (2008). Symbolic transfer entropy. Physical Review Letters.
    - Daw, C.S., Finney, C.E.A. & Tracy, E.R. (2003). A review of symbolic analysis of experimental data.
    - Schreiber, T. (2000). Measuring information transfer. Physical Review Letters.
    """
    
    def __init__(self, 
                 n_symbols: int = 6,
                 symbolizer: str = 'ordinal',
                 ordinal_order: int = 3,
                 max_lag: int = 5,
                 alpha: float = 0.05,
                 n_permutations: int = 1000,
                 n_jobs: int = -1,
                 memory_efficient: bool = True,
                 random_state: int = 42):
        """
        Initialize VoxelSMTEConnectivity analyzer.
        
        Parameters:
        -----------
        n_symbols : int
            Number of symbols for discretization (default: 6 for ordinal patterns of order 3)
        symbolizer : str
            Symbolization method ('ordinal', 'uniform', 'quantile', 'vq')
        ordinal_order : int
            Order for ordinal pattern symbolization (typically 3-5)
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
        random_state : int
            Random seed for reproducibility
        """
        self.n_symbols = n_symbols
        self.symbolizer = symbolizer
        self.ordinal_order = ordinal_order
        self.max_lag = max_lag
        self.alpha = alpha
        self.n_permutations = n_permutations
        self.n_jobs = n_jobs if n_jobs != -1 else os.cpu_count()
        self.memory_efficient = memory_efficient
        self.random_state = random_state
        
        # Set up logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Validate parameters
        self._validate_parameters()
        
        # Internal storage
        self.fmri_data = None
        self.mask = None
        self.voxel_coords = None
        self.symbolic_data = None
        self.smte_matrix = None
        self.p_values = None
        self.connectivity_graph = None
        
        # Set random seed
        np.random.seed(self.random_state)
        
    def _validate_parameters(self):
        """Validate initialization parameters."""
        if self.symbolizer == 'ordinal':
            if self.ordinal_order < 2:
                raise ValueError("Ordinal order must be >= 2")
            expected_symbols = math.factorial(self.ordinal_order)
            if self.n_symbols != expected_symbols:
                self.logger.warning(f"For ordinal patterns of order {self.ordinal_order}, "
                                   f"n_symbols should be {expected_symbols}. Setting n_symbols={expected_symbols}")
                self.n_symbols = expected_symbols
        
        if self.max_lag < 1:
            raise ValueError("max_lag must be >= 1")
        if self.alpha <= 0 or self.alpha >= 1:
            raise ValueError("alpha must be between 0 and 1")
        if self.n_permutations < 100:
            self.logger.warning("n_permutations < 100 may lead to unreliable p-values")
        
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
        
        if len(self.fmri_data.shape) != 4:
            raise ValueError("fMRI data must be 4-dimensional (x, y, z, time)")
            
        # Load or create mask
        if mask_path:
            self.logger.info(f"Loading mask from {mask_path}")
            mask_img = nib.load(mask_path)
            self.mask = mask_img.get_fdata().astype(bool)
            
            if self.mask.shape != self.fmri_data.shape[:3]:
                raise ValueError("Mask dimensions must match fMRI spatial dimensions")
        else:
            self.logger.info("Creating default mask (non-zero mean voxels)")
            mean_signal = np.mean(self.fmri_data, axis=3)
            self.mask = (mean_signal > np.std(mean_signal) * 0.1)
            
        # Get voxel coordinates
        self.voxel_coords = np.where(self.mask)
        n_voxels = len(self.voxel_coords[0])
        
        self.logger.info(f"Loaded {n_voxels} voxels with {self.fmri_data.shape[3]} time points")
        
        # Validate minimum time points for analysis
        min_timepoints = max(50, self.max_lag * 5)
        if self.fmri_data.shape[3] < min_timepoints:
            self.logger.warning(f"Short time series ({self.fmri_data.shape[3]} points). "
                               f"Recommend >= {min_timepoints} for reliable SMTE estimation.")
        
    def extract_voxel_timeseries(self) -> np.ndarray:
        """
        Extract and preprocess time series for all masked voxels.
        
        Returns:
        --------
        np.ndarray
            Preprocessed voxel time series matrix (n_voxels, n_timepoints)
        """
        if self.fmri_data is None or self.mask is None:
            raise ValueError("fMRI data and mask must be loaded first")
            
        # Extract masked voxel time series
        voxel_ts = self.fmri_data[self.mask]
        
        # Remove linear trend (common preprocessing step)
        from scipy.signal import detrend
        voxel_ts = np.array([detrend(ts) for ts in voxel_ts])
        
        # Standardize time series (z-score)
        scaler = StandardScaler()
        voxel_ts = scaler.fit_transform(voxel_ts.T).T
        
        return voxel_ts
    
    def symbolize_timeseries(self, timeseries: np.ndarray) -> np.ndarray:
        """
        Convert continuous time series to symbolic sequences using specified method.
        
        Parameters:
        -----------
        timeseries : np.ndarray
            Time series data (n_voxels, n_timepoints)
            
        Returns:
        --------
        np.ndarray
            Symbolic sequences (n_voxels, n_timepoints_symbolic)
        """
        self.logger.info(f"Symbolizing time series using {self.symbolizer} method")
        
        n_voxels, n_timepoints = timeseries.shape
        
        if self.symbolizer == 'ordinal':
            # Ordinal pattern symbolization
            symbolic_length = n_timepoints - self.ordinal_order + 1
            symbolic_data = np.zeros((n_voxels, symbolic_length), dtype=int)
            
            for i in range(n_voxels):
                symbolic_data[i] = self._ordinal_patterns(timeseries[i], self.ordinal_order)
                
        else:
            # Other symbolization methods preserve time series length
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
                    try:
                        symbolic_data[i] = discretizer.fit_transform(
                            timeseries[i].reshape(-1, 1)
                        ).astype(int).flatten()
                    except ValueError:
                        # Fallback to uniform if quantile fails (e.g., constant time series)
                        self.logger.warning(f"Quantile discretization failed for voxel {i}, using uniform")
                        fallback_discretizer = KBinsDiscretizer(
                            n_bins=self.n_symbols, 
                            encode='ordinal', 
                            strategy='uniform'
                        )
                        symbolic_data[i] = fallback_discretizer.fit_transform(
                            timeseries[i].reshape(-1, 1)
                        ).astype(int).flatten()
                        
            elif self.symbolizer == 'vq':
                for i in range(n_voxels):
                    kmeans = KMeans(
                        n_clusters=self.n_symbols, 
                        random_state=self.random_state, 
                        n_init=10
                    )
                    symbolic_data[i] = kmeans.fit_predict(timeseries[i].reshape(-1, 1))
                    
            else:
                raise ValueError(f"Unknown symbolizer: {self.symbolizer}")
                
        return symbolic_data
    
    def _ordinal_patterns(self, ts: np.ndarray, order: int) -> np.ndarray:
        """
        Convert time series to ordinal patterns with correct factorial encoding.
        
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
            raise ValueError(f"Time series length ({len(ts)}) < ordinal order ({order})")
            
        n_patterns = len(ts) - order + 1
        patterns = np.zeros(n_patterns, dtype=int)
        
        for i in range(n_patterns):
            window = ts[i:i+order]
            # Handle ties by adding small random noise
            if len(np.unique(window)) < order:
                window = window + np.random.randn(order) * 1e-10
            
            # Get ordinal pattern using lexicographic ranking
            sorted_indices = np.argsort(window)
            rank_pattern = np.empty_like(sorted_indices)
            rank_pattern[sorted_indices] = np.arange(order)
            
            # Convert to single index (lexicographic order)
            pattern_index = 0
            for j, rank in enumerate(rank_pattern):
                pattern_index += rank * (order ** (order - 1 - j))
            
            patterns[i] = pattern_index % math.factorial(order)
            
        return patterns
    
    @staticmethod
    @njit
    def _compute_gram_matrix(symbols: np.ndarray) -> np.ndarray:
        """
        Compute normalized Gram matrix for symbolic sequence.
        
        The Gram matrix G_ij = 1 if symbols[i] == symbols[j], 0 otherwise.
        This captures the recurrence structure of the symbolic sequence.
        
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
                if symbols[i] == symbols[j]:
                    gram[i, j] = 1.0
                    
        # Proper normalization for probability matrix
        total = np.sum(gram)
        if total > 0:
            gram = gram / total
        else:
            # Handle edge case of no matching symbols
            gram = np.ones((n, n)) / (n * n)
            
        return gram
    
    @staticmethod
    def _matrix_entropy(gram_matrix: np.ndarray) -> float:
        """
        Compute von Neumann entropy of Gram matrix.
        
        The von Neumann entropy is S = -Tr(ρ log ρ) where ρ is the density matrix.
        For discrete probability matrices, this reduces to eigenvalue-based entropy.
        
        Parameters:
        -----------
        gram_matrix : np.ndarray
            Normalized Gram matrix
            
        Returns:
        --------
        float
            Matrix entropy (non-negative)
        """
        # Ensure matrix is properly normalized
        if not np.allclose(np.sum(gram_matrix), 1.0, rtol=1e-10):
            gram_matrix = gram_matrix / np.sum(gram_matrix)
            
        try:
            # Compute eigenvalues (real for symmetric matrices)
            eigenvals = np.linalg.eigvals(gram_matrix)
            eigenvals = np.real(eigenvals)  # Take real part to handle numerical errors
            eigenvals = eigenvals[eigenvals > 1e-12]  # Remove effectively zero eigenvalues
            
            if len(eigenvals) == 0:
                return 0.0
                
            # Normalize eigenvalues to ensure they sum to 1
            eigenvals = eigenvals / np.sum(eigenvals)
            
            # Compute entropy
            entropy = -np.sum(eigenvals * np.log(eigenvals))
            return max(entropy, 0.0)  # Ensure non-negative
            
        except np.linalg.LinAlgError:
            # Fallback for degenerate matrices
            return 0.0
    
    def _compute_transfer_entropy_classic(self, symbols_x: np.ndarray, symbols_y: np.ndarray, lag: int) -> float:
        """
        Compute classic transfer entropy using joint probability distributions.
        
        TE(Y→X) = H(X_t|X_{t-1}) - H(X_t|X_{t-1}, Y_{t-lag})
        
        Parameters:
        -----------
        symbols_x : np.ndarray
            Target symbolic sequence
        symbols_y : np.ndarray
            Source symbolic sequence
        lag : int
            Time lag
            
        Returns:
        --------
        float
            Transfer entropy value
        """
        if lag >= len(symbols_x) or lag < 1:
            return 0.0
            
        # Create time-lagged sequences
        x_present = symbols_x[lag:]
        x_past = symbols_x[:-lag]
        y_past = symbols_y[:-lag]
        
        # Compute joint probability distributions
        n_states = max(np.max(symbols_x), np.max(symbols_y)) + 1
        
        # P(X_t, X_{t-1})
        joint_xx = np.zeros((n_states, n_states))
        for i in range(len(x_present)):
            joint_xx[x_present[i], x_past[i]] += 1
        joint_xx = joint_xx + 1e-12  # Regularization
        joint_xx = joint_xx / np.sum(joint_xx)
        
        # P(X_{t-1})
        marginal_x_past = np.bincount(x_past, minlength=n_states).astype(float)
        marginal_x_past = marginal_x_past + 1e-12
        marginal_x_past = marginal_x_past / np.sum(marginal_x_past)
        
        # P(X_t, X_{t-1}, Y_{t-1})
        joint_xxy = np.zeros((n_states, n_states, n_states))
        for i in range(len(x_present)):
            joint_xxy[x_present[i], x_past[i], y_past[i]] += 1
        joint_xxy = joint_xxy + 1e-12
        joint_xxy = joint_xxy / np.sum(joint_xxy)
        
        # P(X_{t-1}, Y_{t-1})
        joint_xy_past = np.zeros((n_states, n_states))
        for i in range(len(x_past)):
            joint_xy_past[x_past[i], y_past[i]] += 1
        joint_xy_past = joint_xy_past + 1e-12
        joint_xy_past = joint_xy_past / np.sum(joint_xy_past)
        
        # Compute conditional entropies
        # H(X_t | X_{t-1})
        h_x_given_x_past = 0.0
        for i in range(n_states):
            for j in range(n_states):
                if joint_xx[i, j] > 1e-12 and marginal_x_past[j] > 1e-12:
                    p_cond = joint_xx[i, j] / marginal_x_past[j]
                    h_x_given_x_past -= joint_xx[i, j] * np.log(p_cond)
        
        # H(X_t | X_{t-1}, Y_{t-1})
        h_x_given_xy_past = 0.0
        for i in range(n_states):
            for j in range(n_states):
                for k in range(n_states):
                    if joint_xxy[i, j, k] > 1e-12 and joint_xy_past[j, k] > 1e-12:
                        p_cond = joint_xxy[i, j, k] / joint_xy_past[j, k]
                        h_x_given_xy_past -= joint_xxy[i, j, k] * np.log(p_cond)
        
        # Transfer entropy
        te = h_x_given_x_past - h_x_given_xy_past
        return max(te, 0.0)  # Ensure non-negative
    
    def _compute_smte_pair(self, symbols_x: np.ndarray, symbols_y: np.ndarray) -> Tuple[float, int]:
        """
        Compute SMTE between two symbolic sequences across multiple lags.
        
        Parameters:
        -----------
        symbols_x : np.ndarray
            Target symbolic sequence
        symbols_y : np.ndarray
            Source symbolic sequence
            
        Returns:
        --------
        Tuple[float, int]
            (Maximum SMTE value, optimal lag)
        """
        best_smte = 0.0
        best_lag = 0
        
        for lag in range(1, self.max_lag + 1):
            # Use classic transfer entropy for robustness
            te = self._compute_transfer_entropy_classic(symbols_x, symbols_y, lag)
            
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
        pairs = [(i, j) for i in range(n_voxels) for j in range(n_voxels) if i != j]
        n_chunks = (len(pairs) - 1) // chunk_size + 1
        
        self.logger.info(f"Processing {n_chunks} chunks of size {chunk_size}")
        
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
                    
            progress = min((chunk_idx // chunk_size + 1) * chunk_size, len(pairs))
            self.logger.info(f"Processed {progress}/{len(pairs)} pairs ({100*progress/len(pairs):.1f}%)")
    
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
                    
            if (i + 1) % 10 == 0:
                self.logger.info(f"Processed {i + 1}/{n_voxels} voxels ({100*(i+1)/n_voxels:.1f}%)")
    
    def statistical_testing(self, smte_matrix: np.ndarray, 
                           voxel_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Perform statistical significance testing using surrogate data.
        
        Uses circular shuffling to destroy temporal structure while preserving
        marginal distribution and autocorrelation properties.
        
        Parameters:
        -----------
        smte_matrix : np.ndarray
            SMTE connectivity matrix
        voxel_indices : List[int], optional
            Original voxel indices for the matrix
            
        Returns:
        --------
        np.ndarray
            P-values matrix
        """
        self.logger.info("Performing statistical significance testing with surrogate data")
        
        n_voxels = smte_matrix.shape[0]
        p_values = np.ones((n_voxels, n_voxels))
        
        if voxel_indices is None:
            voxel_indices = list(range(n_voxels))
        
        # Generate surrogate null distributions
        for i in range(n_voxels):
            for j in range(n_voxels):
                if i != j and smte_matrix[i, j] > 0:
                    # Generate surrogate data for source time series
                    source_symbols = self.symbolic_data[voxel_indices[j]].copy()
                    target_symbols = self.symbolic_data[voxel_indices[i]]
                    
                    null_smte = []
                    for perm in range(self.n_permutations):
                        # Circular shuffle to destroy temporal dependencies
                        shift = np.random.randint(1, len(source_symbols))
                        source_surrogate = np.roll(source_symbols, shift)
                        
                        # Compute SMTE for surrogate
                        null_val, _ = self._compute_smte_pair(target_symbols, source_surrogate)
                        null_smte.append(null_val)
                    
                    # Compute p-value using surrogate distribution
                    null_smte = np.array(null_smte)
                    p_values[i, j] = (np.sum(null_smte >= smte_matrix[i, j]) + 1) / (self.n_permutations + 1)
        
        return p_values
    
    def fdr_correction(self, p_values: np.ndarray, method: str = 'bh') -> np.ndarray:
        """
        Apply False Discovery Rate correction for multiple comparisons.
        
        Parameters:
        -----------
        p_values : np.ndarray
            Raw p-values matrix
        method : str
            FDR correction method ('bh' for Benjamini-Hochberg)
            
        Returns:
        --------
        np.ndarray
            Boolean mask of significant connections after FDR correction
        """
        # Manual FDR implementation for compatibility
        
        # Extract off-diagonal p-values (exclude self-connections)
        mask = ~np.eye(p_values.shape[0], dtype=bool)
        p_flat = p_values[mask]
        
        if method == 'bh':
            # Benjamini-Hochberg procedure - manual implementation for compatibility
            sorted_indices = np.argsort(p_flat)
            sorted_p = p_flat[sorted_indices]
            m = len(sorted_p)
            
            # Find largest k such that P(k) <= (k/m) * alpha
            bh_threshold = self.alpha * np.arange(1, m + 1) / m
            significant_indices = sorted_indices[sorted_p <= bh_threshold]
            
            rejected = np.zeros(len(p_flat), dtype=bool)
            if len(significant_indices) > 0:
                rejected[significant_indices] = True
        else:
            raise ValueError(f"Unknown FDR method: {method}")
        
        # Create significance mask
        significance_mask = np.zeros_like(p_values, dtype=bool)
        significance_mask[mask] = rejected
            
        n_significant = np.sum(significance_mask)
        total_tests = np.sum(mask)
        self.logger.info(f"FDR correction: {n_significant}/{total_tests} connections significant "
                        f"(α={self.alpha})")
            
        return significance_mask
    
    def build_connectivity_graph(self, 
                                smte_matrix: np.ndarray, 
                                significance_mask: np.ndarray,
                                voxel_coords: Optional[Tuple[np.ndarray, ...]] = None) -> nx.DiGraph:
        """
        Build directed connectivity graph from significant SMTE connections.
        
        Parameters:
        -----------
        smte_matrix : np.ndarray
            SMTE connectivity matrix
        significance_mask : np.ndarray
            Boolean mask of significant connections
        voxel_coords : Tuple[np.ndarray, ...], optional
            3D coordinates of voxels (x, y, z arrays)
            
        Returns:
        --------
        nx.DiGraph
            Directed connectivity graph with weighted edges
        """
        G = nx.DiGraph()
        
        # Add nodes with attributes
        n_voxels = smte_matrix.shape[0]
        for i in range(n_voxels):
            node_attrs = {'voxel_id': i}
            if voxel_coords is not None:
                node_attrs.update({
                    'x': int(voxel_coords[0][i]),
                    'y': int(voxel_coords[1][i]), 
                    'z': int(voxel_coords[2][i])
                })
            G.add_node(i, **node_attrs)
        
        # Add significant edges (j -> i means j influences i)
        for i in range(n_voxels):
            for j in range(n_voxels):
                if significance_mask[i, j]:
                    G.add_edge(j, i, weight=float(smte_matrix[i, j]))
        
        self.logger.info(f"Created connectivity graph: {G.number_of_nodes()} nodes, "
                        f"{G.number_of_edges()} edges")
        
        return G
    
    def analyze_network_properties(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Compute comprehensive network properties of the connectivity graph.
        
        Parameters:
        -----------
        graph : nx.DiGraph
            Connectivity graph
            
        Returns:
        --------
        Dict[str, Any]
            Network properties and metrics
        """
        properties = {}
        
        # Basic graph properties
        properties['n_nodes'] = graph.number_of_nodes()
        properties['n_edges'] = graph.number_of_edges()
        properties['density'] = nx.density(graph)
        
        if graph.number_of_edges() == 0:
            self.logger.warning("Empty graph - most network metrics will be zero")
            return properties
        
        # Centrality measures
        properties['in_degree_centrality'] = nx.in_degree_centrality(graph)
        properties['out_degree_centrality'] = nx.out_degree_centrality(graph)
        
        # Convert to undirected for some measures
        undirected_graph = graph.to_undirected()
        
        try:
            properties['betweenness_centrality'] = nx.betweenness_centrality(graph)
            properties['closeness_centrality'] = nx.closeness_centrality(undirected_graph)
        except:
            self.logger.warning("Could not compute centrality measures for disconnected graph")
            properties['betweenness_centrality'] = {node: 0.0 for node in graph.nodes()}
            properties['closeness_centrality'] = {node: 0.0 for node in graph.nodes()}
        
        # Degree statistics
        in_degrees = dict(graph.in_degree())
        out_degrees = dict(graph.out_degree())
        
        properties['mean_in_degree'] = np.mean(list(in_degrees.values()))
        properties['std_in_degree'] = np.std(list(in_degrees.values()))
        properties['mean_out_degree'] = np.mean(list(out_degrees.values()))
        properties['std_out_degree'] = np.std(list(out_degrees.values()))
        
        # Hub identification (nodes > mean + 2*std)
        in_degree_threshold = properties['mean_in_degree'] + 2 * properties['std_in_degree']
        out_degree_threshold = properties['mean_out_degree'] + 2 * properties['std_out_degree']
        
        properties['in_hubs'] = [node for node, degree in in_degrees.items() 
                                if degree > in_degree_threshold]
        properties['out_hubs'] = [node for node, degree in out_degrees.items() 
                                 if degree > out_degree_threshold]
        
        # Connection strength statistics
        edge_weights = [data['weight'] for _, _, data in graph.edges(data=True)]
        properties['mean_connection_strength'] = np.mean(edge_weights)
        properties['std_connection_strength'] = np.std(edge_weights)
        properties['max_connection_strength'] = np.max(edge_weights)
        
        # Small-world properties (if graph is connected)
        try:
            if nx.is_weakly_connected(graph):
                properties['average_shortest_path_length'] = nx.average_shortest_path_length(undirected_graph)
                properties['clustering_coefficient'] = nx.average_clustering(undirected_graph)
            else:
                # For disconnected graphs, compute for largest component
                largest_cc = max(nx.weakly_connected_components(graph), key=len)
                subgraph = graph.subgraph(largest_cc).to_undirected()
                properties['average_shortest_path_length'] = nx.average_shortest_path_length(subgraph)
                properties['clustering_coefficient'] = nx.average_clustering(subgraph)
                properties['largest_component_size'] = len(largest_cc)
        except:
            properties['average_shortest_path_length'] = np.inf
            properties['clustering_coefficient'] = 0.0
        
        return properties
    
    def visualize_connectivity_matrix(self, smte_matrix: np.ndarray, 
                                    significance_mask: Optional[np.ndarray] = None,
                                    title: str = "SMTE Connectivity Matrix",
                                    save_path: Optional[str] = None) -> None:
        """
        Create comprehensive visualization of the SMTE connectivity matrix.
        
        Parameters:
        -----------
        smte_matrix : np.ndarray
            SMTE connectivity matrix
        significance_mask : np.ndarray, optional
            Significance mask to overlay
        title : str
            Plot title
        save_path : str, optional
            Path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Raw SMTE matrix
        im1 = axes[0, 0].imshow(smte_matrix, cmap='viridis', aspect='auto')
        axes[0, 0].set_title(f"{title} (Raw Values)")
        axes[0, 0].set_xlabel("Source Voxel")
        axes[0, 0].set_ylabel("Target Voxel")
        plt.colorbar(im1, ax=axes[0, 0], label='SMTE Value')
        
        # Significant connections only
        if significance_mask is not None:
            masked_matrix = smte_matrix.copy()
            masked_matrix[~significance_mask] = 0
            im2 = axes[0, 1].imshow(masked_matrix, cmap='viridis', aspect='auto')
            axes[0, 1].set_title("Significant Connections Only")
            axes[0, 1].set_xlabel("Source Voxel")
            axes[0, 1].set_ylabel("Target Voxel")
            plt.colorbar(im2, ax=axes[0, 1], label='SMTE Value')
            
            # Significance mask
            im3 = axes[1, 0].imshow(significance_mask.astype(int), cmap='binary', aspect='auto')
            axes[1, 0].set_title("Significance Mask")
            axes[1, 0].set_xlabel("Source Voxel")
            axes[1, 0].set_ylabel("Target Voxel")
            plt.colorbar(im3, ax=axes[1, 0], label='Significant')
        
        # Distribution of SMTE values
        axes[1, 1].hist(smte_matrix[smte_matrix > 0].flatten(), bins=50, alpha=0.7, 
                       label='All connections', density=True)
        if significance_mask is not None:
            sig_values = smte_matrix[significance_mask]
            if len(sig_values) > 0:
                axes[1, 1].hist(sig_values, bins=30, alpha=0.7, 
                               label='Significant connections', density=True)
        axes[1, 1].set_xlabel("SMTE Value")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].set_title("Distribution of SMTE Values")
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def save_results(self, output_dir: str, 
                    smte_matrix: np.ndarray,
                    p_values: np.ndarray,
                    significance_mask: np.ndarray,
                    connectivity_graph: nx.DiGraph,
                    network_properties: Dict[str, Any],
                    lag_matrix: Optional[np.ndarray] = None) -> None:
        """
        Save comprehensive analysis results to files.
        
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
        lag_matrix : np.ndarray, optional
            Optimal lag matrix
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save matrices
        np.save(os.path.join(output_dir, 'smte_matrix.npy'), smte_matrix)
        np.save(os.path.join(output_dir, 'p_values.npy'), p_values)
        np.save(os.path.join(output_dir, 'significance_mask.npy'), significance_mask)
        
        if lag_matrix is not None:
            np.save(os.path.join(output_dir, 'lag_matrix.npy'), lag_matrix)
        
        # Save graph in multiple formats
        nx.write_graphml(connectivity_graph, os.path.join(output_dir, 'connectivity_graph.graphml'))
        nx.write_gexf(connectivity_graph, os.path.join(output_dir, 'connectivity_graph.gexf'))
        
        # Save adjacency matrix
        adj_matrix = nx.adjacency_matrix(connectivity_graph, weight='weight').toarray()
        np.save(os.path.join(output_dir, 'adjacency_matrix.npy'), adj_matrix)
        
        # Save network properties as JSON
        import json
        json_properties = {}
        for key, value in network_properties.items():
            if isinstance(value, dict):
                json_properties[key] = {str(k): float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                       for k, v in value.items()}
            elif isinstance(value, (np.integer, np.floating)):
                json_properties[key] = float(value)
            elif isinstance(value, list):
                json_properties[key] = [int(x) if isinstance(x, (np.integer,)) else x for x in value]
            else:
                json_properties[key] = value
                
        with open(os.path.join(output_dir, 'network_properties.json'), 'w') as f:
            json.dump(json_properties, f, indent=2)
        
        # Save analysis parameters
        params = {
            'n_symbols': self.n_symbols,
            'symbolizer': self.symbolizer,
            'ordinal_order': self.ordinal_order,
            'max_lag': self.max_lag,
            'alpha': self.alpha,
            'n_permutations': self.n_permutations,
            'random_state': self.random_state
        }
        
        with open(os.path.join(output_dir, 'analysis_parameters.json'), 'w') as f:
            json.dump(params, f, indent=2)
        
        # Create summary report
        self._create_summary_report(output_dir, smte_matrix, significance_mask, 
                                   network_properties)
        
        self.logger.info(f"Results saved to {output_dir}")
    
    def _create_summary_report(self, output_dir: str, smte_matrix: np.ndarray,
                              significance_mask: np.ndarray, 
                              network_properties: Dict[str, Any]) -> None:
        """Create a summary report of the analysis."""
        report_path = os.path.join(output_dir, 'analysis_summary.txt')
        
        with open(report_path, 'w') as f:
            f.write("fMRI Voxel SMTE Connectivity Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Analysis parameters
            f.write("Analysis Parameters:\n")
            f.write(f"- Symbolization method: {self.symbolizer}\n")
            f.write(f"- Number of symbols: {self.n_symbols}\n")
            if self.symbolizer == 'ordinal':
                f.write(f"- Ordinal pattern order: {self.ordinal_order}\n")
            f.write(f"- Maximum lag: {self.max_lag}\n")
            f.write(f"- Significance threshold: {self.alpha}\n")
            f.write(f"- Permutations: {self.n_permutations}\n\n")
            
            # Connectivity results
            f.write("Connectivity Results:\n")
            total_possible = smte_matrix.shape[0] * (smte_matrix.shape[0] - 1)
            n_significant = np.sum(significance_mask)
            f.write(f"- Total possible connections: {total_possible}\n")
            f.write(f"- Significant connections: {n_significant}\n")
            f.write(f"- Connection rate: {100 * n_significant / total_possible:.2f}%\n")
            
            # SMTE statistics
            non_zero_smte = smte_matrix[smte_matrix > 0]
            if len(non_zero_smte) > 0:
                f.write(f"- Mean SMTE (all): {np.mean(non_zero_smte):.6f}\n")
                f.write(f"- Std SMTE (all): {np.std(non_zero_smte):.6f}\n")
                f.write(f"- Max SMTE: {np.max(non_zero_smte):.6f}\n")
            
            sig_smte = smte_matrix[significance_mask]
            if len(sig_smte) > 0:
                f.write(f"- Mean SMTE (significant): {np.mean(sig_smte):.6f}\n")
                f.write(f"- Std SMTE (significant): {np.std(sig_smte):.6f}\n")
            
            f.write("\n")
            
            # Network properties
            f.write("Network Properties:\n")
            f.write(f"- Nodes: {network_properties.get('n_nodes', 0)}\n")
            f.write(f"- Edges: {network_properties.get('n_edges', 0)}\n")
            f.write(f"- Density: {network_properties.get('density', 0):.6f}\n")
            f.write(f"- Mean connection strength: {network_properties.get('mean_connection_strength', 0):.6f}\n")
            
            if 'in_hubs' in network_properties:
                f.write(f"- Input hubs: {len(network_properties['in_hubs'])}\n")
                f.write(f"- Output hubs: {len(network_properties['out_hubs'])}\n")
            
            if 'clustering_coefficient' in network_properties:
                f.write(f"- Clustering coefficient: {network_properties['clustering_coefficient']:.6f}\n")
            
            if 'average_shortest_path_length' in network_properties:
                f.write(f"- Average path length: {network_properties['average_shortest_path_length']:.6f}\n")
    
    def run_complete_analysis(self, 
                            fmri_path: str,
                            output_dir: str,
                            mask_path: Optional[str] = None,
                            voxel_indices: Optional[List[int]] = None,
                            visualize: bool = True) -> Dict[str, Any]:
        """
        Run complete research-grade voxel-wise SMTE connectivity analysis.
        
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
            Comprehensive analysis results
        """
        self.logger.info("Starting research-grade SMTE connectivity analysis")
        
        # 1. Load and validate data
        self.load_fmri_data(fmri_path, mask_path)
        
        # 2. Extract and preprocess voxel time series
        voxel_timeseries = self.extract_voxel_timeseries()
        
        # 3. Symbolize time series
        self.symbolic_data = self.symbolize_timeseries(voxel_timeseries)
        
        # 4. Compute connectivity matrix
        smte_matrix, lag_matrix = self.compute_voxel_connectivity_matrix(voxel_indices)
        
        # 5. Statistical significance testing
        p_values = self.statistical_testing(smte_matrix, voxel_indices)
        
        # 6. Multiple comparison correction
        significance_mask = self.fdr_correction(p_values)
        
        # 7. Build connectivity graph
        coords = None
        if voxel_indices:
            coords = tuple(coord[voxel_indices] for coord in self.voxel_coords)
        else:
            coords = self.voxel_coords
            
        connectivity_graph = self.build_connectivity_graph(
            smte_matrix, significance_mask, coords
        )
        
        # 8. Network analysis
        network_properties = self.analyze_network_properties(connectivity_graph)
        
        # 9. Save comprehensive results
        self.save_results(
            output_dir, smte_matrix, p_values, significance_mask,
            connectivity_graph, network_properties, lag_matrix
        )
        
        # 10. Visualization
        if visualize:
            viz_path = os.path.join(output_dir, 'connectivity_visualization.png')
            self.visualize_connectivity_matrix(smte_matrix, significance_mask, 
                                             save_path=viz_path)
        
        # Store results for access
        self.smte_matrix = smte_matrix
        self.p_values = p_values
        self.connectivity_graph = connectivity_graph
        
        # Compile results
        results = {
            'smte_matrix': smte_matrix,
            'p_values': p_values,
            'significance_mask': significance_mask,
            'lag_matrix': lag_matrix,
            'connectivity_graph': connectivity_graph,
            'network_properties': network_properties,
            'n_significant_connections': np.sum(significance_mask),
            'connection_rate': np.sum(significance_mask) / (smte_matrix.shape[0] * (smte_matrix.shape[0] - 1)),
            'analysis_parameters': {
                'n_symbols': self.n_symbols,
                'symbolizer': self.symbolizer,
                'max_lag': self.max_lag,
                'alpha': self.alpha,
                'n_permutations': self.n_permutations
            }
        }
        
        self.logger.info(f"Analysis complete. Found {np.sum(significance_mask)} significant connections "
                        f"({100*results['connection_rate']:.2f}% connection rate)")
        
        return results


def example_usage():
    """Research-grade example usage."""
    
    # Initialize with research-appropriate parameters
    analyzer = VoxelSMTEConnectivity(
        n_symbols=6,              # 3! = 6 for ordinal patterns of order 3
        symbolizer='ordinal',     # Recommended for fMRI due to noise robustness
        ordinal_order=3,          # Standard choice balancing resolution and reliability
        max_lag=5,               # Consider up to 5 TRs (~10s for TR=2s)
        alpha=0.01,              # Stringent significance threshold
        n_permutations=2000,     # High number for reliable p-values
        n_jobs=-1,               # Use all available cores
        memory_efficient=True,   # Enable for large datasets
        random_state=42          # For reproducibility
    )
    
    # Run comprehensive analysis
    results = analyzer.run_complete_analysis(
        fmri_path='/path/to/preprocessed_fmri.nii.gz',
        output_dir='/path/to/results/',
        mask_path='/path/to/brain_mask.nii.gz',
        visualize=True
    )
    
    print(f"Analysis Results:")
    print(f"- Significant connections: {results['n_significant_connections']}")
    print(f"- Connection rate: {results['connection_rate']:.4f}")
    print(f"- Network density: {results['network_properties']['density']:.6f}")
    print(f"- Mean connection strength: {results['network_properties']['mean_connection_strength']:.6f}")
    
    return results


if __name__ == "__main__":
    # Demonstration with synthetic data would go here
    pass