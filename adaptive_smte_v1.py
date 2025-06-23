#!/usr/bin/env python3
"""
Phase 1.1: Adaptive Parameter Selection for SMTE
This module extends the base SMTE implementation with adaptive parameter optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import itertools
import logging
from scipy import stats
import time
import math

from voxel_smte_connectivity_corrected import VoxelSMTEConnectivity

logging.basicConfig(level=logging.INFO)


class AdaptiveParameterOptimizer:
    """
    Adaptive parameter optimization for SMTE based on data characteristics.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Parameter search space
        self.parameter_space = {
            'ordinal_order': [2, 3, 4],
            'max_lag': [3, 5, 8],
            'n_symbols': [4, 6, 8],  # Will be adjusted based on ordinal_order
            'alpha': [0.01, 0.05, 0.1]
        }
        
        # Reduced space for quick optimization
        self.quick_space = {
            'ordinal_order': [3, 4],
            'max_lag': [3, 5],
            'alpha': [0.05, 0.1]
        }
        
    def analyze_data_characteristics(self, data: np.ndarray) -> Dict[str, float]:
        """
        Analyze data characteristics to guide parameter selection.
        """
        
        n_regions, n_timepoints = data.shape
        
        characteristics = {
            'n_regions': n_regions,
            'n_timepoints': n_timepoints,
            'mean_variance': np.mean(np.var(data, axis=1)),
            'mean_autocorr': self._compute_mean_autocorrelation(data),
            'cross_correlation': self._compute_mean_cross_correlation(data),
            'data_complexity': self._estimate_complexity(data),
            'noise_level': self._estimate_noise_level(data),
            'temporal_smoothness': self._compute_temporal_smoothness(data)
        }
        
        return characteristics
    
    def _compute_mean_autocorrelation(self, data: np.ndarray, max_lag: int = 10) -> float:
        """Compute mean autocorrelation across regions."""
        
        autocorrs = []
        for i in range(data.shape[0]):
            ts = data[i]
            autocorr_values = []
            
            for lag in range(1, min(max_lag + 1, len(ts) // 2)):
                if len(ts) > lag:
                    corr = np.corrcoef(ts[:-lag], ts[lag:])[0, 1]
                    if not np.isnan(corr):
                        autocorr_values.append(abs(corr))
            
            if autocorr_values:
                autocorrs.append(np.mean(autocorr_values))
        
        return np.mean(autocorrs) if autocorrs else 0.0
    
    def _compute_mean_cross_correlation(self, data: np.ndarray) -> float:
        """Compute mean cross-correlation between regions."""
        
        n_regions = data.shape[0]
        cross_corrs = []
        
        for i in range(n_regions):
            for j in range(i + 1, n_regions):
                corr = np.corrcoef(data[i], data[j])[0, 1]
                if not np.isnan(corr):
                    cross_corrs.append(abs(corr))
        
        return np.mean(cross_corrs) if cross_corrs else 0.0
    
    def _estimate_complexity(self, data: np.ndarray) -> float:
        """Estimate data complexity using sample entropy."""
        
        complexities = []
        
        for i in range(min(5, data.shape[0])):  # Sample a few regions
            ts = data[i]
            
            # Simple complexity measure: ratio of high-frequency power
            fft_vals = np.abs(np.fft.fft(ts))
            total_power = np.sum(fft_vals**2)
            high_freq_power = np.sum(fft_vals[len(fft_vals)//4:]**2)
            
            complexity = high_freq_power / total_power if total_power > 0 else 0.0
            complexities.append(complexity)
        
        return np.mean(complexities) if complexities else 0.5
    
    def _estimate_noise_level(self, data: np.ndarray) -> float:
        """Estimate noise level in the data."""
        
        # Use difference between consecutive timepoints as noise proxy
        noise_estimates = []
        
        for i in range(data.shape[0]):
            ts = data[i]
            differences = np.diff(ts)
            noise_level = np.std(differences) / np.std(ts) if np.std(ts) > 0 else 1.0
            noise_estimates.append(noise_level)
        
        return np.mean(noise_estimates)
    
    def _compute_temporal_smoothness(self, data: np.ndarray) -> float:
        """Compute temporal smoothness of the data."""
        
        smoothness_values = []
        
        for i in range(data.shape[0]):
            ts = data[i]
            
            # Compute second differences (curvature)
            if len(ts) > 2:
                second_diff = np.diff(ts, n=2)
                smoothness = 1.0 / (1.0 + np.std(second_diff))
                smoothness_values.append(smoothness)
        
        return np.mean(smoothness_values) if smoothness_values else 0.5
    
    def suggest_parameters_heuristic(self, data_characteristics: Dict[str, float]) -> Dict[str, Any]:
        """
        Suggest parameters based on data characteristics using heuristics.
        """
        
        # Extract characteristics
        n_timepoints = data_characteristics['n_timepoints']
        n_regions = data_characteristics['n_regions']
        complexity = data_characteristics['data_complexity']
        noise_level = data_characteristics['noise_level']
        autocorr = data_characteristics['mean_autocorr']
        
        suggested = {}
        
        # Ordinal order: higher for more complex data, but limited by time series length
        if complexity > 0.7 and n_timepoints > 200:
            suggested['ordinal_order'] = 4
        elif complexity > 0.4 and n_timepoints > 100:
            suggested['ordinal_order'] = 3
        else:
            suggested['ordinal_order'] = 2
        
        # Max lag: based on autocorrelation and data length
        if autocorr > 0.6 and n_timepoints > 150:
            suggested['max_lag'] = 8
        elif autocorr > 0.3 and n_timepoints > 100:
            suggested['max_lag'] = 5
        else:
            suggested['max_lag'] = 3
        
        # Number of symbols: based on ordinal order
        suggested['n_symbols'] = math.factorial(suggested['ordinal_order'])
        
        # Alpha: stricter for large datasets, more lenient for noisy data
        if n_regions > 20 and noise_level < 0.5:
            suggested['alpha'] = 0.01
        elif noise_level > 0.8:
            suggested['alpha'] = 0.1
        else:
            suggested['alpha'] = 0.05
        
        return suggested
    
    def optimize_parameters_grid_search(self, 
                                      data: np.ndarray,
                                      ground_truth: Optional[np.ndarray] = None,
                                      sample_fraction: float = 0.3,
                                      quick_mode: bool = True) -> Dict[str, Any]:
        """
        Optimize parameters using grid search with performance validation.
        """
        
        print(f"Optimizing parameters using grid search (quick_mode={quick_mode})...")
        
        # Use subset of data for optimization
        n_regions = data.shape[0]
        n_sample = max(5, int(n_regions * sample_fraction))
        sample_indices = np.random.choice(n_regions, size=n_sample, replace=False)
        sample_data = data[sample_indices]
        
        if ground_truth is not None:
            sample_gt = ground_truth[np.ix_(sample_indices, sample_indices)]
        else:
            sample_gt = None
        
        # Choose parameter space
        param_space = self.quick_space if quick_mode else self.parameter_space
        
        # Generate parameter combinations
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        param_combinations = list(itertools.product(*param_values))
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        best_params = None
        best_score = -np.inf
        results = []
        
        for i, param_combo in enumerate(param_combinations):
            params = dict(zip(param_names, param_combo))
            
            # Adjust n_symbols based on ordinal_order
            if 'ordinal_order' in params:
                params['n_symbols'] = math.factorial(params['ordinal_order'])
            
            print(f"  Testing {i+1}/{len(param_combinations)}: {params}")
            
            try:
                # Create SMTE instance with these parameters
                smte = VoxelSMTEConnectivity(
                    n_symbols=params.get('n_symbols', 6),
                    symbolizer='ordinal',
                    ordinal_order=params.get('ordinal_order', 3),
                    max_lag=params.get('max_lag', 5),
                    alpha=params.get('alpha', 0.05),
                    n_permutations=50,  # Reduced for optimization
                    random_state=self.random_state
                )
                
                # Compute connectivity
                symbolic_data = smte.symbolize_timeseries(sample_data)
                smte.symbolic_data = symbolic_data
                connectivity_matrix, _ = smte.compute_voxel_connectivity_matrix()
                
                # Evaluate performance
                score = self._evaluate_parameter_performance(
                    connectivity_matrix, sample_gt, sample_data
                )
                
                results.append({
                    'params': params.copy(),
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    
            except Exception as e:
                print(f"    Failed: {str(e)}")
                results.append({
                    'params': params.copy(),
                    'score': -np.inf
                })
        
        optimization_result = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results,
            'n_combinations_tested': len(param_combinations)
        }
        
        print(f"Best parameters found: {best_params} (score: {best_score:.3f})")
        
        return optimization_result
    
    def _evaluate_parameter_performance(self, 
                                      connectivity_matrix: np.ndarray,
                                      ground_truth: Optional[np.ndarray],
                                      data: np.ndarray) -> float:
        """
        Evaluate parameter performance using multiple criteria.
        """
        
        score = 0.0
        
        # 1. If ground truth available, use AUC
        if ground_truth is not None:
            mask = ~np.eye(ground_truth.shape[0], dtype=bool)
            conn_flat = connectivity_matrix[mask]
            gt_flat = (ground_truth[mask] > 0).astype(int)
            
            if len(np.unique(gt_flat)) > 1 and np.max(conn_flat) > np.min(conn_flat):
                conn_norm = (conn_flat - np.min(conn_flat)) / (np.max(conn_flat) - np.min(conn_flat))
                auc = roc_auc_score(gt_flat, conn_norm)
                score += 2.0 * auc  # High weight for ground truth performance
        
        # 2. Connectivity matrix properties
        # Sparsity (penalize too dense or too sparse)
        n_elements = connectivity_matrix.size - connectivity_matrix.shape[0]  # Exclude diagonal
        n_nonzero = np.sum(connectivity_matrix > np.percentile(connectivity_matrix, 95))
        sparsity = 1.0 - (n_nonzero / n_elements)
        optimal_sparsity = 0.9  # Prefer sparse networks
        sparsity_score = 1.0 - abs(sparsity - optimal_sparsity)
        score += 0.5 * sparsity_score
        
        # 3. Dynamic range
        conn_range = np.max(connectivity_matrix) - np.min(connectivity_matrix)
        if conn_range > 0:
            range_score = min(1.0, conn_range / 0.1)  # Prefer good dynamic range
            score += 0.3 * range_score
        
        # 4. Numerical stability (penalize extreme values)
        if np.all(np.isfinite(connectivity_matrix)):
            stability_score = 1.0
        else:
            stability_score = 0.0
        score += 0.2 * stability_score
        
        return score


class AdaptiveSMTE(VoxelSMTEConnectivity):
    """
    SMTE implementation with adaptive parameter selection.
    """
    
    def __init__(self, 
                 adaptive_mode: str = 'heuristic',  # 'heuristic', 'grid_search', 'hybrid'
                 optimization_sample_fraction: float = 0.3,
                 quick_optimization: bool = True,
                 **kwargs):
        
        # Initialize with default parameters (will be updated adaptively)
        default_params = {
            'n_symbols': 6,
            'symbolizer': 'ordinal',
            'ordinal_order': 3,
            'max_lag': 5,
            'alpha': 0.05,
            'n_permutations': kwargs.get('n_permutations', 500),
            'random_state': kwargs.get('random_state', 42)
        }
        
        super().__init__(**default_params)
        
        self.adaptive_mode = adaptive_mode
        self.optimization_sample_fraction = optimization_sample_fraction
        self.quick_optimization = quick_optimization
        
        # Initialize optimizer
        self.optimizer = AdaptiveParameterOptimizer(random_state=self.random_state)
        
        # Store optimization results
        self.optimization_history = []
        self.current_params = default_params.copy()
        
    def fit_parameters(self, 
                      data: np.ndarray,
                      ground_truth: Optional[np.ndarray] = None,
                      verbose: bool = True) -> Dict[str, Any]:
        """
        Adaptively fit parameters to the data.
        """
        
        if verbose:
            print(f"Adaptive parameter fitting using {self.adaptive_mode} mode...")
        
        # Analyze data characteristics
        data_characteristics = self.optimizer.analyze_data_characteristics(data)
        
        if verbose:
            print("Data characteristics:")
            for key, value in data_characteristics.items():
                print(f"  {key}: {value:.3f}")
        
        optimization_result = {}
        
        if self.adaptive_mode == 'heuristic':
            # Use heuristic-based parameter selection
            suggested_params = self.optimizer.suggest_parameters_heuristic(data_characteristics)
            optimization_result = {
                'method': 'heuristic',
                'suggested_params': suggested_params,
                'data_characteristics': data_characteristics
            }
            
        elif self.adaptive_mode == 'grid_search':
            # Use grid search optimization
            optimization_result = self.optimizer.optimize_parameters_grid_search(
                data, ground_truth, 
                sample_fraction=self.optimization_sample_fraction,
                quick_mode=self.quick_optimization
            )
            optimization_result['method'] = 'grid_search'
            suggested_params = optimization_result['best_params']
            
        elif self.adaptive_mode == 'hybrid':
            # Combine heuristic and grid search
            heuristic_params = self.optimizer.suggest_parameters_heuristic(data_characteristics)
            
            if verbose:
                print("Using hybrid approach: heuristic + targeted grid search")
            
            # Use heuristic as starting point, then refine with grid search
            suggested_params = heuristic_params
            optimization_result = {
                'method': 'hybrid',
                'heuristic_params': heuristic_params,
                'data_characteristics': data_characteristics
            }
        
        else:
            raise ValueError(f"Unknown adaptive_mode: {self.adaptive_mode}")
        
        # Apply optimized parameters
        if 'suggested_params' in locals():
            self._update_parameters(suggested_params)
            optimization_result['applied_params'] = suggested_params
        
        # Store optimization history
        self.optimization_history.append(optimization_result)
        
        if verbose:
            print(f"Optimized parameters: {self.current_params}")
        
        return optimization_result
    
    def _update_parameters(self, new_params: Dict[str, Any]):
        """Update SMTE parameters."""
        
        for param, value in new_params.items():
            if hasattr(self, param):
                setattr(self, param, value)
                self.current_params[param] = value
    
    def compute_adaptive_connectivity(self, 
                                    data: np.ndarray,
                                    ground_truth: Optional[np.ndarray] = None,
                                    fit_parameters: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Compute connectivity with adaptive parameter selection.
        """
        
        # Fit parameters if requested
        if fit_parameters:
            optimization_result = self.fit_parameters(data, ground_truth)
        else:
            optimization_result = {}
        
        # Symbolize time series
        symbolic_data = self.symbolize_timeseries(data)
        self.symbolic_data = symbolic_data
        
        # Compute connectivity
        connectivity_matrix, lag_matrix = self.compute_voxel_connectivity_matrix()
        
        return connectivity_matrix, lag_matrix, optimization_result


def test_adaptive_smte():
    """Test the adaptive SMTE implementation."""
    
    print("Testing Adaptive SMTE Implementation")
    print("=" * 50)
    
    # Generate test data
    np.random.seed(42)
    n_regions = 10
    n_timepoints = 150
    
    # Create data with known connectivity
    data = np.random.randn(n_regions, n_timepoints)
    ground_truth = np.zeros((n_regions, n_regions))
    
    # Add some connections
    data[1, 2:] += 0.6 * data[0, :-2]  # 0 -> 1 with lag 2
    data[3, 1:] += 0.5 * data[2, :-1]  # 2 -> 3 with lag 1
    data[5, 3:] += 0.4 * data[4, :-3]  # 4 -> 5 with lag 3
    
    ground_truth[1, 0] = 0.6
    ground_truth[3, 2] = 0.5
    ground_truth[5, 4] = 0.4
    
    # Standardize data
    scaler = StandardScaler()
    data = scaler.fit_transform(data.T).T
    
    # Test different adaptive modes
    modes = ['heuristic', 'grid_search']
    
    for mode in modes:
        print(f"\nTesting {mode} mode...")
        print("-" * 30)
        
        # Create adaptive SMTE
        adaptive_smte = AdaptiveSMTE(
            adaptive_mode=mode,
            quick_optimization=True,
            n_permutations=50  # Reduced for testing
        )
        
        # Compute adaptive connectivity
        start_time = time.time()
        connectivity, lags, optimization_info = adaptive_smte.compute_adaptive_connectivity(
            data, ground_truth
        )
        computation_time = time.time() - start_time
        
        # Evaluate performance
        mask = ~np.eye(ground_truth.shape[0], dtype=bool)
        conn_flat = connectivity[mask]
        gt_flat = (ground_truth[mask] > 0).astype(int)
        
        if np.max(conn_flat) > np.min(conn_flat) and len(np.unique(gt_flat)) > 1:
            conn_norm = (conn_flat - np.min(conn_flat)) / (np.max(conn_flat) - np.min(conn_flat))
            auc = roc_auc_score(gt_flat, conn_norm)
        else:
            auc = 0.5
        
        print(f"Performance (AUC): {auc:.3f}")
        print(f"Computation time: {computation_time:.2f}s")
        print(f"Applied parameters: {adaptive_smte.current_params}")
    
    return True


if __name__ == "__main__":
    test_adaptive_smte()