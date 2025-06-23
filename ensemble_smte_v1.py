#!/usr/bin/env python3
"""
Phase 2.2: Ensemble Statistical Framework for SMTE
This module implements ensemble methods to improve statistical power and robustness.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import logging
from scipy import stats
from scipy.stats import combine_pvalues
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from multiscale_smte_v1 import MultiScaleSMTE

logging.basicConfig(level=logging.INFO)


class SurrogateGenerator:
    """
    Advanced surrogate data generation methods for statistical testing.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Available surrogate methods
        self.surrogate_methods = {
            'aaft': self._amplitude_adjusted_fourier_transform,
            'iaaft': self._iterative_amplitude_adjusted_fourier_transform,
            'twin_surrogate': self._twin_surrogate,
            'bootstrap': self._bootstrap_surrogate,
            'phase_randomization': self._phase_randomization,
            'constrained_randomization': self._constrained_randomization
        }
    
    def generate_surrogates(self, 
                          data: np.ndarray,
                          method: str = 'aaft',
                          n_surrogates: int = 100) -> np.ndarray:
        """
        Generate surrogate datasets for statistical testing.
        
        Parameters:
        -----------
        data : np.ndarray
            Original time series data (n_regions, n_timepoints)
        method : str
            Surrogate generation method
        n_surrogates : int
            Number of surrogate datasets to generate
            
        Returns:
        --------
        np.ndarray
            Surrogate datasets (n_surrogates, n_regions, n_timepoints)
        """
        
        if method not in self.surrogate_methods:
            raise ValueError(f"Unknown surrogate method: {method}. Available: {list(self.surrogate_methods.keys())}")
        
        surrogate_func = self.surrogate_methods[method]
        n_regions, n_timepoints = data.shape
        
        surrogates = np.zeros((n_surrogates, n_regions, n_timepoints))
        
        for i in range(n_surrogates):
            surrogates[i] = surrogate_func(data)
        
        return surrogates
    
    def _amplitude_adjusted_fourier_transform(self, data: np.ndarray) -> np.ndarray:
        """Amplitude Adjusted Fourier Transform (AAFT) surrogate."""
        
        n_regions, n_timepoints = data.shape
        surrogate = np.zeros_like(data)
        
        for i in range(n_regions):
            ts = data[i]
            
            # Sort the original time series
            sorted_original = np.sort(ts)
            
            # Generate Gaussian noise with same length
            gaussian_noise = np.random.randn(n_timepoints)
            
            # Take FFT of Gaussian noise
            fft_noise = np.fft.fft(gaussian_noise)
            
            # Use phases from noise, amplitudes from original
            fft_original = np.fft.fft(ts)
            amplitudes = np.abs(fft_original)
            phases = np.angle(fft_noise)
            
            # Reconstruct signal
            surrogate_fft = amplitudes * np.exp(1j * phases)
            surrogate_ts = np.real(np.fft.ifft(surrogate_fft))
            
            # Rank order the surrogate to match original amplitude distribution
            sorted_indices = np.argsort(surrogate_ts)
            rank_ordered_surrogate = np.zeros_like(surrogate_ts)
            rank_ordered_surrogate[sorted_indices] = sorted_original
            
            surrogate[i] = rank_ordered_surrogate
        
        return surrogate
    
    def _iterative_amplitude_adjusted_fourier_transform(self, data: np.ndarray, max_iter: int = 10) -> np.ndarray:
        """Iterative Amplitude Adjusted Fourier Transform (IAAFT) surrogate."""
        
        n_regions, n_timepoints = data.shape
        surrogate = np.zeros_like(data)
        
        for i in range(n_regions):
            ts = data[i]
            
            # Initialize with AAFT
            current_surrogate = self._amplitude_adjusted_fourier_transform(ts.reshape(1, -1))[0]
            
            # Iterative refinement
            for iteration in range(max_iter):
                # Preserve amplitude spectrum
                fft_original = np.fft.fft(ts)
                fft_surrogate = np.fft.fft(current_surrogate)
                
                amplitudes_original = np.abs(fft_original)
                phases_surrogate = np.angle(fft_surrogate)
                
                # Reconstruct with original amplitudes, surrogate phases
                new_fft = amplitudes_original * np.exp(1j * phases_surrogate)
                new_surrogate = np.real(np.fft.ifft(new_fft))
                
                # Preserve rank order (amplitude distribution)
                sorted_original = np.sort(ts)
                sorted_indices = np.argsort(new_surrogate)
                current_surrogate = np.zeros_like(new_surrogate)
                current_surrogate[sorted_indices] = sorted_original
            
            surrogate[i] = current_surrogate
        
        return surrogate
    
    def _twin_surrogate(self, data: np.ndarray) -> np.ndarray:
        """Twin surrogate method."""
        
        # Simple implementation: time-shifted version
        n_regions, n_timepoints = data.shape
        surrogate = np.zeros_like(data)
        
        for i in range(n_regions):
            # Random circular shift
            shift = np.random.randint(1, n_timepoints)
            surrogate[i] = np.roll(data[i], shift)
        
        return surrogate
    
    def _bootstrap_surrogate(self, data: np.ndarray) -> np.ndarray:
        """Bootstrap surrogate method."""
        
        n_regions, n_timepoints = data.shape
        surrogate = np.zeros_like(data)
        
        for i in range(n_regions):
            # Bootstrap sampling with replacement
            indices = np.random.choice(n_timepoints, size=n_timepoints, replace=True)
            surrogate[i] = data[i, indices]
        
        return surrogate
    
    def _phase_randomization(self, data: np.ndarray) -> np.ndarray:
        """Phase randomization surrogate."""
        
        n_regions, n_timepoints = data.shape
        surrogate = np.zeros_like(data)
        
        for i in range(n_regions):
            ts = data[i]
            
            # Take FFT
            fft_data = np.fft.fft(ts)
            amplitudes = np.abs(fft_data)
            
            # Randomize phases
            random_phases = np.random.uniform(0, 2*np.pi, n_timepoints)
            
            # Reconstruct with randomized phases
            surrogate_fft = amplitudes * np.exp(1j * random_phases)
            surrogate[i] = np.real(np.fft.ifft(surrogate_fft))
        
        return surrogate
    
    def _constrained_randomization(self, data: np.ndarray) -> np.ndarray:
        """Constrained randomization preserving local structure."""
        
        n_regions, n_timepoints = data.shape
        surrogate = np.zeros_like(data)
        
        # Block size for preserving local structure
        block_size = max(5, n_timepoints // 20)
        
        for i in range(n_regions):
            ts = data[i]
            
            # Create blocks
            n_blocks = n_timepoints // block_size
            blocks = []
            
            for j in range(n_blocks):
                start_idx = j * block_size
                end_idx = min((j + 1) * block_size, n_timepoints)
                blocks.append(ts[start_idx:end_idx])
            
            # Randomly permute blocks
            np.random.shuffle(blocks)
            
            # Reconstruct time series
            surrogate_ts = np.concatenate(blocks)
            
            # Pad if necessary
            if len(surrogate_ts) < n_timepoints:
                padding = n_timepoints - len(surrogate_ts)
                surrogate_ts = np.pad(surrogate_ts, (0, padding), mode='edge')
            
            surrogate[i] = surrogate_ts[:n_timepoints]
        
        return surrogate


class EnsembleStatisticalTesting:
    """
    Ensemble statistical testing framework using multiple surrogate methods.
    """
    
    def __init__(self, 
                 surrogate_methods: List[str] = ['aaft', 'iaaft', 'phase_randomization'],
                 n_surrogates_per_method: int = 50,
                 combination_method: str = 'fisher',
                 random_state: int = 42):
        
        self.surrogate_methods = surrogate_methods
        self.n_surrogates_per_method = n_surrogates_per_method
        self.combination_method = combination_method
        self.random_state = random_state
        
        # Initialize surrogate generator
        self.surrogate_generator = SurrogateGenerator(random_state=random_state)
        
        # Available p-value combination methods
        self.combination_methods = {
            'fisher': self._fisher_combination,
            'stouffer': self._stouffer_combination,
            'tippett': self._tippett_combination,
            'weighted_fisher': self._weighted_fisher_combination
        }
    
    def ensemble_statistical_test(self, 
                                connectivity_matrix: np.ndarray,
                                data: np.ndarray,
                                smte_analyzer: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform ensemble statistical testing using multiple surrogate methods.
        
        Parameters:
        -----------
        connectivity_matrix : np.ndarray
            Observed connectivity matrix
        data : np.ndarray
            Original time series data
        smte_analyzer : Any
            SMTE analyzer instance
            
        Returns:
        --------
        Tuple[np.ndarray, Dict[str, Any]]
            Combined p-values and detailed results
        """
        
        print(f"Performing ensemble statistical testing with {len(self.surrogate_methods)} methods...")
        
        n_regions = connectivity_matrix.shape[0]
        method_p_values = {}
        method_details = {}
        
        # Test with each surrogate method
        for method in self.surrogate_methods:
            print(f"  Testing with {method} surrogates...")
            
            method_p_vals, method_info = self._test_with_surrogate_method(
                connectivity_matrix, data, smte_analyzer, method
            )
            
            method_p_values[method] = method_p_vals
            method_details[method] = method_info
        
        # Combine p-values across methods
        print(f"  Combining p-values using {self.combination_method} method...")
        combined_p_values = self._combine_p_values(method_p_values)
        
        # Calculate ensemble statistics
        ensemble_stats = self._calculate_ensemble_statistics(method_p_values, method_details)
        
        results = {
            'combined_p_values': combined_p_values,
            'individual_p_values': method_p_values,
            'method_details': method_details,
            'ensemble_statistics': ensemble_stats,
            'combination_method': self.combination_method
        }
        
        return combined_p_values, results
    
    def _test_with_surrogate_method(self, 
                                  connectivity_matrix: np.ndarray,
                                  data: np.ndarray,
                                  smte_analyzer: Any,
                                  method: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Test with a specific surrogate method."""
        
        n_regions = connectivity_matrix.shape[0]
        
        # Generate surrogates
        surrogates = self.surrogate_generator.generate_surrogates(
            data, method, self.n_surrogates_per_method
        )
        
        # Compute null distribution
        null_connectivity_values = []
        
        for surrogate_idx in range(self.n_surrogates_per_method):
            # Symbolize surrogate data
            surrogate_data = surrogates[surrogate_idx]
            symbolic_surrogate = smte_analyzer.symbolize_timeseries(surrogate_data)
            
            # Temporarily store original symbolic data
            original_symbolic = smte_analyzer.symbolic_data
            smte_analyzer.symbolic_data = symbolic_surrogate
            
            # Compute connectivity for surrogate
            surrogate_connectivity, _ = smte_analyzer.compute_voxel_connectivity_matrix()
            null_connectivity_values.append(surrogate_connectivity)
            
            # Restore original symbolic data
            smte_analyzer.symbolic_data = original_symbolic
        
        # Convert to array
        null_distribution = np.array(null_connectivity_values)  # (n_surrogates, n_regions, n_regions)
        
        # Compute p-values
        p_values = np.zeros((n_regions, n_regions))
        
        for i in range(n_regions):
            for j in range(n_regions):
                if i != j:
                    observed_value = connectivity_matrix[i, j]
                    null_values = null_distribution[:, i, j]
                    
                    # P-value: proportion of null values >= observed value
                    p_value = np.mean(null_values >= observed_value)
                    p_values[i, j] = max(p_value, 1.0 / (self.n_surrogates_per_method + 1))  # Avoid p=0
        
        method_info = {
            'n_surrogates': self.n_surrogates_per_method,
            'null_distribution_stats': {
                'mean': np.mean(null_distribution),
                'std': np.std(null_distribution),
                'min': np.min(null_distribution),
                'max': np.max(null_distribution)
            }
        }
        
        return p_values, method_info
    
    def _combine_p_values(self, method_p_values: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine p-values across different surrogate methods."""
        
        if self.combination_method not in self.combination_methods:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
        
        return self.combination_methods[self.combination_method](method_p_values)
    
    def _fisher_combination(self, method_p_values: Dict[str, np.ndarray]) -> np.ndarray:
        """Fisher's method for combining p-values."""
        
        # Get shape from first method
        first_method = list(method_p_values.keys())[0]
        shape = method_p_values[first_method].shape
        combined_p = np.zeros(shape)
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                if i != j:
                    p_vals = [method_p_values[method][i, j] for method in method_p_values.keys()]
                    
                    # Fisher's method: -2 * sum(log(p_i)) ~ chi^2(2k)
                    try:
                        stat, combined_p[i, j] = combine_pvalues(p_vals, method='fisher')
                    except (ValueError, RuntimeWarning):
                        # Fallback: geometric mean
                        combined_p[i, j] = np.prod(p_vals) ** (1.0 / len(p_vals))
        
        return combined_p
    
    def _stouffer_combination(self, method_p_values: Dict[str, np.ndarray]) -> np.ndarray:
        """Stouffer's method for combining p-values."""
        
        first_method = list(method_p_values.keys())[0]
        shape = method_p_values[first_method].shape
        combined_p = np.zeros(shape)
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                if i != j:
                    p_vals = [method_p_values[method][i, j] for method in method_p_values.keys()]
                    
                    # Stouffer's method: sum(Z_i) / sqrt(k)
                    try:
                        stat, combined_p[i, j] = combine_pvalues(p_vals, method='stouffer')
                    except (ValueError, RuntimeWarning):
                        combined_p[i, j] = np.mean(p_vals)
        
        return combined_p
    
    def _tippett_combination(self, method_p_values: Dict[str, np.ndarray]) -> np.ndarray:
        """Tippett's method (minimum p-value) for combining p-values."""
        
        first_method = list(method_p_values.keys())[0]
        shape = method_p_values[first_method].shape
        combined_p = np.zeros(shape)
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                if i != j:
                    p_vals = [method_p_values[method][i, j] for method in method_p_values.keys()]
                    
                    # Tippett's method: minimum p-value with Bonferroni correction
                    min_p = np.min(p_vals)
                    combined_p[i, j] = min(1.0, min_p * len(p_vals))
        
        return combined_p
    
    def _weighted_fisher_combination(self, method_p_values: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted Fisher's method with method-specific weights."""
        
        # Define weights for different methods based on reliability
        method_weights = {
            'aaft': 1.0,
            'iaaft': 1.2,  # Slightly higher weight for iterative method
            'phase_randomization': 0.8,
            'twin_surrogate': 0.6,
            'bootstrap': 0.5,
            'constrained_randomization': 0.9
        }
        
        first_method = list(method_p_values.keys())[0]
        shape = method_p_values[first_method].shape
        combined_p = np.zeros(shape)
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                if i != j:
                    weighted_log_p = 0.0
                    total_weight = 0.0
                    
                    for method, p_vals in method_p_values.items():
                        weight = method_weights.get(method, 1.0)
                        p_val = p_vals[i, j]
                        
                        if p_val > 0:
                            weighted_log_p += weight * np.log(p_val)
                            total_weight += weight
                    
                    if total_weight > 0:
                        # Convert back to p-value
                        chi2_stat = -2 * weighted_log_p
                        df = 2 * total_weight
                        combined_p[i, j] = 1 - stats.chi2.cdf(chi2_stat, df)
                    else:
                        combined_p[i, j] = 1.0
        
        return combined_p
    
    def _calculate_ensemble_statistics(self, 
                                     method_p_values: Dict[str, np.ndarray],
                                     method_details: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate ensemble statistics across methods."""
        
        stats_dict = {}
        
        # P-value consistency across methods
        p_value_arrays = list(method_p_values.values())
        
        if len(p_value_arrays) > 1:
            # Calculate correlation between methods
            correlations = {}
            methods = list(method_p_values.keys())
            
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods[i+1:], i+1):
                    mask = ~np.eye(p_value_arrays[0].shape[0], dtype=bool)
                    p1_flat = method_p_values[method1][mask]
                    p2_flat = method_p_values[method2][mask]
                    
                    # Log transform for better correlation
                    log_p1 = -np.log10(p1_flat + 1e-10)
                    log_p2 = -np.log10(p2_flat + 1e-10)
                    
                    corr, _ = stats.pearsonr(log_p1, log_p2)
                    correlations[f"{method1}_vs_{method2}"] = corr
            
            stats_dict['method_correlations'] = correlations
            stats_dict['mean_correlation'] = np.mean(list(correlations.values()))
        
        # Method agreement
        n_methods = len(method_p_values)
        alpha = 0.05
        agreement_matrix = np.zeros(p_value_arrays[0].shape)
        
        for method_p_vals in p_value_arrays:
            agreement_matrix += (method_p_vals < alpha).astype(int)
        
        stats_dict['method_agreement'] = {
            'agreement_matrix': agreement_matrix,
            'mean_agreement': np.mean(agreement_matrix),
            'full_agreement_count': np.sum(agreement_matrix == n_methods),
            'no_agreement_count': np.sum(agreement_matrix == 0)
        }
        
        return stats_dict


class EnsembleSMTE(MultiScaleSMTE):
    """
    SMTE implementation with ensemble statistical framework.
    """
    
    def __init__(self,
                 use_ensemble_testing: bool = True,
                 surrogate_methods: List[str] = ['aaft', 'iaaft'],
                 n_surrogates_per_method: int = 50,
                 combination_method: str = 'fisher',
                 parallel_processing: bool = False,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.use_ensemble_testing = use_ensemble_testing
        self.surrogate_methods = surrogate_methods
        self.n_surrogates_per_method = n_surrogates_per_method
        self.combination_method = combination_method
        self.parallel_processing = parallel_processing
        
        # Initialize ensemble testing framework
        if self.use_ensemble_testing:
            self.ensemble_tester = EnsembleStatisticalTesting(
                surrogate_methods=surrogate_methods,
                n_surrogates_per_method=n_surrogates_per_method,
                combination_method=combination_method,
                random_state=getattr(self, 'random_state', 42)
            )
        
        # Store ensemble results
        self.ensemble_results = None
    
    def compute_ensemble_connectivity(self,
                                    data: np.ndarray,
                                    roi_labels: List[str],
                                    ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute connectivity with ensemble statistical testing.
        """
        
        print("Computing ensemble SMTE connectivity...")
        
        if not self.use_ensemble_testing:
            # Fall back to multi-scale analysis
            return self.compute_multiscale_connectivity(data, roi_labels, ground_truth)
        
        # First compute multi-scale connectivity to get base connectivity matrix
        multiscale_results = self.compute_multiscale_connectivity(data, roi_labels, ground_truth)
        
        # Extract combined connectivity matrix for ensemble testing
        combined_connectivity = multiscale_results['combined_connectivity']
        
        # Perform ensemble statistical testing
        print("Performing ensemble statistical testing...")
        ensemble_p_values, ensemble_details = self.ensemble_tester.ensemble_statistical_test(
            combined_connectivity, data, self
        )
        
        # Apply FDR correction to ensemble p-values
        print("Applying FDR correction to ensemble p-values...")
        original_alpha = self.alpha
        ensemble_significance = self.fdr_correction(ensemble_p_values)
        
        # Apply physiological constraints if enabled
        if self.use_physiological_constraints:
            print("Applying physiological constraints to ensemble results...")
            
            # Get network assignments safely
            network_structure = getattr(self, 'network_structure', None)
            network_assignments = None
            if network_structure and isinstance(network_structure, dict):
                network_assignments = network_structure.get('network_assignments', None)
            
            # Use first scale's lag matrix as reference
            reference_lags = multiscale_results['individual_scale_results'][self.scales_to_analyze[0]]['lag_matrix']
            
            physio_mask, constraint_info = self.apply_physiological_filtering(
                combined_connectivity,
                reference_lags,
                roi_labels,
                network_assignments
            )
            
            # Final significance: ensemble + physiological
            final_significance = ensemble_significance & physio_mask
        else:
            physio_mask = np.ones_like(combined_connectivity, dtype=bool)
            final_significance = ensemble_significance
            constraint_info = {'constraints_applied': ['none']}
        
        # Update multiscale results with ensemble information
        ensemble_results = {
            **multiscale_results,  # Include all multiscale results
            'ensemble_p_values': ensemble_p_values,
            'ensemble_significance': ensemble_significance,
            'ensemble_details': ensemble_details,
            'final_significance_mask': final_significance,
            'n_ensemble_significant': np.sum(ensemble_significance),
            'n_final_significant': np.sum(final_significance),
            'ensemble_constraint_info': constraint_info
        }
        
        # Store results
        self.ensemble_results = ensemble_results
        
        print(f"Ensemble analysis complete:")
        print(f"  Multi-scale combined: {multiscale_results['n_combined_significant']} significant")
        print(f"  Ensemble testing: {np.sum(ensemble_significance)} significant")
        print(f"  Final (with constraints): {np.sum(final_significance)} significant")
        
        # Report ensemble statistics
        if 'ensemble_statistics' in ensemble_details:
            stats = ensemble_details['ensemble_statistics']
            if 'mean_correlation' in stats:
                print(f"  Method correlation: {stats['mean_correlation']:.3f}")
            if 'method_agreement' in stats:
                agreement = stats['method_agreement']['mean_agreement']
                print(f"  Method agreement: {agreement:.3f}")
        
        return ensemble_results
    
    def create_ensemble_visualizations(self,
                                     ensemble_results: Dict[str, Any],
                                     roi_labels: List[str],
                                     save_prefix: str = 'ensemble_smte'):
        """Create comprehensive ensemble analysis visualizations."""
        
        # 1. Ensemble vs individual methods comparison
        individual_p_values = ensemble_results['ensemble_details']['individual_p_values']
        ensemble_p_values = ensemble_results['ensemble_p_values']
        
        n_methods = len(individual_p_values)
        fig1, axes1 = plt.subplots(2, n_methods + 1, figsize=(5*(n_methods+1), 10))
        
        # Individual method p-values
        for idx, (method, p_vals) in enumerate(individual_p_values.items()):
            # Raw p-values
            im1 = axes1[0, idx].imshow(-np.log10(p_vals + 1e-10), cmap='hot', aspect='auto')
            axes1[0, idx].set_title(f'{method}\n-log10(p-values)')
            axes1[0, idx].set_xlabel('Source ROI')
            axes1[0, idx].set_ylabel('Target ROI')
            plt.colorbar(im1, ax=axes1[0, idx], fraction=0.046, pad=0.04)
            
            # Significant connections
            alpha = 0.05
            significant = (p_vals < alpha).astype(int)
            im2 = axes1[1, idx].imshow(significant, cmap='Reds', aspect='auto')
            axes1[1, idx].set_title(f'{method}\nSignificant (α=0.05)')
            axes1[1, idx].set_xlabel('Source ROI')
            axes1[1, idx].set_ylabel('Target ROI')
            plt.colorbar(im2, ax=axes1[1, idx], fraction=0.046, pad=0.04)
        
        # Ensemble results
        im1_ens = axes1[0, -1].imshow(-np.log10(ensemble_p_values + 1e-10), cmap='hot', aspect='auto')
        axes1[0, -1].set_title('Ensemble\n-log10(p-values)')
        axes1[0, -1].set_xlabel('Source ROI')
        axes1[0, -1].set_ylabel('Target ROI')
        plt.colorbar(im1_ens, ax=axes1[0, -1], fraction=0.046, pad=0.04)
        
        ensemble_significant = (ensemble_p_values < 0.05).astype(int)
        im2_ens = axes1[1, -1].imshow(ensemble_significant, cmap='Reds', aspect='auto')
        axes1[1, -1].set_title('Ensemble\nSignificant (α=0.05)')
        axes1[1, -1].set_xlabel('Source ROI')
        axes1[1, -1].set_ylabel('Target ROI')
        plt.colorbar(im2_ens, ax=axes1[1, -1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_ensemble_methods.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Method agreement and correlation analysis
        if 'ensemble_statistics' in ensemble_results['ensemble_details']:
            stats = ensemble_results['ensemble_details']['ensemble_statistics']
            
            fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
            
            # Method agreement matrix
            if 'method_agreement' in stats:
                agreement_matrix = stats['method_agreement']['agreement_matrix']
                im1 = axes2[0].imshow(agreement_matrix, cmap='viridis', aspect='auto')
                axes2[0].set_title('Method Agreement\n(Number of methods agreeing)')
                axes2[0].set_xlabel('Source ROI')
                axes2[0].set_ylabel('Target ROI')
                plt.colorbar(im1, ax=axes2[0], fraction=0.046, pad=0.04)
            
            # Method correlations
            if 'method_correlations' in stats:
                correlations = stats['method_correlations']
                corr_values = list(correlations.values())
                corr_labels = list(correlations.keys())
                
                bars = axes2[1].bar(range(len(corr_values)), corr_values)
                axes2[1].set_title('Method Correlations\n(P-value correlations)')
                axes2[1].set_ylabel('Correlation')
                axes2[1].set_xticks(range(len(corr_labels)))
                axes2[1].set_xticklabels(corr_labels, rotation=45)
                
                # Add value labels
                for bar, val in zip(bars, corr_values):
                    height = bar.get_height()
                    axes2[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{val:.3f}', ha='center', va='bottom')
            
            # Statistical power comparison
            power_comparison = []
            method_names = []
            
            for method, p_vals in individual_p_values.items():
                n_significant = np.sum(p_vals < 0.05)
                power_comparison.append(n_significant)
                method_names.append(method)
            
            # Add ensemble
            n_ensemble_significant = np.sum(ensemble_p_values < 0.05)
            power_comparison.append(n_ensemble_significant)
            method_names.append('Ensemble')
            
            bars = axes2[2].bar(method_names, power_comparison)
            axes2[2].set_title('Statistical Power\n(Number of significant connections)')
            axes2[2].set_ylabel('Number Significant')
            axes2[2].tick_params(axis='x', rotation=45)
            
            # Highlight ensemble
            bars[-1].set_color('red')
            
            # Add value labels
            for bar, val in zip(bars, power_comparison):
                height = bar.get_height()
                axes2[2].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{val}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f'{save_prefix}_ensemble_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_ensemble_report(self,
                               ensemble_results: Dict[str, Any],
                               roi_labels: List[str]) -> str:
        """Generate comprehensive ensemble analysis report."""
        
        report = []
        report.append("# Ensemble SMTE Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Dataset summary
        report.append("## Dataset Summary")
        report.append("")
        report.append(f"**Regions analyzed:** {len(roi_labels)}")
        report.append(f"**Ensemble methods:** {len(self.surrogate_methods)}")
        report.append(f"**Surrogates per method:** {self.n_surrogates_per_method}")
        report.append(f"**P-value combination:** {self.combination_method}")
        report.append("")
        
        # Ensemble results summary
        report.append("## Ensemble Results Summary")
        report.append("")
        
        individual_p_values = ensemble_results['ensemble_details']['individual_p_values']
        ensemble_p_values = ensemble_results['ensemble_p_values']
        
        report.append("| Method | Significant Connections | Power Gain |")
        report.append("|--------|------------------------|------------|")
        
        baseline_significant = 0
        for method, p_vals in individual_p_values.items():
            n_significant = np.sum(p_vals < 0.05)
            if baseline_significant == 0:  # Use first method as baseline
                baseline_significant = n_significant
            
            power_gain = ((n_significant - baseline_significant) / baseline_significant * 100) if baseline_significant > 0 else 0
            report.append(f"| {method} | {n_significant} | {power_gain:+.1f}% |")
        
        # Ensemble results
        n_ensemble_significant = np.sum(ensemble_p_values < 0.05)
        ensemble_power_gain = ((n_ensemble_significant - baseline_significant) / baseline_significant * 100) if baseline_significant > 0 else 0
        report.append(f"| **Ensemble** | **{n_ensemble_significant}** | **{ensemble_power_gain:+.1f}%** |")
        
        report.append("")
        
        # Ensemble statistics
        if 'ensemble_statistics' in ensemble_results['ensemble_details']:
            stats = ensemble_results['ensemble_details']['ensemble_statistics']
            
            report.append("## Ensemble Statistics")
            report.append("")
            
            if 'mean_correlation' in stats:
                report.append(f"**Mean method correlation:** {stats['mean_correlation']:.3f}")
            
            if 'method_agreement' in stats:
                agreement_stats = stats['method_agreement']
                total_connections = len(roi_labels) * (len(roi_labels) - 1)
                
                report.append(f"**Mean agreement score:** {agreement_stats['mean_agreement']:.3f}")
                report.append(f"**Full agreement:** {agreement_stats['full_agreement_count']} connections")
                report.append(f"**No agreement:** {agreement_stats['no_agreement_count']} connections")
            
            report.append("")
        
        # Multi-scale integration
        if 'scale_statistics' in ensemble_results:
            report.append("## Multi-Scale Integration")
            report.append("")
            
            scale_stats = ensemble_results['scale_statistics']
            for scale_name, stats in scale_stats.items():
                report.append(f"**{scale_name.title()} Scale:**")
                report.append(f"- Significant connections: {stats['n_significant']}")
                report.append(f"- Mean connectivity: {stats['mean_connectivity']:.4f}")
                report.append("")
        
        # Final results
        report.append("## Final Results")
        report.append("")
        
        n_multiscale = ensemble_results.get('n_combined_significant', 0)
        n_ensemble = ensemble_results.get('n_ensemble_significant', 0)
        n_final = ensemble_results.get('n_final_significant', 0)
        
        report.append(f"**Multi-scale significant:** {n_multiscale}")
        report.append(f"**Ensemble significant:** {n_ensemble}")
        report.append(f"**Final significant (with constraints):** {n_final}")
        
        improvement_over_multiscale = ((n_ensemble - n_multiscale) / n_multiscale * 100) if n_multiscale > 0 else 0
        report.append(f"**Ensemble improvement:** {improvement_over_multiscale:+.1f}% over multi-scale alone")
        
        return "\n".join(report)


def test_ensemble_smte():
    """Test the ensemble SMTE implementation."""
    
    print("Testing Ensemble SMTE Implementation")
    print("=" * 60)
    
    # Generate test data with multiple connectivity patterns
    np.random.seed(42)
    n_regions = 10
    n_timepoints = 150
    TR = 2.0
    
    # Create ROI labels
    roi_labels = [f"Region_{i+1}" for i in range(n_regions)]
    
    # Generate realistic data with multiple types of coupling
    data = []
    for i in range(n_regions):
        # Multi-frequency signal
        t = np.arange(n_timepoints) * TR
        slow = 0.6 * np.sin(2 * np.pi * 0.02 * t)
        fast = 0.4 * np.sin(2 * np.pi * 0.1 * t)
        noise = 0.5 * np.random.randn(n_timepoints)
        signal = slow + fast + noise
        data.append(signal)
    
    data = np.array(data)
    
    # Add various types of connectivity
    # Linear connectivity
    data[1, 2:] += 0.5 * data[0, :-2]  # 0 -> 1 with lag 2
    
    # Nonlinear connectivity
    data[3, 1:] += 0.3 * np.tanh(data[2, :-1])  # 2 -> 3 nonlinear
    
    # Delayed connectivity
    data[5, 4:] += 0.4 * data[4, :-4]  # 4 -> 5 with lag 4
    
    # Standardize data
    scaler = StandardScaler()
    data = scaler.fit_transform(data.T).T
    
    # Test standard multi-scale SMTE (baseline)
    print("\n1. Testing Standard Multi-Scale SMTE (Baseline)")
    print("-" * 55)
    
    standard_smte = EnsembleSMTE(
        use_ensemble_testing=False,  # Disable ensemble
        use_multiscale_analysis=True,
        scales_to_analyze=['fast', 'intermediate'],
        adaptive_mode='heuristic',
        n_permutations=50,  # Reduced for testing
        random_state=42
    )
    
    standard_results = standard_smte.compute_ensemble_connectivity(data, roi_labels)
    print(f"Standard multi-scale: {standard_results['n_combined_significant']} significant connections")
    
    # Test ensemble SMTE
    print("\n2. Testing Ensemble SMTE")
    print("-" * 30)
    
    ensemble_smte = EnsembleSMTE(
        use_ensemble_testing=True,   # Enable ensemble
        surrogate_methods=['aaft', 'phase_randomization'],  # Reduced for testing
        n_surrogates_per_method=30,  # Reduced for testing speed
        combination_method='fisher',
        use_multiscale_analysis=True,
        scales_to_analyze=['fast', 'intermediate'],
        adaptive_mode='heuristic',
        n_permutations=50,
        random_state=42
    )
    
    ensemble_results = ensemble_smte.compute_ensemble_connectivity(data, roi_labels)
    
    # Generate report
    report = ensemble_smte.generate_ensemble_report(ensemble_results, roi_labels)
    print("\n" + "=" * 60)
    print("ENSEMBLE ANALYSIS REPORT")
    print("=" * 60)
    print(report)
    
    # Create visualizations
    ensemble_smte.create_ensemble_visualizations(ensemble_results, roi_labels)
    
    return ensemble_results


if __name__ == "__main__":
    results = test_ensemble_smte()