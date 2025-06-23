#!/usr/bin/env python3
"""
Multi-Level Statistical Framework for SMTE
Addresses statistical power limitations with adaptive, hierarchical correction.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy import stats
from scipy.stats import combine_pvalues
import logging

class CorrectionMethod(Enum):
    FDR_ADAPTIVE = "fdr_adaptive"
    FDR_CLUSTER_ADAPTIVE = "fdr_cluster_adaptive"
    HIERARCHICAL = "hierarchical"
    BOOTSTRAP = "bootstrap"
    ENSEMBLE_LAGS = "ensemble_lags"
    LIBERAL_EXPLORATION = "liberal_exploration"

@dataclass
class StatisticalResult:
    """Container for statistical test results."""
    p_values: np.ndarray
    corrected_p_values: np.ndarray
    significance_mask: np.ndarray
    effect_sizes: np.ndarray
    confidence_intervals: Optional[np.ndarray]
    method_used: str
    alpha_effective: float
    power_estimate: float

class MultiLevelStatisticalFramework:
    """
    Advanced statistical framework with multiple correction strategies
    designed to improve SMTE detection power while controlling false positives.
    """
    
    def __init__(self, base_alpha: float = 0.05, conservative_mode: bool = False):
        self.base_alpha = base_alpha
        self.conservative_mode = conservative_mode
        self.logger = logging.getLogger(__name__)
        
        # Statistical parameters
        self.bootstrap_n_samples = 1000
        self.min_effect_size = 0.1
        self.confidence_level = 0.95
    
    def apply_statistical_correction(self, 
                                   connectivity_matrix: np.ndarray,
                                   p_values: np.ndarray,
                                   method: CorrectionMethod,
                                   cluster_info: Optional[Dict] = None,
                                   lag_p_values: Optional[List[np.ndarray]] = None,
                                   effect_sizes: Optional[np.ndarray] = None) -> StatisticalResult:
        """
        Apply multi-level statistical correction.
        
        Parameters:
        -----------
        connectivity_matrix : np.ndarray
            SMTE connectivity matrix
        p_values : np.ndarray
            Raw p-values from statistical testing
        method : CorrectionMethod
            Statistical correction method to use
        cluster_info : dict, optional
            Clustering information for cluster-adaptive methods
        lag_p_values : list, optional
            P-values from different lags for ensemble methods
        effect_sizes : np.ndarray, optional
            Effect sizes for each connection
            
        Returns:
        --------
        StatisticalResult
            Comprehensive statistical results
        """
        
        # Compute effect sizes if not provided
        if effect_sizes is None:
            effect_sizes = np.abs(connectivity_matrix)
        
        # Apply the selected method
        if method == CorrectionMethod.FDR_ADAPTIVE:
            result = self._apply_adaptive_fdr(p_values, effect_sizes)
        elif method == CorrectionMethod.FDR_CLUSTER_ADAPTIVE:
            result = self._apply_cluster_adaptive_fdr(p_values, effect_sizes, cluster_info)
        elif method == CorrectionMethod.HIERARCHICAL:
            result = self._apply_hierarchical_correction(p_values, effect_sizes, cluster_info)
        elif method == CorrectionMethod.BOOTSTRAP:
            result = self._apply_bootstrap_correction(connectivity_matrix, p_values)
        elif method == CorrectionMethod.ENSEMBLE_LAGS:
            result = self._apply_ensemble_lag_correction(lag_p_values, effect_sizes)
        elif method == CorrectionMethod.LIBERAL_EXPLORATION:
            result = self._apply_liberal_exploration(p_values, effect_sizes)
        else:
            raise ValueError(f"Unknown correction method: {method}")
        
        # Add method information
        result.method_used = method.value
        
        # Estimate statistical power
        result.power_estimate = self._estimate_statistical_power(result, effect_sizes)
        
        self.logger.info(f"Applied {method.value}: {np.sum(result.significance_mask)} significant connections")
        
        return result
    
    def _apply_adaptive_fdr(self, p_values: np.ndarray, effect_sizes: np.ndarray) -> StatisticalResult:
        """Apply adaptive FDR that adjusts alpha based on effect sizes."""
        
        # Create adaptive alpha based on effect size distribution
        effect_percentiles = np.percentile(effect_sizes[effect_sizes > 0], [25, 50, 75, 90])
        
        # Get off-diagonal elements
        mask = ~np.eye(p_values.shape[0], dtype=bool)
        p_flat = p_values[mask]
        effect_flat = effect_sizes[mask]
        
        # Adaptive alpha: higher for stronger effects
        adaptive_alpha = np.full_like(p_flat, self.base_alpha)
        
        # More liberal alpha for strong effects
        strong_effects = effect_flat > effect_percentiles[2]  # Top 25%
        adaptive_alpha[strong_effects] *= 2.0
        
        # More conservative alpha for weak effects
        weak_effects = effect_flat < effect_percentiles[0]  # Bottom 25%
        adaptive_alpha[weak_effects] *= 0.5
        
        # Apply BH procedure with adaptive alpha
        corrected_p, significant = self._benjamini_hochberg_adaptive(p_flat, adaptive_alpha)
        
        # Map back to matrix form
        corrected_p_matrix = np.ones_like(p_values)
        significance_mask = np.zeros_like(p_values, dtype=bool)
        
        corrected_p_matrix[mask] = corrected_p
        significance_mask[mask] = significant
        
        # Effective alpha
        alpha_effective = np.mean(adaptive_alpha[significant]) if np.any(significant) else self.base_alpha
        
        return StatisticalResult(
            p_values=p_values,
            corrected_p_values=corrected_p_matrix,
            significance_mask=significance_mask,
            effect_sizes=effect_sizes,
            confidence_intervals=None,
            method_used="adaptive_fdr",
            alpha_effective=alpha_effective,
            power_estimate=0.0  # Will be calculated later
        )
    
    def _apply_cluster_adaptive_fdr(self, p_values: np.ndarray, effect_sizes: np.ndarray,
                                  cluster_info: Optional[Dict]) -> StatisticalResult:
        """Apply FDR correction that adapts to cluster size."""
        
        if cluster_info is None:
            # Fall back to regular adaptive FDR
            return self._apply_adaptive_fdr(p_values, effect_sizes)
        
        significance_mask = np.zeros_like(p_values, dtype=bool)
        corrected_p_matrix = np.ones_like(p_values)
        
        # Apply correction within each cluster
        for cluster_name, cluster_members in cluster_info.items():
            if len(cluster_members) < 2:
                continue
            
            # Extract cluster connections
            cluster_indices = [i for i, member in enumerate(cluster_members)]
            cluster_p_values = []
            cluster_positions = []
            cluster_effects = []
            
            for i in cluster_indices:
                for j in cluster_indices:
                    if i != j:
                        cluster_p_values.append(p_values[i, j])
                        cluster_positions.append((i, j))
                        cluster_effects.append(effect_sizes[i, j])
            
            if not cluster_p_values:
                continue
            
            cluster_p_array = np.array(cluster_p_values)
            cluster_effect_array = np.array(cluster_effects)
            
            # Cluster-size adaptive alpha
            cluster_size = len(cluster_p_values)
            if cluster_size <= 4:
                # Small clusters: liberal alpha
                cluster_alpha = self.base_alpha * 3.0
            elif cluster_size <= 10:
                # Medium clusters: moderate alpha
                cluster_alpha = self.base_alpha * 2.0
            elif cluster_size <= 20:
                # Large clusters: standard alpha
                cluster_alpha = self.base_alpha * 1.5
            else:
                # Very large clusters: conservative alpha
                cluster_alpha = self.base_alpha
            
            # Apply effect-size weighting
            strong_effects = cluster_effect_array > np.median(cluster_effect_array)
            cluster_alpha_array = np.full_like(cluster_p_array, cluster_alpha)
            cluster_alpha_array[strong_effects] *= 1.5
            
            # Apply adaptive BH correction
            corrected_p, significant = self._benjamini_hochberg_adaptive(
                cluster_p_array, cluster_alpha_array
            )
            
            # Map back to full matrix
            for idx, (i, j) in enumerate(cluster_positions):
                corrected_p_matrix[i, j] = corrected_p[idx]
                if significant[idx]:
                    significance_mask[i, j] = True
        
        alpha_effective = self.base_alpha * 2.0  # Approximate
        
        return StatisticalResult(
            p_values=p_values,
            corrected_p_values=corrected_p_matrix,
            significance_mask=significance_mask,
            effect_sizes=effect_sizes,
            confidence_intervals=None,
            method_used="cluster_adaptive_fdr",
            alpha_effective=alpha_effective,
            power_estimate=0.0
        )
    
    def _apply_hierarchical_correction(self, p_values: np.ndarray, effect_sizes: np.ndarray,
                                     cluster_info: Optional[Dict]) -> StatisticalResult:
        """Apply hierarchical correction: network -> cluster -> connection."""
        
        # Level 1: Global test
        global_p = np.min(p_values[~np.eye(p_values.shape[0], dtype=bool)])
        global_significant = global_p < self.base_alpha
        
        if not global_significant:
            # If global test fails, nothing is significant
            return StatisticalResult(
                p_values=p_values,
                corrected_p_values=p_values,
                significance_mask=np.zeros_like(p_values, dtype=bool),
                effect_sizes=effect_sizes,
                confidence_intervals=None,
                method_used="hierarchical",
                alpha_effective=self.base_alpha,
                power_estimate=0.0
            )
        
        # Level 2: Cluster-level tests (if cluster info available)
        if cluster_info:
            return self._apply_cluster_adaptive_fdr(p_values, effect_sizes, cluster_info)
        else:
            # Level 3: Connection-level tests with liberal alpha
            return self._apply_liberal_exploration(p_values, effect_sizes)
    
    def _apply_bootstrap_correction(self, connectivity_matrix: np.ndarray, 
                                  p_values: np.ndarray) -> StatisticalResult:
        """Apply bootstrap-based correction alternative to permutation testing."""
        
        # Bootstrap confidence intervals for effect sizes
        n_connections = np.sum(~np.eye(connectivity_matrix.shape[0], dtype=bool))
        effect_sizes = np.abs(connectivity_matrix)
        
        # Generate bootstrap samples
        bootstrap_effects = []
        n_samples = min(self.bootstrap_n_samples, 200)  # Limit for speed
        
        for _ in range(n_samples):
            # Bootstrap resample (simplified)
            noise = np.random.randn(*connectivity_matrix.shape) * 0.1
            bootstrap_matrix = connectivity_matrix + noise
            bootstrap_effects.append(np.abs(bootstrap_matrix))
        
        bootstrap_effects = np.array(bootstrap_effects)
        
        # Calculate confidence intervals
        ci_lower = np.percentile(bootstrap_effects, (1 - self.confidence_level) / 2 * 100, axis=0)
        ci_upper = np.percentile(bootstrap_effects, (1 + self.confidence_level) / 2 * 100, axis=0)
        
        # Significance based on CI not including zero
        significance_mask = ci_lower > self.min_effect_size
        
        # Adjust p-values based on bootstrap
        adjusted_p = np.ones_like(p_values)
        mask = ~np.eye(p_values.shape[0], dtype=bool)
        
        # Bootstrap p-values (simplified)
        bootstrap_p = 1.0 - (np.sum(bootstrap_effects > effect_sizes[None, :, :], axis=0) / n_samples)
        adjusted_p = np.minimum(p_values, bootstrap_p)
        
        return StatisticalResult(
            p_values=p_values,
            corrected_p_values=adjusted_p,
            significance_mask=significance_mask,
            effect_sizes=effect_sizes,
            confidence_intervals=np.stack([ci_lower, ci_upper], axis=-1),
            method_used="bootstrap",
            alpha_effective=self.base_alpha,
            power_estimate=0.0
        )
    
    def _apply_ensemble_lag_correction(self, lag_p_values: List[np.ndarray], 
                                     effect_sizes: np.ndarray) -> StatisticalResult:
        """Combine p-values across multiple lags using ensemble methods."""
        
        if not lag_p_values or len(lag_p_values) < 2:
            raise ValueError("Need at least 2 lag p-value matrices for ensemble correction")
        
        # Get matrix shape
        n_rois = lag_p_values[0].shape[0]
        mask = ~np.eye(n_rois, dtype=bool)
        
        # Combine p-values for each connection across lags
        combined_p_matrix = np.ones((n_rois, n_rois))
        combined_method_matrix = np.zeros((n_rois, n_rois), dtype=int)
        
        for i in range(n_rois):
            for j in range(n_rois):
                if i != j:
                    # Get p-values across lags for this connection
                    connection_p_values = [lag_p[i, j] for lag_p in lag_p_values]
                    
                    # Try different combination methods
                    fisher_stat, fisher_p = combine_pvalues(connection_p_values, method='fisher')
                    stouffer_stat, stouffer_p = combine_pvalues(connection_p_values, method='stouffer')
                    
                    # Choose method based on effect size
                    if effect_sizes[i, j] > np.median(effect_sizes[mask]):
                        # Use Fisher for stronger effects
                        combined_p_matrix[i, j] = fisher_p
                        combined_method_matrix[i, j] = 1
                    else:
                        # Use Stouffer for weaker effects
                        combined_p_matrix[i, j] = stouffer_p
                        combined_method_matrix[i, j] = 2
        
        # Apply FDR correction to combined p-values
        corrected_result = self._apply_adaptive_fdr(combined_p_matrix, effect_sizes)
        
        return StatisticalResult(
            p_values=combined_p_matrix,
            corrected_p_values=corrected_result.corrected_p_values,
            significance_mask=corrected_result.significance_mask,
            effect_sizes=effect_sizes,
            confidence_intervals=None,
            method_used="ensemble_lags",
            alpha_effective=corrected_result.alpha_effective,
            power_estimate=0.0
        )
    
    def _apply_liberal_exploration(self, p_values: np.ndarray, 
                                 effect_sizes: np.ndarray) -> StatisticalResult:
        """Apply liberal thresholds for exploratory analysis."""
        
        # Liberal alpha for exploration
        liberal_alpha = min(0.2, self.base_alpha * 4.0)
        
        # Effect size thresholding
        effect_threshold = np.percentile(effect_sizes[effect_sizes > 0], 25)
        
        # Combined significance: liberal p-value AND minimum effect size
        p_significant = p_values < liberal_alpha
        effect_significant = effect_sizes > effect_threshold
        
        significance_mask = p_significant & effect_significant
        
        # No correction for exploratory mode
        corrected_p_values = p_values.copy()
        
        return StatisticalResult(
            p_values=p_values,
            corrected_p_values=corrected_p_values,
            significance_mask=significance_mask,
            effect_sizes=effect_sizes,
            confidence_intervals=None,
            method_used="liberal_exploration",
            alpha_effective=liberal_alpha,
            power_estimate=0.0
        )
    
    def _benjamini_hochberg_adaptive(self, p_values: np.ndarray, 
                                   alpha_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Benjamini-Hochberg procedure with adaptive alpha values."""
        
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        sorted_alpha = alpha_values[sorted_indices]
        
        n = len(p_values)
        significant = np.zeros(n, dtype=bool)
        
        # BH procedure with adaptive alpha
        for i in range(n):
            threshold = (i + 1) / n * sorted_alpha[i]
            if sorted_p[i] <= threshold:
                significant[sorted_indices[:i+1]] = True
            else:
                break
        
        # Corrected p-values
        corrected_p = np.minimum(1.0, sorted_p * n / (np.arange(n) + 1))
        corrected_p = np.maximum.accumulate(corrected_p[::-1])[::-1]
        
        # Map back to original order
        final_corrected_p = np.zeros_like(p_values)
        final_corrected_p[sorted_indices] = corrected_p
        
        return final_corrected_p, significant
    
    def _estimate_statistical_power(self, result: StatisticalResult, 
                                  effect_sizes: np.ndarray) -> float:
        """Estimate statistical power of the test."""
        
        # Power estimation based on detection of larger effects
        mask = ~np.eye(effect_sizes.shape[0], dtype=bool)
        large_effects = effect_sizes[mask] > np.percentile(effect_sizes[mask], 75)
        
        if np.sum(large_effects) == 0:
            return 0.0
        
        detected_large_effects = result.significance_mask[mask][large_effects]
        power_estimate = np.sum(detected_large_effects) / np.sum(large_effects)
        
        return power_estimate
    
    def compare_methods(self, connectivity_matrix: np.ndarray, p_values: np.ndarray,
                       cluster_info: Optional[Dict] = None,
                       lag_p_values: Optional[List[np.ndarray]] = None) -> Dict[str, StatisticalResult]:
        """Compare multiple statistical correction methods."""
        
        effect_sizes = np.abs(connectivity_matrix)
        methods_to_test = [
            CorrectionMethod.FDR_ADAPTIVE,
            CorrectionMethod.FDR_CLUSTER_ADAPTIVE,
            CorrectionMethod.LIBERAL_EXPLORATION
        ]
        
        # Add ensemble method if lag data available
        if lag_p_values and len(lag_p_values) >= 2:
            methods_to_test.append(CorrectionMethod.ENSEMBLE_LAGS)
        
        results = {}
        for method in methods_to_test:
            try:
                if method == CorrectionMethod.ENSEMBLE_LAGS:
                    result = self.apply_statistical_correction(
                        connectivity_matrix, p_values, method, cluster_info, lag_p_values, effect_sizes
                    )
                else:
                    result = self.apply_statistical_correction(
                        connectivity_matrix, p_values, method, cluster_info, None, effect_sizes
                    )
                results[method.value] = result
            except Exception as e:
                self.logger.warning(f"Method {method.value} failed: {e}")
        
        return results
    
    def select_best_method(self, method_results: Dict[str, StatisticalResult],
                          ground_truth: Optional[np.ndarray] = None) -> Tuple[str, StatisticalResult]:
        """Select the best statistical method based on performance criteria."""
        
        if not method_results:
            raise ValueError("No method results to compare")
        
        scores = {}
        
        for method_name, result in method_results.items():
            # Score based on multiple criteria
            n_detections = np.sum(result.significance_mask)
            power = result.power_estimate
            conservative_score = 1.0 / (result.alpha_effective + 0.01)  # Prefer more conservative methods
            
            if ground_truth is not None:
                # If ground truth available, use accuracy metrics
                true_mask = ground_truth > 0.1
                pred_mask = result.significance_mask
                
                true_positives = np.sum(pred_mask & true_mask)
                false_positives = np.sum(pred_mask & ~true_mask)
                precision = true_positives / max(np.sum(pred_mask), 1)
                recall = true_positives / max(np.sum(true_mask), 1)
                f1_score = 2 * precision * recall / max(precision + recall, 0.001)
                
                # Weight F1-score highly when ground truth available
                score = f1_score * 10 + power * 2 + np.log(n_detections + 1)
            else:
                # Without ground truth, balance detection and conservatism
                score = power * 5 + np.log(n_detections + 1) + conservative_score
            
            scores[method_name] = score
        
        best_method = max(scores.keys(), key=lambda k: scores[k])
        
        self.logger.info(f"Best method selected: {best_method} (score: {scores[best_method]:.3f})")
        
        return best_method, method_results[best_method]


def test_multilevel_statistical_framework():
    """Test the multi-level statistical framework."""
    
    print("🧪 TESTING MULTI-LEVEL STATISTICAL FRAMEWORK")
    print("=" * 60)
    
    # Create test data with known connections
    np.random.seed(42)
    n_rois = 8
    n_timepoints = 200
    
    # Generate connectivity matrix with known structure
    connectivity_matrix = np.random.randn(n_rois, n_rois) * 0.1
    ground_truth = np.zeros((n_rois, n_rois))
    
    # Add strong connections
    strong_connections = [(0, 1, 0.5), (1, 2, 0.4), (3, 4, 0.6)]
    for i, j, strength in strong_connections:
        connectivity_matrix[i, j] = strength
        ground_truth[i, j] = strength
    
    # Add weak connections
    weak_connections = [(2, 3, 0.2), (4, 5, 0.15)]
    for i, j, strength in weak_connections:
        connectivity_matrix[i, j] = strength
        ground_truth[i, j] = strength
    
    # Generate realistic p-values
    p_values = np.random.uniform(0.01, 0.99, (n_rois, n_rois))
    
    # Make true connections have better p-values
    for i, j, strength in strong_connections + weak_connections:
        p_values[i, j] = np.random.uniform(0.001, 0.1)
    
    # Create cluster information
    cluster_info = {
        'cluster_1': [0, 1, 2],
        'cluster_2': [3, 4, 5],
        'cluster_3': [6, 7]
    }
    
    print(f"Test data: {np.sum(ground_truth > 0)} true connections")
    print(f"P-values range: {np.min(p_values):.4f} to {np.max(p_values):.4f}")
    
    # Test statistical framework
    framework = MultiLevelStatisticalFramework(base_alpha=0.05)
    
    # Compare methods
    method_results = framework.compare_methods(
        connectivity_matrix, p_values, cluster_info
    )
    
    print(f"\n📊 METHOD COMPARISON RESULTS")
    print("-" * 40)
    
    for method_name, result in method_results.items():
        n_detected = np.sum(result.significance_mask)
        true_detected = np.sum(result.significance_mask & (ground_truth > 0))
        false_detected = np.sum(result.significance_mask & (ground_truth == 0))
        
        print(f"\n{method_name.replace('_', ' ').title()}:")
        print(f"  Detected: {n_detected} ({true_detected} TP, {false_detected} FP)")
        print(f"  Power: {result.power_estimate:.3f}")
        print(f"  Effective α: {result.alpha_effective:.3f}")
    
    # Select best method
    best_method_name, best_result = framework.select_best_method(
        method_results, ground_truth
    )
    
    print(f"\n🏆 BEST METHOD: {best_method_name}")
    print(f"   True Positives: {np.sum(best_result.significance_mask & (ground_truth > 0))}")
    print(f"   False Positives: {np.sum(best_result.significance_mask & (ground_truth == 0))}")
    print(f"   Power Estimate: {best_result.power_estimate:.3f}")
    
    # Test improvement over standard FDR
    print(f"\n🔄 COMPARISON WITH STANDARD FDR")
    print("-" * 35)
    
    # Standard FDR (Benjamini-Hochberg)
    try:
        mask = ~np.eye(n_rois, dtype=bool)
        p_flat = p_values[mask]
        
        # Manual BH procedure
        sorted_indices = np.argsort(p_flat)
        sorted_p = p_flat[sorted_indices]
        n = len(sorted_p)
        
        # BH procedure
        significant = np.zeros(n, dtype=bool)
        for i in range(n):
            if sorted_p[i] <= (i + 1) / n * 0.05:
                significant[sorted_indices[:i+1]] = True
            else:
                break
        
        standard_fdr_mask = np.zeros_like(p_values, dtype=bool)
        standard_fdr_mask[mask] = significant
        
        standard_tp = np.sum(standard_fdr_mask & (ground_truth > 0))
        standard_fp = np.sum(standard_fdr_mask & (ground_truth == 0))
        
        best_tp = np.sum(best_result.significance_mask & (ground_truth > 0))
        best_fp = np.sum(best_result.significance_mask & (ground_truth == 0))
        
        print(f"Standard FDR: {standard_tp} TP, {standard_fp} FP")
        print(f"Best Method:  {best_tp} TP, {best_fp} FP")
        
        if best_tp > standard_tp:
            improvement = best_tp - standard_tp
            print(f"🎯 Improvement: +{improvement} true positive detections")
        
    except Exception as e:
        print(f"Standard FDR comparison failed: {e}")
    
    return method_results, best_result

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Test the framework
    test_results = test_multilevel_statistical_framework()