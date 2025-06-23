#!/usr/bin/env python3
"""
Enhanced SMTE System for fMRI Applications
Integrated system combining adaptive temporal resolution and multi-level statistics.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging
import warnings

from adaptive_temporal_system import AdaptiveSMTEConnectivity, TemporalParameters
from multilevel_statistical_framework import MultiLevelStatisticalFramework, CorrectionMethod, StatisticalResult

@dataclass
class EnhancedSMTEResult:
    """Comprehensive result from enhanced SMTE analysis."""
    connectivity_matrix: np.ndarray
    statistical_result: StatisticalResult
    temporal_params: TemporalParameters
    optimization_summary: Dict[str, Any]
    performance_metrics: Dict[str, float]
    recommendations: List[str]

class EnhancedSMTESystem:
    """
    Complete enhanced SMTE system for fMRI applications.
    
    Combines:
    1. Adaptive temporal resolution optimization
    2. Multi-level statistical framework
    3. Intelligent parameter selection
    4. Performance optimization
    """
    
    def __init__(self, 
                 tr: float,
                 conservative_mode: bool = False,
                 auto_optimize: bool = True,
                 base_alpha: float = 0.05):
        """
        Initialize enhanced SMTE system.
        
        Parameters:
        -----------
        tr : float
            Repetition time in seconds
        conservative_mode : bool
            Whether to use conservative statistical thresholds
        auto_optimize : bool
            Whether to automatically optimize parameters
        base_alpha : float
            Base significance level
        """
        
        self.tr = tr
        self.conservative_mode = conservative_mode
        self.auto_optimize = auto_optimize
        self.base_alpha = base_alpha
        
        # Initialize components
        self.adaptive_smte = None
        self.statistical_framework = MultiLevelStatisticalFramework(
            base_alpha=base_alpha, 
            conservative_mode=conservative_mode
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Analysis history
        self.analysis_history = []
        
    def analyze_connectivity(self, 
                           data: np.ndarray,
                           roi_labels: Optional[List[str]] = None,
                           cluster_info: Optional[Dict] = None,
                           ground_truth: Optional[np.ndarray] = None,
                           method_preference: Optional[CorrectionMethod] = None) -> EnhancedSMTEResult:
        """
        Complete enhanced SMTE connectivity analysis.
        
        Parameters:
        -----------
        data : np.ndarray
            fMRI data (n_rois x n_timepoints)
        roi_labels : list, optional
            Region labels
        cluster_info : dict, optional
            Clustering information for adaptive correction
        ground_truth : np.ndarray, optional
            Ground truth for validation
        method_preference : CorrectionMethod, optional
            Preferred statistical method
            
        Returns:
        --------
        EnhancedSMTEResult
            Comprehensive analysis results
        """
        
        n_rois, n_timepoints = data.shape
        
        if roi_labels is None:
            roi_labels = [f"ROI_{i:02d}" for i in range(n_rois)]
        
        self.logger.info(f"Starting enhanced SMTE analysis: {n_rois} ROIs, {n_timepoints} timepoints")
        
        # Step 1: Initialize adaptive SMTE with temporal optimization
        if self.adaptive_smte is None:
            self.adaptive_smte = AdaptiveSMTEConnectivity(
                tr=self.tr,
                n_timepoints=n_timepoints,
                n_permutations=100 if n_timepoints > 200 else 50
            )
        
        # Step 2: Auto-configure parameters
        if self.auto_optimize:
            config_results = self.adaptive_smte.auto_configure(data)
            temporal_params = config_results['temporal_params']
            optimization_summary = {
                'temporal_optimization': config_results,
                'auto_configured': True
            }
        else:
            # Use default parameters
            temporal_params = TemporalParameters(
                tr=self.tr, max_lag=3, n_symbols=2, ordinal_order=2,
                hemodynamic_delay=3.0, optimal_lags=[1, 2, 3], confidence_score=0.5
            )
            optimization_summary = {'auto_configured': False}
        
        # Step 3: Compute SMTE connectivity
        self.logger.info("Computing SMTE connectivity matrix...")
        connectivity_matrix, p_values = self._compute_smte_connectivity(data)
        
        # Step 4: Apply enhanced statistical correction
        self.logger.info("Applying multi-level statistical correction...")
        if method_preference:
            statistical_result = self.statistical_framework.apply_statistical_correction(
                connectivity_matrix, p_values, method_preference, cluster_info
            )
        else:
            # Compare methods and select best
            method_results = self.statistical_framework.compare_methods(
                connectivity_matrix, p_values, cluster_info
            )
            
            # Select best method
            best_method_name, statistical_result = self.statistical_framework.select_best_method(
                method_results, ground_truth
            )
            
            optimization_summary['method_comparison'] = {
                name: {
                    'n_detections': np.sum(result.significance_mask),
                    'power': result.power_estimate,
                    'alpha_effective': result.alpha_effective
                }
                for name, result in method_results.items()
            }
            optimization_summary['selected_method'] = best_method_name
        
        # Step 5: Compute performance metrics
        performance_metrics = self._compute_performance_metrics(
            statistical_result, ground_truth, temporal_params
        )
        
        # Step 6: Generate recommendations
        recommendations = self._generate_recommendations(
            statistical_result, temporal_params, performance_metrics, data.shape
        )
        
        # Create comprehensive result
        result = EnhancedSMTEResult(
            connectivity_matrix=connectivity_matrix,
            statistical_result=statistical_result,
            temporal_params=temporal_params,
            optimization_summary=optimization_summary,
            performance_metrics=performance_metrics,
            recommendations=recommendations
        )
        
        # Store in history
        self.analysis_history.append({
            'timestamp': np.datetime64('now'),
            'data_shape': data.shape,
            'method_used': statistical_result.method_used,
            'n_detections': np.sum(statistical_result.significance_mask),
            'performance': performance_metrics
        })
        
        self.logger.info(f"Analysis complete: {np.sum(statistical_result.significance_mask)} connections detected")
        
        return result
    
    def _compute_smte_connectivity(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute SMTE connectivity matrix and p-values."""
        
        # Set up SMTE computation
        self.adaptive_smte.base_smte.fmri_data = data
        self.adaptive_smte.base_smte.mask = np.ones(data.shape[0], dtype=bool)
        
        # Symbolize time series
        symbolic_data = self.adaptive_smte.base_smte.symbolize_timeseries(data)
        self.adaptive_smte.base_smte.symbolic_data = symbolic_data
        
        # Compute connectivity
        connectivity_matrix, _ = self.adaptive_smte.base_smte.compute_voxel_connectivity_matrix()
        
        # Statistical testing
        p_values = self.adaptive_smte.base_smte.statistical_testing(connectivity_matrix)
        
        return connectivity_matrix, p_values
    
    def _compute_performance_metrics(self, 
                                   statistical_result: StatisticalResult,
                                   ground_truth: Optional[np.ndarray],
                                   temporal_params: TemporalParameters) -> Dict[str, float]:
        """Compute comprehensive performance metrics."""
        
        metrics = {
            'n_significant': float(np.sum(statistical_result.significance_mask)),
            'power_estimate': statistical_result.power_estimate,
            'alpha_effective': statistical_result.alpha_effective,
            'temporal_confidence': temporal_params.confidence_score,
            'max_lag_seconds': temporal_params.max_lag * temporal_params.tr,
            'detection_rate': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'specificity': 0.0
        }
        
        if ground_truth is not None:
            # Validation metrics
            true_mask = ground_truth > 0.1
            pred_mask = statistical_result.significance_mask
            
            true_positives = np.sum(pred_mask & true_mask)
            false_positives = np.sum(pred_mask & ~true_mask)
            false_negatives = np.sum(~pred_mask & true_mask)
            true_negatives = np.sum(~pred_mask & ~true_mask)
            
            metrics['true_positives'] = float(true_positives)
            metrics['false_positives'] = float(false_positives)
            metrics['false_negatives'] = float(false_negatives)
            metrics['true_negatives'] = float(true_negatives)
            
            total_true = np.sum(true_mask)
            if total_true > 0:
                metrics['detection_rate'] = true_positives / total_true
                metrics['recall'] = true_positives / total_true
            
            total_predicted = np.sum(pred_mask)
            if total_predicted > 0:
                metrics['precision'] = true_positives / total_predicted
            
            if metrics['precision'] + metrics['recall'] > 0:
                metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
            
            if true_negatives + false_positives > 0:
                metrics['specificity'] = true_negatives / (true_negatives + false_positives)
        
        return metrics
    
    def _generate_recommendations(self, 
                                statistical_result: StatisticalResult,
                                temporal_params: TemporalParameters,
                                performance_metrics: Dict[str, float],
                                data_shape: Tuple[int, int]) -> List[str]:
        """Generate actionable recommendations based on analysis results."""
        
        recommendations = []
        n_rois, n_timepoints = data_shape
        
        # Temporal optimization recommendations
        if temporal_params.confidence_score < 0.7:
            recommendations.append(
                f"⚠️ Low temporal optimization confidence ({temporal_params.confidence_score:.3f}). "
                f"Consider using TR between 0.5-2.0s for optimal performance."
            )
        
        if temporal_params.tr > 2.5:
            recommendations.append(
                f"⚠️ High TR ({temporal_params.tr}s) may limit sensitivity. "
                f"Consider higher temporal resolution if possible."
            )
        
        if n_timepoints < temporal_params.max_lag * 15:
            recommendations.append(
                f"⚠️ Short time series ({n_timepoints} timepoints) relative to max lag. "
                f"Consider longer acquisition for better reliability."
            )
        
        # Statistical performance recommendations
        if performance_metrics['power_estimate'] < 0.3:
            recommendations.append(
                f"⚠️ Low statistical power ({performance_metrics['power_estimate']:.3f}). "
                f"Consider liberal exploration mode or larger effect sizes."
            )
        
        if performance_metrics['n_significant'] == 0:
            recommendations.append(
                "❌ No significant connections detected. Consider: "
                "(1) Liberal exploration mode, (2) Longer acquisition, (3) Parameter optimization."
            )
        elif performance_metrics['n_significant'] < 5:
            recommendations.append(
                f"🔍 Few connections detected ({int(performance_metrics['n_significant'])}). "
                f"Results may be conservative - consider validation with liberal thresholds."
            )
        
        # Method-specific recommendations
        if statistical_result.method_used == 'liberal_exploration':
            recommendations.append(
                "🔬 Using liberal exploration mode. Results are for hypothesis generation - "
                "validate findings with confirmatory analysis."
            )
        
        if statistical_result.alpha_effective > 0.1:
            recommendations.append(
                f"📊 Liberal statistical threshold (α={statistical_result.alpha_effective:.3f}). "
                f"Consider replication to confirm findings."
            )
        
        # Validation-based recommendations
        if 'f1_score' in performance_metrics and performance_metrics['f1_score'] > 0:
            if performance_metrics['precision'] < 0.5:
                recommendations.append(
                    f"⚠️ Low precision ({performance_metrics['precision']:.3f}) suggests high false positive rate. "
                    f"Consider more conservative thresholds."
                )
            
            if performance_metrics['recall'] < 0.3:
                recommendations.append(
                    f"⚠️ Low recall ({performance_metrics['recall']:.3f}) suggests missing true connections. "
                    f"Consider more liberal thresholds or parameter optimization."
                )
        
        # Success recommendations
        if not recommendations:
            recommendations.append(
                "✅ Analysis looks good! Consider validating results with different parameters "
                "or replication data."
            )
        
        return recommendations
    
    def compare_with_baseline(self, 
                            data: np.ndarray,
                            ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Compare enhanced system with baseline SMTE implementation."""
        
        self.logger.info("Comparing enhanced vs. baseline SMTE...")
        
        # Enhanced analysis
        enhanced_result = self.analyze_connectivity(data, ground_truth=ground_truth)
        
        # Baseline analysis
        from voxel_smte_connectivity_corrected import VoxelSMTEConnectivity
        
        baseline_smte = VoxelSMTEConnectivity(
            n_symbols=2, ordinal_order=2, max_lag=3, n_permutations=50
        )
        
        baseline_smte.fmri_data = data
        baseline_smte.mask = np.ones(data.shape[0], dtype=bool)
        
        symbolic_data = baseline_smte.symbolize_timeseries(data)
        baseline_smte.symbolic_data = symbolic_data
        connectivity_matrix, _ = baseline_smte.compute_voxel_connectivity_matrix()
        baseline_p_values = baseline_smte.statistical_testing(connectivity_matrix)
        
        # Baseline FDR correction
        baseline_significance = baseline_smte.fdr_correction(baseline_p_values)
        
        # Compare results
        comparison = {
            'enhanced': {
                'n_detections': np.sum(enhanced_result.statistical_result.significance_mask),
                'method': enhanced_result.statistical_result.method_used,
                'power': enhanced_result.statistical_result.power_estimate,
                'temporal_confidence': enhanced_result.temporal_params.confidence_score,
                'max_lag': enhanced_result.temporal_params.max_lag
            },
            'baseline': {
                'n_detections': np.sum(baseline_significance),
                'method': 'standard_fdr',
                'power': 0.0,  # Not computed for baseline
                'temporal_confidence': 0.5,  # Default
                'max_lag': 3
            }
        }
        
        # Compute improvement metrics
        enhanced_detections = comparison['enhanced']['n_detections']
        baseline_detections = comparison['baseline']['n_detections']
        
        if baseline_detections > 0:
            detection_improvement = (enhanced_detections - baseline_detections) / baseline_detections * 100
        else:
            detection_improvement = enhanced_detections * 100  # All detections are improvement
        
        comparison['improvement'] = {
            'detection_rate': detection_improvement,
            'parameter_optimization': enhanced_result.temporal_params.confidence_score > 0.7,
            'statistical_method': enhanced_result.statistical_result.method_used != 'fdr_adaptive'
        }
        
        # Validation metrics if ground truth available
        if ground_truth is not None:
            enhanced_metrics = enhanced_result.performance_metrics
            
            # Baseline validation
            baseline_mask = ground_truth > 0.1
            baseline_pred = baseline_significance
            baseline_tp = np.sum(baseline_pred & baseline_mask)
            baseline_fp = np.sum(baseline_pred & ~baseline_mask)
            baseline_precision = baseline_tp / max(np.sum(baseline_pred), 1)
            baseline_recall = baseline_tp / max(np.sum(baseline_mask), 1)
            baseline_f1 = 2 * baseline_precision * baseline_recall / max(baseline_precision + baseline_recall, 0.001)
            
            comparison['validation'] = {
                'enhanced_f1': enhanced_metrics.get('f1_score', 0.0),
                'baseline_f1': baseline_f1,
                'f1_improvement': enhanced_metrics.get('f1_score', 0.0) - baseline_f1,
                'enhanced_precision': enhanced_metrics.get('precision', 0.0),
                'baseline_precision': baseline_precision,
                'enhanced_recall': enhanced_metrics.get('recall', 0.0),
                'baseline_recall': baseline_recall
            }
        
        self.logger.info(f"Comparison complete: Enhanced={enhanced_detections}, Baseline={baseline_detections}")
        
        return comparison
    
    def get_analysis_summary(self, result: EnhancedSMTEResult) -> str:
        """Generate human-readable analysis summary."""
        
        summary = [
            "🧠 ENHANCED SMTE ANALYSIS SUMMARY",
            "=" * 50,
            "",
            "📊 Detection Results:",
            f"  Significant connections: {int(result.performance_metrics['n_significant'])}",
            f"  Statistical method: {result.statistical_result.method_used.replace('_', ' ').title()}",
            f"  Effective α-level: {result.statistical_result.alpha_effective:.3f}",
            f"  Power estimate: {result.statistical_result.power_estimate:.3f}",
            "",
            "⚙️ Temporal Optimization:",
            f"  TR: {result.temporal_params.tr:.3f}s",
            f"  Max lag: {result.temporal_params.max_lag} samples ({result.temporal_params.max_lag * result.temporal_params.tr:.1f}s)",
            f"  Optimal lags: {result.temporal_params.optimal_lags}",
            f"  Symbolization: {result.temporal_params.n_symbols} symbols, order {result.temporal_params.ordinal_order}",
            f"  Confidence: {result.temporal_params.confidence_score:.3f}",
            ""
        ]
        
        # Add validation metrics if available
        if 'f1_score' in result.performance_metrics and result.performance_metrics['f1_score'] > 0:
            summary.extend([
                "✅ Validation Metrics:",
                f"  Precision: {result.performance_metrics['precision']:.3f}",
                f"  Recall: {result.performance_metrics['recall']:.3f}",
                f"  F1-Score: {result.performance_metrics['f1_score']:.3f}",
                f"  Detection Rate: {result.performance_metrics['detection_rate']:.1%}",
                ""
            ])
        
        # Add recommendations
        summary.extend([
            "💡 Recommendations:",
        ])
        
        for i, rec in enumerate(result.recommendations, 1):
            summary.append(f"  {i}. {rec}")
        
        return "\n".join(summary)


def test_enhanced_smte_system():
    """Test the complete enhanced SMTE system."""
    
    print("🚀 TESTING ENHANCED SMTE SYSTEM")
    print("=" * 60)
    
    # Create realistic test data
    np.random.seed(42)
    tr = 2.0
    n_rois = 8
    n_timepoints = 250
    
    # Generate data with hemodynamic-realistic connections
    data = np.random.randn(n_rois, n_timepoints)
    ground_truth = np.zeros((n_rois, n_rois))
    
    # Add connections with proper hemodynamic delays
    hrf_lag = int(6.0 / tr)  # ~6s hemodynamic delay
    connections = [
        (0, 1, 0.4),  # Strong connection
        (1, 2, 0.3),  # Medium connection  
        (3, 4, 0.5),  # Strong connection
        (5, 6, 0.2),  # Weak connection
    ]
    
    for source, target, strength in connections:
        if hrf_lag < n_timepoints:
            data[target, hrf_lag:] += strength * data[source, :-hrf_lag]
            ground_truth[source, target] = strength
    
    # Add some noise connections at different lags
    data[7, 1:] += 0.15 * data[0, :-1]  # Immediate connection
    ground_truth[0, 7] = 0.15
    
    print(f"Test data: TR={tr}s, {n_rois} ROIs, {n_timepoints} timepoints")
    print(f"Ground truth: {np.sum(ground_truth > 0)} connections")
    
    # Initialize enhanced system
    enhanced_system = EnhancedSMTESystem(tr=tr, auto_optimize=True)
    
    # Run complete analysis
    result = enhanced_system.analyze_connectivity(
        data, 
        ground_truth=ground_truth
    )
    
    # Display results
    print("\n" + enhanced_system.get_analysis_summary(result))
    
    # Compare with baseline
    print("\n🔄 BASELINE COMPARISON")
    print("-" * 30)
    
    comparison = enhanced_system.compare_with_baseline(data, ground_truth)
    
    print(f"Enhanced detections: {comparison['enhanced']['n_detections']}")
    print(f"Baseline detections: {comparison['baseline']['n_detections']}")
    print(f"Detection improvement: {comparison['improvement']['detection_rate']:.1f}%")
    
    if 'validation' in comparison:
        print(f"Enhanced F1-score: {comparison['validation']['enhanced_f1']:.3f}")
        print(f"Baseline F1-score: {comparison['validation']['baseline_f1']:.3f}")
        print(f"F1 improvement: {comparison['validation']['f1_improvement']:.3f}")
    
    # Test different TR scenarios
    print("\n🔬 MULTI-TR VALIDATION")
    print("-" * 25)
    
    tr_scenarios = [0.5, 1.0, 2.0, 3.0]
    
    for test_tr in tr_scenarios:
        print(f"\nTesting TR={test_tr}s:")
        
        # Adjust data for different TR
        scale_factor = tr / test_tr
        scaled_timepoints = int(n_timepoints * scale_factor)
        scaled_data = np.random.randn(n_rois, scaled_timepoints)
        scaled_ground_truth = np.zeros((n_rois, n_rois))
        
        # Add connections with appropriate lag for this TR
        scaled_hrf_lag = max(1, int(6.0 / test_tr))
        
        for source, target, strength in connections[:2]:  # Test subset for speed
            if scaled_hrf_lag < scaled_timepoints:
                scaled_data[target, scaled_hrf_lag:] += strength * scaled_data[source, :-scaled_hrf_lag]
                scaled_ground_truth[source, target] = strength
        
        # Quick test
        tr_system = EnhancedSMTESystem(tr=test_tr, auto_optimize=True)
        tr_system.adaptive_smte = AdaptiveSMTEConnectivity(tr=test_tr, n_permutations=20)  # Reduced for speed
        
        try:
            tr_result = tr_system.analyze_connectivity(scaled_data, ground_truth=scaled_ground_truth)
            
            print(f"  Optimized max_lag: {tr_result.temporal_params.max_lag} ({tr_result.temporal_params.max_lag * test_tr:.1f}s)")
            print(f"  Detections: {int(tr_result.performance_metrics['n_significant'])}")
            print(f"  Confidence: {tr_result.temporal_params.confidence_score:.3f}")
            print(f"  Method: {tr_result.statistical_result.method_used}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    print(f"\n✅ Enhanced SMTE system testing complete!")
    
    return result, comparison

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Test the enhanced system
    test_result, test_comparison = test_enhanced_smte_system()