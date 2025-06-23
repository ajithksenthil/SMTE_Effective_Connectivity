#!/usr/bin/env python3
"""
Adaptive Temporal Resolution System for SMTE
Critical fix to address temporal resolution mismatch in fMRI applications.
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging

@dataclass
class TemporalParameters:
    """Container for temporal resolution parameters."""
    tr: float
    max_lag: int
    n_symbols: int
    ordinal_order: int
    hemodynamic_delay: float
    optimal_lags: List[int]
    confidence_score: float

class AdaptiveTemporalSystem:
    """
    Adaptive temporal resolution system that optimizes SMTE parameters
    based on fMRI acquisition parameters and hemodynamic constraints.
    """
    
    def __init__(self):
        # Hemodynamic response function parameters
        self.hrf_peak_time = 6.0  # seconds
        self.hrf_dispersion = 1.5  # seconds
        self.hrf_undershoot_time = 16.0  # seconds
        
        # SMTE constraints
        self.min_max_lag = 3
        self.max_max_lag = 15
        self.min_timepoints_per_lag = 10  # Minimum timepoints needed per lag
        
        self.logger = logging.getLogger(__name__)
    
    def optimize_temporal_parameters(self, tr: float, n_timepoints: int, 
                                   data_properties: Optional[Dict] = None) -> TemporalParameters:
        """
        Optimize SMTE temporal parameters for given TR and data characteristics.
        
        Parameters:
        -----------
        tr : float
            Repetition time in seconds
        n_timepoints : int
            Number of timepoints in the data
        data_properties : dict, optional
            Additional data properties (SNR, motion, etc.)
            
        Returns:
        --------
        TemporalParameters
            Optimized parameters for this temporal resolution
        """
        
        # Calculate optimal max_lag based on hemodynamic constraints
        optimal_max_lag = self._calculate_optimal_max_lag(tr, n_timepoints)
        
        # Determine optimal lag range
        optimal_lags = self._determine_optimal_lag_range(tr, optimal_max_lag)
        
        # Optimize symbolization parameters
        n_symbols, ordinal_order = self._optimize_symbolization_params(
            tr, n_timepoints, data_properties
        )
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score(tr, n_timepoints, optimal_max_lag)
        
        # Estimate hemodynamic delay in samples
        hemodynamic_delay_samples = self.hrf_peak_time / tr
        
        params = TemporalParameters(
            tr=tr,
            max_lag=optimal_max_lag,
            n_symbols=n_symbols,
            ordinal_order=ordinal_order,
            hemodynamic_delay=hemodynamic_delay_samples,
            optimal_lags=optimal_lags,
            confidence_score=confidence
        )
        
        # Log optimization results
        self._log_optimization_results(params)
        
        return params
    
    def _calculate_optimal_max_lag(self, tr: float, n_timepoints: int) -> int:
        """Calculate optimal max_lag based on TR and hemodynamic constraints."""
        
        # Method 1: Based on hemodynamic response peak
        hrf_based_lag = max(3, int(np.ceil(self.hrf_peak_time / tr)))
        
        # Method 2: Based on temporal resolution
        # For high temporal resolution (TR < 1s), allow more lags
        # For low temporal resolution (TR > 2s), use fewer lags
        if tr <= 1.0:
            tr_based_lag = min(15, max(5, int(8 / tr)))
        elif tr <= 2.0:
            tr_based_lag = min(10, max(3, int(6 / tr)))
        else:
            tr_based_lag = max(3, int(4 / tr))
        
        # Method 3: Based on data length constraint
        # Need enough timepoints for reliable estimation
        data_based_lag = min(n_timepoints // self.min_timepoints_per_lag, 10)
        
        # Take the minimum to ensure feasibility
        optimal_lag = min(hrf_based_lag, tr_based_lag, data_based_lag)
        
        # Ensure it's within bounds
        optimal_lag = max(self.min_max_lag, min(self.max_max_lag, optimal_lag))
        
        self.logger.debug(f"Lag calculation: HRF={hrf_based_lag}, TR={tr_based_lag}, "
                         f"Data={data_based_lag}, Final={optimal_lag}")
        
        return optimal_lag
    
    def _determine_optimal_lag_range(self, tr: float, max_lag: int) -> List[int]:
        """Determine which specific lags to focus on."""
        
        # Always include lag 1 (immediate response)
        optimal_lags = [1]
        
        # Add hemodynamic response peak lag
        hrf_lag = max(1, int(np.round(self.hrf_peak_time / tr)))
        if hrf_lag <= max_lag and hrf_lag not in optimal_lags:
            optimal_lags.append(hrf_lag)
        
        # Add intermediate lags for high temporal resolution
        if tr <= 1.0 and max_lag >= 5:
            # For high TR, add more intermediate lags
            intermediate_lags = [2, 3, int(max_lag * 0.7), max_lag]
        elif tr <= 2.0:
            # For medium TR, add selective lags
            intermediate_lags = [2, max(3, int(max_lag * 0.6)), max_lag]
        else:
            # For low TR, focus on key lags
            intermediate_lags = [max(2, max_lag // 2), max_lag]
        
        # Add intermediate lags that aren't already included
        for lag in intermediate_lags:
            if lag <= max_lag and lag not in optimal_lags:
                optimal_lags.append(lag)
        
        return sorted(optimal_lags)
    
    def _optimize_symbolization_params(self, tr: float, n_timepoints: int, 
                                     data_properties: Optional[Dict] = None) -> Tuple[int, int]:
        """Optimize symbolization parameters based on temporal resolution."""
        
        # Default conservative parameters
        n_symbols = 2
        ordinal_order = 2
        
        # Adjust based on temporal resolution and data length
        if tr <= 1.0 and n_timepoints >= 300:
            # High temporal resolution, long time series
            n_symbols = 3
            ordinal_order = 3
        elif tr <= 2.0 and n_timepoints >= 200:
            # Medium temporal resolution
            n_symbols = 2
            ordinal_order = 3
        else:
            # Low temporal resolution or short time series
            n_symbols = 2
            ordinal_order = 2
        
        # Adjust based on data properties if available
        if data_properties:
            snr = data_properties.get('snr', 1.0)
            if snr < 0.5:  # Low SNR
                n_symbols = 2  # Use simpler symbolization
                ordinal_order = 2
            elif snr > 2.0:  # High SNR
                n_symbols = min(3, n_symbols)
                ordinal_order = min(4, ordinal_order + 1)
        
        return n_symbols, ordinal_order
    
    def _calculate_confidence_score(self, tr: float, n_timepoints: int, max_lag: int) -> float:
        """Calculate confidence score for the optimization."""
        
        scores = []
        
        # Temporal resolution score (higher for TR close to optimal ~1-2s)
        if 0.5 <= tr <= 2.0:
            tr_score = 1.0 - abs(tr - 1.0) / 1.5
        else:
            tr_score = max(0.1, 1.0 - abs(tr - 1.0) / 3.0)
        scores.append(tr_score)
        
        # Data length score
        min_required = max_lag * self.min_timepoints_per_lag
        if n_timepoints >= min_required * 2:
            length_score = 1.0
        elif n_timepoints >= min_required:
            length_score = 0.7
        else:
            length_score = 0.3
        scores.append(length_score)
        
        # Hemodynamic compatibility score
        optimal_hrf_lag = self.hrf_peak_time / tr
        if max_lag >= optimal_hrf_lag * 0.8:
            hrf_score = 1.0
        else:
            hrf_score = max_lag / optimal_hrf_lag
        scores.append(hrf_score)
        
        # Overall confidence is the geometric mean
        confidence = np.prod(scores) ** (1.0 / len(scores))
        
        return confidence
    
    def _log_optimization_results(self, params: TemporalParameters):
        """Log the optimization results."""
        
        self.logger.info(f"Temporal Parameter Optimization Results:")
        self.logger.info(f"  TR: {params.tr:.2f}s")
        self.logger.info(f"  Max Lag: {params.max_lag} samples ({params.max_lag * params.tr:.1f}s)")
        self.logger.info(f"  Optimal Lags: {params.optimal_lags}")
        self.logger.info(f"  Symbolization: {params.n_symbols} symbols, order {params.ordinal_order}")
        self.logger.info(f"  Hemodynamic Delay: {params.hemodynamic_delay:.1f} samples")
        self.logger.info(f"  Confidence Score: {params.confidence_score:.3f}")
        
        # Add warnings for suboptimal conditions
        if params.confidence_score < 0.5:
            warnings.warn(f"Low confidence ({params.confidence_score:.3f}) in temporal optimization. "
                         f"Consider using TR between 0.5-2.0s for optimal SMTE performance.")
        
        if params.tr > 3.0:
            warnings.warn(f"High TR ({params.tr}s) may limit SMTE sensitivity. "
                         f"Consider higher temporal resolution data.")
    
    def validate_parameters(self, params: TemporalParameters, 
                          data_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Validate optimized parameters against actual data."""
        
        n_rois, n_timepoints = data_shape
        validation_results = {}
        
        # Check if we have enough timepoints
        min_required = params.max_lag * self.min_timepoints_per_lag
        sufficient_data = n_timepoints >= min_required
        
        validation_results['sufficient_data'] = sufficient_data
        validation_results['required_timepoints'] = min_required
        validation_results['actual_timepoints'] = n_timepoints
        
        # Check temporal coverage
        max_time_lag = params.max_lag * params.tr
        validation_results['max_temporal_lag'] = max_time_lag
        validation_results['covers_hrf_peak'] = max_time_lag >= self.hrf_peak_time * 0.8
        
        # Check symbolization feasibility
        patterns_per_lag = params.n_symbols ** params.ordinal_order
        effective_timepoints = n_timepoints - params.max_lag
        samples_per_pattern = effective_timepoints / patterns_per_lag
        
        validation_results['samples_per_pattern'] = samples_per_pattern
        validation_results['sufficient_symbolization'] = samples_per_pattern >= 5
        
        # Overall validation
        validation_results['overall_valid'] = (
            sufficient_data and 
            validation_results['covers_hrf_peak'] and 
            validation_results['sufficient_symbolization']
        )
        
        return validation_results


class AdaptiveSMTEConnectivity:
    """
    Enhanced SMTE connectivity class with adaptive temporal resolution.
    Extends the existing implementation with automatic parameter optimization.
    """
    
    def __init__(self, tr: float, n_timepoints: int = None, 
                 data_properties: Optional[Dict] = None,
                 force_parameters: Optional[Dict] = None,
                 **kwargs):
        """
        Initialize with adaptive temporal parameter optimization.
        
        Parameters:
        -----------
        tr : float
            Repetition time in seconds
        n_timepoints : int, optional
            Number of timepoints (can be set later)
        data_properties : dict, optional
            Data characteristics for optimization
        force_parameters : dict, optional
            Override automatic optimization
        **kwargs : dict
            Additional parameters for base SMTE class
        """
        
        self.tr = tr
        self.n_timepoints = n_timepoints
        self.data_properties = data_properties or {}
        self.temporal_system = AdaptiveTemporalSystem()
        
        # Optimize parameters if not forced
        if force_parameters:
            self.temporal_params = self._create_forced_params(force_parameters)
        elif n_timepoints is not None:
            self.temporal_params = self.temporal_system.optimize_temporal_parameters(
                tr, n_timepoints, data_properties
            )
        else:
            self.temporal_params = None
        
        # Initialize base SMTE with optimized parameters
        if self.temporal_params:
            base_params = {
                'n_symbols': self.temporal_params.n_symbols,
                'ordinal_order': self.temporal_params.ordinal_order,
                'max_lag': self.temporal_params.max_lag,
                **kwargs
            }
        else:
            base_params = kwargs
        
        # Import and initialize base class
        from voxel_smte_connectivity_corrected import VoxelSMTEConnectivity
        self.base_smte = VoxelSMTEConnectivity(**base_params)
        
        # Copy base class methods
        for attr in dir(self.base_smte):
            if not attr.startswith('_') and callable(getattr(self.base_smte, attr)):
                setattr(self, attr, getattr(self.base_smte, attr))
    
    def _create_forced_params(self, force_parameters: Dict) -> TemporalParameters:
        """Create TemporalParameters from forced parameters."""
        
        return TemporalParameters(
            tr=self.tr,
            max_lag=force_parameters.get('max_lag', 3),
            n_symbols=force_parameters.get('n_symbols', 2),
            ordinal_order=force_parameters.get('ordinal_order', 2),
            hemodynamic_delay=force_parameters.get('hemodynamic_delay', 3.0),
            optimal_lags=force_parameters.get('optimal_lags', [1, 2, 3]),
            confidence_score=force_parameters.get('confidence_score', 0.5)
        )
    
    def auto_configure(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Automatically configure parameters based on data.
        
        Parameters:
        -----------
        data : np.ndarray
            fMRI data (n_rois x n_timepoints)
            
        Returns:
        --------
        dict
            Configuration results and validation
        """
        
        n_rois, n_timepoints = data.shape
        self.n_timepoints = n_timepoints
        
        # Analyze data properties
        data_properties = self._analyze_data_properties(data)
        self.data_properties.update(data_properties)
        
        # Optimize parameters
        self.temporal_params = self.temporal_system.optimize_temporal_parameters(
            self.tr, n_timepoints, self.data_properties
        )
        
        # Update base SMTE parameters
        self.base_smte.n_symbols = self.temporal_params.n_symbols
        self.base_smte.ordinal_order = self.temporal_params.ordinal_order
        self.base_smte.max_lag = self.temporal_params.max_lag
        
        # Validate configuration
        validation = self.temporal_system.validate_parameters(
            self.temporal_params, (n_rois, n_timepoints)
        )
        
        return {
            'temporal_params': self.temporal_params,
            'data_properties': self.data_properties,
            'validation': validation,
            'recommendations': self._generate_recommendations(validation)
        }
    
    def _analyze_data_properties(self, data: np.ndarray) -> Dict[str, float]:
        """Analyze data properties for parameter optimization."""
        
        # Estimate SNR
        signal_var = np.var(np.mean(data, axis=0))
        noise_var = np.mean(np.var(data, axis=1))
        snr = signal_var / max(noise_var, 1e-10)
        
        # Estimate temporal smoothness
        temporal_corr = np.mean([
            np.corrcoef(data[i, :-1], data[i, 1:])[0, 1] 
            for i in range(data.shape[0])
        ])
        
        # Estimate motion (proxy via high-frequency content)
        high_freq_power = np.mean([
            np.var(np.diff(data[i])) for i in range(data.shape[0])
        ])
        
        return {
            'snr': snr,
            'temporal_correlation': temporal_corr,
            'high_frequency_power': high_freq_power,
            'mean_signal': np.mean(data),
            'signal_range': np.ptp(data)
        }
    
    def _generate_recommendations(self, validation: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        
        recommendations = []
        
        if not validation['sufficient_data']:
            recommendations.append(
                f"Consider longer acquisition: need ≥{validation['required_timepoints']} "
                f"timepoints, have {validation['actual_timepoints']}"
            )
        
        if not validation['covers_hrf_peak']:
            recommendations.append(
                f"Consider higher temporal resolution: max lag covers "
                f"{validation['max_temporal_lag']:.1f}s, need ≥{self.temporal_system.hrf_peak_time * 0.8:.1f}s"
            )
        
        if not validation['sufficient_symbolization']:
            recommendations.append(
                f"Consider simpler symbolization: only {validation['samples_per_pattern']:.1f} "
                f"samples per pattern, recommend ≥5"
            )
        
        if self.temporal_params.confidence_score < 0.7:
            recommendations.append(
                f"Low optimization confidence ({self.temporal_params.confidence_score:.3f}). "
                f"Consider optimizing acquisition parameters."
            )
        
        if not recommendations:
            recommendations.append("Configuration looks good for SMTE analysis!")
        
        return recommendations
    
    def get_optimization_summary(self) -> str:
        """Get human-readable summary of optimization."""
        
        if not self.temporal_params:
            return "No optimization performed yet. Call auto_configure() first."
        
        summary = [
            "🎯 ADAPTIVE TEMPORAL OPTIMIZATION SUMMARY",
            "=" * 50,
            f"Input Parameters:",
            f"  TR: {self.tr:.3f}s",
            f"  Timepoints: {self.n_timepoints}",
            "",
            f"Optimized SMTE Parameters:",
            f"  Max Lag: {self.temporal_params.max_lag} samples ({self.temporal_params.max_lag * self.tr:.2f}s)",
            f"  Focus Lags: {self.temporal_params.optimal_lags}",
            f"  Symbolization: {self.temporal_params.n_symbols} symbols, order {self.temporal_params.ordinal_order}",
            f"  HRF Coverage: {self.temporal_params.hemodynamic_delay:.1f} samples",
            "",
            f"Optimization Quality:",
            f"  Confidence Score: {self.temporal_params.confidence_score:.3f}/1.0",
            ""
        ]
        
        if hasattr(self, '_last_validation'):
            validation = self._last_validation
            summary.extend([
                f"Validation Results:",
                f"  Sufficient Data: {'✅' if validation['sufficient_data'] else '❌'}",
                f"  HRF Coverage: {'✅' if validation['covers_hrf_peak'] else '❌'}",
                f"  Symbolization: {'✅' if validation['sufficient_symbolization'] else '❌'}",
                f"  Overall Valid: {'✅' if validation['overall_valid'] else '❌'}",
            ])
        
        return "\n".join(summary)


def test_adaptive_temporal_system():
    """Test the adaptive temporal system with different scenarios."""
    
    print("🧪 TESTING ADAPTIVE TEMPORAL RESOLUTION SYSTEM")
    print("=" * 60)
    
    # Test scenarios with different TRs
    test_scenarios = [
        {"name": "High-res fMRI", "tr": 0.5, "n_timepoints": 600},
        {"name": "Standard fMRI", "tr": 2.0, "n_timepoints": 300},
        {"name": "Clinical fMRI", "tr": 3.0, "n_timepoints": 200},
        {"name": "Long acquisition", "tr": 1.0, "n_timepoints": 800},
        {"name": "Short acquisition", "tr": 2.0, "n_timepoints": 150}
    ]
    
    temporal_system = AdaptiveTemporalSystem()
    results = {}
    
    for scenario in test_scenarios:
        print(f"\n📊 Testing {scenario['name']} (TR={scenario['tr']}s, n={scenario['n_timepoints']})")
        
        # Generate test data
        np.random.seed(42)
        n_rois = 10
        data = np.random.randn(n_rois, scenario['n_timepoints'])
        
        # Add realistic hemodynamic connection
        hrf_lag = max(1, int(6.0 / scenario['tr']))  # HRF peak delay
        if hrf_lag < scenario['n_timepoints']:
            data[1, hrf_lag:] += 0.5 * data[0, :-hrf_lag]
        
        # Test adaptive SMTE
        adaptive_smte = AdaptiveSMTEConnectivity(tr=scenario['tr'])
        config_results = adaptive_smte.auto_configure(data)
        
        # Test detection
        try:
            adaptive_smte.base_smte.fmri_data = data
            adaptive_smte.base_smte.mask = np.ones(n_rois, dtype=bool)
            
            symbolic_data = adaptive_smte.base_smte.symbolize_timeseries(data)
            adaptive_smte.base_smte.symbolic_data = symbolic_data
            
            connectivity_matrix, _ = adaptive_smte.base_smte.compute_voxel_connectivity_matrix()
            p_values = adaptive_smte.base_smte.statistical_testing(connectivity_matrix)
            
            # Test detection with uncorrected p-values
            uncorrected_detections = np.sum(p_values < 0.05)
            
            # Store results
            results[scenario['name']] = {
                'temporal_params': config_results['temporal_params'],
                'validation': config_results['validation'],
                'detections': uncorrected_detections,
                'min_p_value': np.min(p_values),
                'test_connection_p': p_values[0, 1]  # Our known connection
            }
            
            print(f"   Optimized max_lag: {config_results['temporal_params'].max_lag}")
            print(f"   Confidence: {config_results['temporal_params'].confidence_score:.3f}")
            print(f"   Detections: {uncorrected_detections}")
            print(f"   Test connection p-value: {p_values[0, 1]:.6f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[scenario['name']] = {'error': str(e)}
    
    # Summary
    print(f"\n📋 SUMMARY OF ADAPTIVE SYSTEM PERFORMANCE")
    print("=" * 50)
    
    for name, result in results.items():
        if 'error' in result:
            print(f"{name}: Failed - {result['error']}")
        else:
            print(f"{name}:")
            print(f"  Max lag: {result['temporal_params'].max_lag} samples")
            print(f"  Confidence: {result['temporal_params'].confidence_score:.3f}")
            print(f"  Detections: {result['detections']}")
            print(f"  Test p-value: {result['test_connection_p']:.6f}")
    
    return results

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Test the adaptive system
    test_results = test_adaptive_temporal_system()