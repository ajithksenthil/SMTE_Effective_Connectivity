#!/usr/bin/env python3
"""
Phase 2.1: Multi-Scale Temporal Analysis for SMTE
This module implements multi-scale temporal analysis to capture dynamics across different time scales.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.preprocessing import StandardScaler
import logging
from scipy import signal
from scipy.stats import pearsonr
import seaborn as sns

from physiological_smte_v1 import PhysiologicalSMTE

logging.basicConfig(level=logging.INFO)


class MultiScaleAnalyzer:
    """
    Implements multi-scale temporal analysis for brain connectivity.
    """
    
    def __init__(self, TR: float = 2.0):
        self.TR = TR
        
        # Define temporal scales for fMRI analysis
        self.temporal_scales = {
            'fast': {
                'description': 'Fast neural dynamics',
                'lag_range': (1, 3),     # 1-3 TRs (2-6 seconds)
                'frequency_band': (0.08, 0.25),  # High frequency for fMRI
                'window_size': 20,       # Short windows
                'expected_networks': ['sensorimotor', 'visual', 'auditory']
            },
            'intermediate': {
                'description': 'Intermediate cognitive dynamics',
                'lag_range': (3, 8),     # 3-8 TRs (6-16 seconds)
                'frequency_band': (0.03, 0.08),  # Intermediate frequency
                'window_size': 40,       # Medium windows
                'expected_networks': ['executive', 'salience', 'attention']
            },
            'slow': {
                'description': 'Slow network dynamics',
                'lag_range': (8, 20),    # 8-20 TRs (16-40 seconds)
                'frequency_band': (0.01, 0.03),  # Low frequency
                'window_size': 60,       # Long windows
                'expected_networks': ['default_mode', 'global']
            }
        }
        
        # Scale-specific parameters
        self.scale_parameters = {
            'fast': {
                'ordinal_order': 3,
                'n_symbols': 6,
                'alpha': 0.05,
                'weight': 1.0
            },
            'intermediate': {
                'ordinal_order': 4,
                'n_symbols': 24,
                'alpha': 0.03,
                'weight': 1.2  # Slightly higher weight for cognitive scales
            },
            'slow': {
                'ordinal_order': 3,
                'n_symbols': 6,
                'alpha': 0.01,
                'weight': 0.8  # Lower weight for slow scales
            }
        }
    
    def preprocess_for_scale(self, 
                           data: np.ndarray,
                           scale_name: str) -> np.ndarray:
        """
        Preprocess data for specific temporal scale analysis.
        """
        
        scale_info = self.temporal_scales[scale_name]
        frequency_band = scale_info['frequency_band']
        
        # Apply band-pass filtering for the scale
        preprocessed_data = self._apply_bandpass_filter(data, frequency_band)
        
        # Optional: Apply scale-specific downsampling or smoothing
        if scale_name == 'slow':
            # Light smoothing for slow scale
            preprocessed_data = self._apply_temporal_smoothing(preprocessed_data, window=3)
        elif scale_name == 'fast':
            # Minimal processing for fast scale to preserve dynamics
            pass
        
        return preprocessed_data
    
    def _apply_bandpass_filter(self, 
                             data: np.ndarray,
                             frequency_band: Tuple[float, float]) -> np.ndarray:
        """
        Apply band-pass filter to isolate frequency band of interest.
        """
        
        # Calculate Nyquist frequency
        fs = 1.0 / self.TR  # Sampling frequency
        nyquist = fs / 2.0
        
        # Normalize frequencies
        low_freq = frequency_band[0] / nyquist
        high_freq = frequency_band[1] / nyquist
        
        # Ensure frequencies are within valid range
        low_freq = max(0.01, min(low_freq, 0.99))
        high_freq = max(low_freq + 0.01, min(high_freq, 0.99))
        
        filtered_data = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            try:
                # Design band-pass filter
                b, a = signal.butter(3, [low_freq, high_freq], btype='bandpass')
                
                # Apply filter
                filtered_signal = signal.filtfilt(b, a, data[i])
                filtered_data[i] = filtered_signal
                
            except Exception as e:
                # If filtering fails, use original signal
                filtered_data[i] = data[i]
        
        return filtered_data
    
    def _apply_temporal_smoothing(self, 
                                data: np.ndarray,
                                window: int = 3) -> np.ndarray:
        """
        Apply temporal smoothing using moving average.
        """
        
        smoothed_data = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            # Apply moving average
            smoothed_signal = np.convolve(data[i], np.ones(window)/window, mode='same')
            smoothed_data[i] = smoothed_signal
        
        return smoothed_data
    
    def compute_scale_specific_connectivity(self,
                                          data: np.ndarray,
                                          scale_name: str,
                                          smte_analyzer: Any) -> Dict[str, Any]:
        """
        Compute connectivity for a specific temporal scale.
        """
        
        print(f"Computing connectivity for {scale_name} scale...")
        
        scale_info = self.temporal_scales[scale_name]
        scale_params = self.scale_parameters[scale_name]
        
        # Preprocess data for this scale
        scale_data = self.preprocess_for_scale(data, scale_name)
        
        # Update SMTE parameters for this scale
        original_params = self._backup_smte_parameters(smte_analyzer)
        self._apply_scale_parameters(smte_analyzer, scale_params, scale_info)
        
        try:
            # Compute connectivity
            symbolic_data = smte_analyzer.symbolize_timeseries(scale_data)
            smte_analyzer.symbolic_data = symbolic_data
            connectivity_matrix, lag_matrix = smte_analyzer.compute_voxel_connectivity_matrix()
            
            # Statistical testing with scale-specific alpha
            p_values = smte_analyzer.statistical_testing(connectivity_matrix)
            
            # Temporarily set alpha for FDR correction
            original_alpha = smte_analyzer.alpha
            smte_analyzer.alpha = scale_params['alpha']
            significance_mask = smte_analyzer.fdr_correction(p_values)
            smte_analyzer.alpha = original_alpha  # Restore original alpha
            
            scale_results = {
                'scale_name': scale_name,
                'connectivity_matrix': connectivity_matrix,
                'lag_matrix': lag_matrix,
                'p_values': p_values,
                'significance_mask': significance_mask,
                'n_significant': np.sum(significance_mask),
                'scale_info': scale_info,
                'scale_parameters': scale_params,
                'preprocessed_data': scale_data
            }
            
        finally:
            # Restore original parameters
            self._restore_smte_parameters(smte_analyzer, original_params)
        
        return scale_results
    
    def _backup_smte_parameters(self, smte_analyzer: Any) -> Dict[str, Any]:
        """Backup current SMTE parameters."""
        return {
            'ordinal_order': smte_analyzer.ordinal_order,
            'n_symbols': smte_analyzer.n_symbols,
            'max_lag': smte_analyzer.max_lag,
            'alpha': smte_analyzer.alpha
        }
    
    def _apply_scale_parameters(self, 
                              smte_analyzer: Any,
                              scale_params: Dict[str, Any],
                              scale_info: Dict[str, Any]):
        """Apply scale-specific parameters to SMTE analyzer."""
        smte_analyzer.ordinal_order = scale_params['ordinal_order']
        smte_analyzer.n_symbols = scale_params['n_symbols']
        smte_analyzer.max_lag = scale_info['lag_range'][1]  # Use max lag for scale
        smte_analyzer.alpha = scale_params['alpha']
    
    def _restore_smte_parameters(self, 
                               smte_analyzer: Any,
                               original_params: Dict[str, Any]):
        """Restore original SMTE parameters."""
        for param, value in original_params.items():
            setattr(smte_analyzer, param, value)
    
    def combine_multiscale_results(self,
                                 scale_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine results across multiple temporal scales.
        """
        
        print("Combining multi-scale results...")
        
        if not scale_results:
            raise ValueError("No scale results provided")
        
        # Get dimensions from first scale
        first_scale = list(scale_results.values())[0]
        n_rois = first_scale['connectivity_matrix'].shape[0]
        
        # Initialize combined results
        combined_connectivity = np.zeros((n_rois, n_rois))
        combined_significance = np.zeros((n_rois, n_rois), dtype=bool)
        combined_p_values = np.ones((n_rois, n_rois))
        scale_contributions = np.zeros((n_rois, n_rois, len(scale_results)))
        
        # Combine across scales using weighted average
        total_weight = 0.0
        scale_names = []
        
        for scale_idx, (scale_name, results) in enumerate(scale_results.items()):
            scale_names.append(scale_name)
            weight = self.scale_parameters[scale_name]['weight']
            
            # Weighted combination of connectivity matrices
            combined_connectivity += weight * results['connectivity_matrix']
            
            # Combined significance (any scale significant)
            combined_significance |= results['significance_mask']
            
            # Minimum p-value across scales
            combined_p_values = np.minimum(combined_p_values, results['p_values'])
            
            # Store individual scale contributions
            scale_contributions[:, :, scale_idx] = results['connectivity_matrix']
            
            total_weight += weight
        
        # Normalize by total weight
        combined_connectivity /= total_weight
        
        # Compute scale-specific statistics
        scale_statistics = self._compute_scale_statistics(scale_results)
        
        # Identify dominant scale for each connection
        dominant_scales = self._identify_dominant_scales(scale_contributions, scale_names)
        
        combined_results = {
            'combined_connectivity': combined_connectivity,
            'combined_significance': combined_significance,
            'combined_p_values': combined_p_values,
            'n_combined_significant': np.sum(combined_significance),
            'scale_contributions': scale_contributions,
            'scale_names': scale_names,
            'scale_statistics': scale_statistics,
            'dominant_scales': dominant_scales,
            'individual_scale_results': scale_results
        }
        
        return combined_results
    
    def _compute_scale_statistics(self,
                                scale_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistics across scales."""
        
        stats = {}
        
        for scale_name, results in scale_results.items():
            connectivity = results['connectivity_matrix']
            significance = results['significance_mask']
            
            stats[scale_name] = {
                'n_significant': results['n_significant'],
                'mean_connectivity': np.mean(connectivity[significance]) if np.any(significance) else 0.0,
                'max_connectivity': np.max(connectivity),
                'connectivity_range': np.max(connectivity) - np.min(connectivity),
                'significance_proportion': np.sum(significance) / (significance.size - significance.shape[0])  # Exclude diagonal
            }
        
        return stats
    
    def _identify_dominant_scales(self,
                                scale_contributions: np.ndarray,
                                scale_names: List[str]) -> np.ndarray:
        """Identify the dominant scale for each connection."""
        
        n_rois = scale_contributions.shape[0]
        dominant_scales = np.zeros((n_rois, n_rois), dtype=int)
        
        for i in range(n_rois):
            for j in range(n_rois):
                if i != j:
                    # Find scale with maximum contribution
                    contributions = scale_contributions[i, j, :]
                    dominant_scale_idx = np.argmax(contributions)
                    dominant_scales[i, j] = dominant_scale_idx
        
        return dominant_scales


class MultiScaleSMTE(PhysiologicalSMTE):
    """
    SMTE implementation with multi-scale temporal analysis.
    """
    
    def __init__(self,
                 use_multiscale_analysis: bool = True,
                 scales_to_analyze: List[str] = ['fast', 'intermediate', 'slow'],
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.use_multiscale_analysis = use_multiscale_analysis
        self.scales_to_analyze = scales_to_analyze
        
        # Initialize multi-scale analyzer
        self.multiscale_analyzer = MultiScaleAnalyzer(TR=getattr(self, 'TR', 2.0))
        
        # Store multi-scale results
        self.multiscale_results = None
        
    def compute_multiscale_connectivity(self,
                                      data: np.ndarray,
                                      roi_labels: List[str],
                                      ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute connectivity with multi-scale temporal analysis.
        """
        
        print("Computing multi-scale SMTE connectivity...")
        print(f"Analyzing scales: {self.scales_to_analyze}")
        
        if not self.use_multiscale_analysis:
            # Fall back to single-scale analysis
            return self.compute_physiologically_constrained_connectivity(
                data, roi_labels, ground_truth
            )
        
        # Compute connectivity for each temporal scale
        scale_results = {}
        
        for scale_name in self.scales_to_analyze:
            if scale_name in self.multiscale_analyzer.temporal_scales:
                scale_result = self.multiscale_analyzer.compute_scale_specific_connectivity(
                    data, scale_name, self
                )
                scale_results[scale_name] = scale_result
            else:
                print(f"Warning: Unknown scale '{scale_name}', skipping...")
        
        if not scale_results:
            raise ValueError("No valid scales analyzed")
        
        # Combine multi-scale results
        combined_results = self.multiscale_analyzer.combine_multiscale_results(scale_results)
        
        # Apply physiological constraints if enabled
        if self.use_physiological_constraints:
            print("Applying physiological constraints to combined results...")
            
            # Apply constraints to combined connectivity
            # Get network assignments safely
            network_structure = getattr(self, 'network_structure', None)
            network_assignments = None
            if network_structure and isinstance(network_structure, dict):
                network_assignments = network_structure.get('network_assignments', None)
            
            physio_mask, constraint_info = self.apply_physiological_filtering(
                combined_results['combined_connectivity'],
                scale_results[self.scales_to_analyze[0]]['lag_matrix'],  # Use first scale's lags as reference
                roi_labels,
                network_assignments
            )
            
            # Update combined significance with physiological constraints
            final_significance = combined_results['combined_significance'] & physio_mask
            
            combined_results.update({
                'physiological_mask': physio_mask,
                'final_significance_mask': final_significance,
                'n_final_significant': np.sum(final_significance),
                'constraint_info': constraint_info
            })
        else:
            combined_results.update({
                'physiological_mask': np.ones_like(combined_results['combined_connectivity'], dtype=bool),
                'final_significance_mask': combined_results['combined_significance'],
                'n_final_significant': combined_results['n_combined_significant'],
                'constraint_info': {'constraints_applied': ['none']}
            })
        
        # Store results
        self.multiscale_results = combined_results
        
        print(f"Multi-scale analysis complete:")
        for scale_name in self.scales_to_analyze:
            if scale_name in scale_results:
                n_sig = scale_results[scale_name]['n_significant']
                print(f"  {scale_name} scale: {n_sig} significant connections")
        print(f"  Combined: {combined_results['n_combined_significant']} significant connections")
        print(f"  Final (with constraints): {combined_results['n_final_significant']} connections")
        
        return combined_results
    
    def analyze_scale_contributions(self,
                                  multiscale_results: Dict[str, Any],
                                  roi_labels: List[str]) -> Dict[str, Any]:
        """
        Analyze the contribution of different scales to connectivity patterns.
        """
        
        print("Analyzing scale contributions...")
        
        scale_contributions = multiscale_results['scale_contributions']
        scale_names = multiscale_results['scale_names']
        dominant_scales = multiscale_results['dominant_scales']
        
        n_rois = len(roi_labels)
        analysis = {
            'scale_dominance': {},
            'scale_overlap': {},
            'network_scale_preferences': {},
            'connection_categorization': {}
        }
        
        # Analyze scale dominance
        for scale_idx, scale_name in enumerate(scale_names):
            dominance_count = np.sum(dominant_scales == scale_idx) - n_rois  # Exclude diagonal
            total_connections = n_rois * (n_rois - 1)
            dominance_proportion = dominance_count / total_connections
            
            analysis['scale_dominance'][scale_name] = {
                'dominant_connections': dominance_count,
                'dominance_proportion': dominance_proportion
            }
        
        # Analyze scale overlap (connections significant in multiple scales)
        individual_results = multiscale_results['individual_scale_results']
        overlap_matrix = np.zeros((len(scale_names), len(scale_names)))
        
        for i, scale1 in enumerate(scale_names):
            for j, scale2 in enumerate(scale_names):
                if i != j:
                    sig1 = individual_results[scale1]['significance_mask']
                    sig2 = individual_results[scale2]['significance_mask']
                    overlap = np.sum(sig1 & sig2) - n_rois  # Exclude diagonal
                    overlap_matrix[i, j] = overlap
        
        analysis['scale_overlap'] = {
            'overlap_matrix': overlap_matrix,
            'scale_names': scale_names
        }
        
        # Network-scale preferences (if network info available)
        if hasattr(self, 'network_structure') and self.network_structure:
            network_assignments = self.network_structure.get('network_assignments', {})
            analysis['network_scale_preferences'] = self._analyze_network_scale_preferences(
                individual_results, network_assignments, roi_labels
            )
        
        return analysis
    
    def _analyze_network_scale_preferences(self,
                                         individual_results: Dict[str, Any],
                                         network_assignments: Dict[int, str],
                                         roi_labels: List[str]) -> Dict[str, Any]:
        """Analyze which scales are preferred for different brain networks."""
        
        preferences = {}
        
        # Group ROIs by network
        networks = {}
        for roi_idx, network in network_assignments.items():
            if network not in networks:
                networks[network] = []
            networks[network].append(roi_idx)
        
        # Analyze scale preferences for each network
        for network_name, roi_indices in networks.items():
            network_preferences = {}
            
            for scale_name, results in individual_results.items():
                significance_mask = results['significance_mask']
                
                # Count within-network connections for this scale
                within_network_connections = 0
                total_within_network = 0
                
                for i in roi_indices:
                    for j in roi_indices:
                        if i != j and i < len(roi_labels) and j < len(roi_labels):
                            total_within_network += 1
                            if significance_mask[i, j]:
                                within_network_connections += 1
                
                if total_within_network > 0:
                    preference_score = within_network_connections / total_within_network
                else:
                    preference_score = 0.0
                
                network_preferences[scale_name] = {
                    'within_network_significant': within_network_connections,
                    'total_within_network': total_within_network,
                    'preference_score': preference_score
                }
            
            preferences[network_name] = network_preferences
        
        return preferences
    
    def create_multiscale_visualizations(self,
                                       multiscale_results: Dict[str, Any],
                                       roi_labels: List[str],
                                       save_prefix: str = 'multiscale_smte'):
        """Create comprehensive multi-scale visualizations."""
        
        # 1. Individual scale results
        individual_results = multiscale_results['individual_scale_results']
        n_scales = len(individual_results)
        
        fig1, axes1 = plt.subplots(2, n_scales, figsize=(5*n_scales, 10))
        if n_scales == 1:
            axes1 = axes1.reshape(2, 1)
        
        for idx, (scale_name, results) in enumerate(individual_results.items()):
            # Raw connectivity
            im1 = axes1[0, idx].imshow(results['connectivity_matrix'], cmap='viridis', aspect='auto')
            axes1[0, idx].set_title(f'{scale_name.title()} Scale\nConnectivity')
            axes1[0, idx].set_xlabel('Source ROI')
            axes1[0, idx].set_ylabel('Target ROI')
            plt.colorbar(im1, ax=axes1[0, idx], fraction=0.046, pad=0.04)
            
            # Significant connections
            significant_conn = results['connectivity_matrix'] * results['significance_mask']
            im2 = axes1[1, idx].imshow(significant_conn, cmap='viridis', aspect='auto')
            axes1[1, idx].set_title(f'{scale_name.title()} Scale\nSignificant ({results["n_significant"]})')
            axes1[1, idx].set_xlabel('Source ROI')
            axes1[1, idx].set_ylabel('Target ROI')
            plt.colorbar(im2, ax=axes1[1, idx], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_individual_scales.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Combined results and scale dominance
        fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))
        
        # Combined connectivity
        im1 = axes2[0, 0].imshow(multiscale_results['combined_connectivity'], cmap='viridis', aspect='auto')
        axes2[0, 0].set_title('Combined Multi-Scale\nConnectivity')
        axes2[0, 0].set_xlabel('Source ROI')
        axes2[0, 0].set_ylabel('Target ROI')
        plt.colorbar(im1, ax=axes2[0, 0], fraction=0.046, pad=0.04)
        
        # Final significant connections
        final_significant = (multiscale_results['combined_connectivity'] * 
                           multiscale_results['final_significance_mask'])
        im2 = axes2[0, 1].imshow(final_significant, cmap='viridis', aspect='auto')
        axes2[0, 1].set_title(f'Final Significant\n({multiscale_results["n_final_significant"]} connections)')
        axes2[0, 1].set_xlabel('Source ROI')
        axes2[0, 1].set_ylabel('Target ROI')
        plt.colorbar(im2, ax=axes2[0, 1], fraction=0.046, pad=0.04)
        
        # Dominant scales
        dominant_scales = multiscale_results['dominant_scales']
        im3 = axes2[1, 0].imshow(dominant_scales, cmap='tab10', aspect='auto')
        axes2[1, 0].set_title('Dominant Scale\nper Connection')
        axes2[1, 0].set_xlabel('Source ROI')
        axes2[1, 0].set_ylabel('Target ROI')
        cbar3 = plt.colorbar(im3, ax=axes2[1, 0], fraction=0.046, pad=0.04)
        cbar3.set_ticks(range(len(multiscale_results['scale_names'])))
        cbar3.set_ticklabels(multiscale_results['scale_names'])
        
        # Scale statistics
        scale_stats = multiscale_results['scale_statistics']
        scale_names = list(scale_stats.keys())
        proportions = [scale_stats[scale]['significance_proportion'] for scale in scale_names]
        
        bars = axes2[1, 1].bar(scale_names, proportions)
        axes2[1, 1].set_title('Significance Proportion\nby Scale')
        axes2[1, 1].set_xlabel('Temporal Scale')
        axes2[1, 1].set_ylabel('Proportion Significant')
        axes2[1, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, prop in zip(bars, proportions):
            height = bar.get_height()
            axes2[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                            f'{prop:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_combined_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


def test_multiscale_smte():
    """Test the multi-scale SMTE implementation."""
    
    print("Testing Multi-Scale SMTE Implementation")
    print("=" * 60)
    
    # Generate realistic test data with multiple time scales
    np.random.seed(42)
    n_regions = 12
    n_timepoints = 200  # Longer for multi-scale analysis
    TR = 2.0
    
    # Create ROI labels
    roi_labels = [
        'V1_L', 'V1_R',           # Visual (fast)
        'M1_L', 'M1_R',           # Motor (fast-intermediate)
        'S1_L', 'S1_R',           # Sensory (fast-intermediate)
        'DLPFC_L', 'DLPFC_R',     # Executive (intermediate-slow)
        'PCC', 'mPFC',            # Default mode (slow)
        'ACC', 'Insula'           # Salience (intermediate)
    ]
    
    # Generate time vector
    t = np.arange(n_timepoints) * TR
    
    # Create multi-scale signals
    data = []
    for i in range(n_regions):
        # Base slow oscillation (slow scale)
        slow_component = 0.8 * np.sin(2 * np.pi * 0.015 * t)
        
        # Intermediate oscillation (intermediate scale)
        intermediate_component = 0.5 * np.sin(2 * np.pi * 0.05 * t)
        
        # Fast oscillation (fast scale)
        fast_component = 0.3 * np.sin(2 * np.pi * 0.15 * t)
        
        # Noise
        noise = 0.4 * np.random.randn(n_timepoints)
        
        # Combine components with region-specific weights
        if 'V1' in roi_labels[i] or 'M1' in roi_labels[i] or 'S1' in roi_labels[i]:
            # Sensorimotor regions: more fast components
            signal = 1.2 * fast_component + 0.8 * intermediate_component + 0.4 * slow_component + noise
        elif 'DLPFC' in roi_labels[i] or 'ACC' in roi_labels[i] or 'Insula' in roi_labels[i]:
            # Executive/salience: more intermediate components
            signal = 0.6 * fast_component + 1.2 * intermediate_component + 0.8 * slow_component + noise
        else:
            # Default mode: more slow components
            signal = 0.4 * fast_component + 0.6 * intermediate_component + 1.2 * slow_component + noise
        
        data.append(signal)
    
    data = np.array(data)
    
    # Add scale-specific connectivity
    # Fast scale: V1 -> M1 (visual-motor)
    data[2, 1:] += 0.4 * data[0, :-1]  # V1_L -> M1_L (1 TR lag)
    
    # Intermediate scale: DLPFC -> ACC (executive control)
    data[10, 3:] += 0.3 * data[6, :-3]  # DLPFC_L -> ACC (3 TR lag)
    
    # Slow scale: PCC -> mPFC (default mode)
    data[9, 8:] += 0.5 * data[8, :-8]  # PCC -> mPFC (8 TR lag)
    
    # Standardize data
    scaler = StandardScaler()
    data = scaler.fit_transform(data.T).T
    
    # Define known networks
    known_networks = {
        'visual': [0, 1],
        'sensorimotor': [2, 3, 4, 5],
        'executive': [6, 7, 10],
        'default': [8, 9],
        'salience': [11]
    }
    
    # Test single-scale analysis (baseline)
    print("\n1. Testing Single-Scale Analysis (Baseline)")
    print("-" * 50)
    
    single_scale_smte = MultiScaleSMTE(
        use_multiscale_analysis=False,  # Disable multi-scale
        adaptive_mode='heuristic',
        use_network_correction=True,
        use_physiological_constraints=True,
        known_networks=known_networks,
        TR=TR,
        n_permutations=100,
        random_state=42
    )
    
    single_results = single_scale_smte.compute_multiscale_connectivity(
        data, roi_labels
    )
    
    print(f"Single-scale: {single_results['n_final_significant']} significant connections")
    
    # Test multi-scale analysis
    print("\n2. Testing Multi-Scale Analysis")
    print("-" * 40)
    
    multiscale_smte = MultiScaleSMTE(
        use_multiscale_analysis=True,   # Enable multi-scale
        scales_to_analyze=['fast', 'intermediate', 'slow'],
        adaptive_mode='heuristic',
        use_network_correction=True,
        use_physiological_constraints=True,
        known_networks=known_networks,
        TR=TR,
        n_permutations=100,
        random_state=42
    )
    
    multiscale_results = multiscale_smte.compute_multiscale_connectivity(
        data, roi_labels
    )
    
    # Analyze scale contributions
    scale_analysis = multiscale_smte.analyze_scale_contributions(
        multiscale_results, roi_labels
    )
    
    print(f"\nScale dominance analysis:")
    for scale_name, dominance_info in scale_analysis['scale_dominance'].items():
        prop = dominance_info['dominance_proportion']
        print(f"  {scale_name}: {prop:.1%} of connections")
    
    # Create visualizations
    multiscale_smte.create_multiscale_visualizations(
        multiscale_results, roi_labels
    )
    
    return multiscale_results, scale_analysis


if __name__ == "__main__":
    results, analysis = test_multiscale_smte()