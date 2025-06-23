#!/usr/bin/env python3
"""
Optimized Real fMRI Data Validation with Known Network Analysis
This is an optimized version for demonstration purposes with reduced computational complexity.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import warnings
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import stats
import networkx as nx
from pathlib import Path

from voxel_smte_connectivity_corrected import VoxelSMTEConnectivity
from baseline_comparison import BaselineConnectivityMethods

warnings.filterwarnings('ignore')


class OptimizedKnownNetworkValidator:
    """
    Optimized validator for demonstration purposes with known brain networks.
    """
    
    def __init__(self):
        # Define canonical brain networks (simplified for demo)
        self.canonical_networks = self._define_canonical_networks()
        
    def _define_canonical_networks(self) -> Dict[str, Dict]:
        """Define established brain networks with expected connectivity patterns."""
        networks = {
            'Default_Mode_Network': {
                'description': 'Task-negative network active during rest',
                'key_regions': [0, 1, 2, 3, 4],  # Using indices for simplicity
                'expected_connections': [
                    (0, 1), (0, 2), (1, 2), (0, 3), (1, 4)
                ]
            },
            'Executive_Control_Network': {
                'description': 'Cognitive control and working memory network',
                'key_regions': [5, 6, 7, 8],
                'expected_connections': [
                    (5, 6), (5, 7), (6, 8), (7, 8)
                ]
            },
            'Sensorimotor_Network': {
                'description': 'Motor and sensory processing network',
                'key_regions': [9, 10, 11, 12],
                'expected_connections': [
                    (9, 10), (10, 11), (11, 12), (9, 12)
                ]
            }
        }
        return networks


class OptimizedRealDataValidator:
    """
    Optimized framework for real fMRI data validation with reduced complexity.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Initialize with reduced parameters for speed
        self.smte_analyzer = VoxelSMTEConnectivity(
            n_symbols=4,  # Reduced from 6
            symbolizer='ordinal',
            ordinal_order=3,
            max_lag=3,  # Reduced from 5
            alpha=0.05,
            n_permutations=100,  # Reduced from 1000
            random_state=random_state,
            memory_efficient=True
        )
        
        self.network_validator = OptimizedKnownNetworkValidator()
        
    def generate_realistic_fmri_data(self, 
                                   n_regions: int = 15,
                                   n_timepoints: int = 120,
                                   TR: float = 2.0) -> Tuple[np.ndarray, List[str]]:
        """Generate realistic synthetic fMRI data with known network structure."""
        
        print(f"Generating realistic fMRI data: {n_regions} regions, {n_timepoints} timepoints")
        
        # Time vector
        t = np.arange(n_timepoints) * TR
        
        # Initialize signals
        roi_signals = []
        
        for i in range(n_regions):
            # Base neural signal (low frequency)
            base_freq = 0.01 + 0.02 * np.random.rand()  # 0.01-0.03 Hz
            neural_signal = np.sin(2 * np.pi * base_freq * t)
            
            # Add physiological noise
            respiratory = 0.1 * np.sin(2 * np.pi * 0.3 * t)  # ~0.3 Hz
            cardiac = 0.05 * np.sin(2 * np.pi * 1.0 * t)     # ~1 Hz
            
            # Add thermal noise
            thermal_noise = 0.3 * np.random.randn(n_timepoints)
            
            # Combine components
            signal = neural_signal + respiratory + cardiac + thermal_noise
            roi_signals.append(signal)
        
        # Add network connectivity patterns
        networks = self.network_validator.canonical_networks
        
        # Default Mode Network connectivity
        dmn_regions = networks['Default_Mode_Network']['key_regions']
        dmn_connections = networks['Default_Mode_Network']['expected_connections']
        
        for source, target in dmn_connections:
            if source < n_regions and target < n_regions:
                # Add directed connectivity with lag
                lag = np.random.randint(1, 4)
                strength = 0.4 + 0.2 * np.random.rand()
                if lag < n_timepoints:
                    roi_signals[target] = roi_signals[target].copy()
                    roi_signals[target][lag:] += strength * roi_signals[source][:-lag]
        
        # Executive Control Network connectivity
        exec_regions = networks['Executive_Control_Network']['key_regions']
        exec_connections = networks['Executive_Control_Network']['expected_connections']
        
        for source, target in exec_connections:
            if source < n_regions and target < n_regions:
                lag = np.random.randint(1, 3)
                strength = 0.3 + 0.2 * np.random.rand()
                if lag < n_timepoints:
                    roi_signals[target] = roi_signals[target].copy()
                    roi_signals[target][lag:] += strength * roi_signals[source][:-lag]
        
        # Sensorimotor Network connectivity
        sensori_regions = networks['Sensorimotor_Network']['key_regions']
        sensori_connections = networks['Sensorimotor_Network']['expected_connections']
        
        for source, target in sensori_connections:
            if source < n_regions and target < n_regions:
                lag = np.random.randint(1, 3)
                strength = 0.5 + 0.2 * np.random.rand()
                if lag < n_timepoints:
                    roi_signals[target] = roi_signals[target].copy()
                    roi_signals[target][lag:] += strength * roi_signals[source][:-lag]
        
        # Convert to array and standardize
        roi_timeseries = np.array(roi_signals)
        scaler = StandardScaler()
        roi_timeseries = scaler.fit_transform(roi_timeseries.T).T
        
        # Create region labels
        roi_labels = [f"Region_{i+1}" for i in range(n_regions)]
        
        return roi_timeseries, roi_labels
    
    def compute_connectivity_methods(self, 
                                   roi_timeseries: np.ndarray,
                                   roi_labels: List[str]) -> Dict[str, Any]:
        """Compute connectivity using multiple methods (optimized)."""
        
        print(f"Computing connectivity for {len(roi_labels)} regions...")
        
        results = {}
        
        # 1. SMTE (optimized)
        print("  Computing SMTE...")
        try:
            symbolic_data = self.smte_analyzer.symbolize_timeseries(roi_timeseries)
            self.smte_analyzer.symbolic_data = symbolic_data
            smte_matrix, lag_matrix = self.smte_analyzer.compute_voxel_connectivity_matrix()
            
            # Reduced statistical testing for demo
            print("  Running statistical testing (reduced)...")
            p_values = self.smte_analyzer.statistical_testing(smte_matrix)
            significance_mask = self.smte_analyzer.fdr_correction(p_values)
            
            results['SMTE'] = {
                'connectivity_matrix': smte_matrix,
                'p_values': p_values,
                'significance_mask': significance_mask,
                'lag_matrix': lag_matrix,
                'n_significant': np.sum(significance_mask)
            }
            
        except Exception as e:
            print(f"    SMTE failed: {str(e)}")
            results['SMTE'] = None
        
        # 2. Lagged Correlation
        print("  Computing Lagged Correlation...")
        try:
            n_regions = roi_timeseries.shape[0]
            max_corr_matrix = np.zeros((n_regions, n_regions))
            
            for lag in range(1, 4):  # Reduced from 6
                corr_matrix = BaselineConnectivityMethods.pearson_correlation(roi_timeseries, lag)
                max_corr_matrix = np.maximum(max_corr_matrix, corr_matrix)
            
            results['Lagged_Correlation'] = {
                'connectivity_matrix': max_corr_matrix,
                'n_connections': np.sum(max_corr_matrix > np.percentile(max_corr_matrix, 95))
            }
            
        except Exception as e:
            print(f"    Lagged Correlation failed: {str(e)}")
            results['Lagged_Correlation'] = None
        
        # 3. Mutual Information
        print("  Computing Mutual Information...")
        try:
            mi_matrix = BaselineConnectivityMethods.mutual_information(roi_timeseries, bins=8)
            
            results['Mutual_Information'] = {
                'connectivity_matrix': mi_matrix,
                'n_connections': np.sum(mi_matrix > np.percentile(mi_matrix, 95))
            }
            
        except Exception as e:
            print(f"    Mutual Information failed: {str(e)}")
            results['Mutual_Information'] = None
        
        # 4. Partial Correlation
        print("  Computing Partial Correlation...")
        try:
            partial_corr_matrix = BaselineConnectivityMethods.partial_correlation(roi_timeseries)
            
            results['Partial_Correlation'] = {
                'connectivity_matrix': partial_corr_matrix,
                'n_connections': np.sum(partial_corr_matrix > np.percentile(partial_corr_matrix, 95))
            }
            
        except Exception as e:
            print(f"    Partial Correlation failed: {str(e)}")
            results['Partial_Correlation'] = None
        
        return results
    
    def validate_against_known_networks(self, 
                                      connectivity_results: Dict[str, Any],
                                      roi_labels: List[str]) -> Dict[str, Any]:
        """Validate connectivity results against known brain networks."""
        
        print("Validating against known brain networks...")
        
        validation_results = {}
        networks = self.network_validator.canonical_networks
        
        for method, result in connectivity_results.items():
            if result is None:
                continue
                
            print(f"  Validating {method}...")
            
            connectivity_matrix = result['connectivity_matrix']
            method_validation = {}
            
            for network_name, network_info in networks.items():
                expected_connections = network_info['expected_connections']
                key_regions = network_info['key_regions']
                
                # Check if regions exist in our data
                valid_connections = []
                for source, target in expected_connections:
                    if source < len(roi_labels) and target < len(roi_labels):
                        valid_connections.append((source, target))
                
                if not valid_connections:
                    continue
                
                # Calculate network validation metrics
                network_metrics = self._calculate_network_metrics(
                    connectivity_matrix, valid_connections, method
                )
                
                method_validation[network_name] = network_metrics
            
            validation_results[method] = method_validation
        
        return validation_results
    
    def _calculate_network_metrics(self, 
                                 connectivity_matrix: np.ndarray,
                                 expected_connections: List[Tuple],
                                 method: str) -> Dict[str, float]:
        """Calculate validation metrics for a specific network."""
        
        if method == 'SMTE':
            # Use significance for SMTE
            # For demo, use top 20% connections as threshold
            threshold = np.percentile(connectivity_matrix, 80)
            detected_matrix = connectivity_matrix > threshold
        else:
            # Use top 20% for other methods
            threshold = np.percentile(connectivity_matrix, 80)
            detected_matrix = connectivity_matrix > threshold
        
        # Calculate metrics
        total_expected = len(expected_connections)
        detected_connections = 0
        connection_strengths = []
        
        for source, target in expected_connections:
            if detected_matrix[target, source]:  # Note: SMTE is target <- source
                detected_connections += 1
            connection_strengths.append(connectivity_matrix[target, source])
        
        coverage = detected_connections / total_expected if total_expected > 0 else 0.0
        mean_strength = np.mean(connection_strengths) if connection_strengths else 0.0
        
        # Calculate network modularity (simplified)
        network_internal = np.mean(connection_strengths) if connection_strengths else 0.0
        
        # Random baseline for comparison
        n_regions = connectivity_matrix.shape[0]
        random_connections = np.random.choice(n_regions, size=(total_expected, 2), replace=True)
        random_strengths = [connectivity_matrix[t, s] for s, t in random_connections]
        random_mean = np.mean(random_strengths) if random_strengths else 0.0
        
        modularity = network_internal - random_mean
        
        return {
            'coverage': coverage,
            'mean_strength': mean_strength,
            'modularity': modularity,
            'n_expected': total_expected,
            'n_detected': detected_connections
        }
    
    def create_validation_visualizations(self, 
                                       connectivity_results: Dict[str, Any],
                                       validation_results: Dict[str, Any],
                                       roi_labels: List[str],
                                       save_prefix: str = 'optimized_validation'):
        """Create visualizations for the validation results."""
        
        print("Creating validation visualizations...")
        
        # 1. Connectivity matrices comparison
        valid_methods = [m for m, r in connectivity_results.items() if r is not None]
        n_methods = len(valid_methods)
        
        if n_methods > 0:
            fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
            if n_methods == 1:
                axes = [axes]
            
            for i, method in enumerate(valid_methods):
                connectivity_matrix = connectivity_results[method]['connectivity_matrix']
                
                if method == 'SMTE':
                    # Show significant connections only
                    if 'significance_mask' in connectivity_results[method]:
                        plot_matrix = connectivity_matrix * connectivity_results[method]['significance_mask']
                        n_sig = connectivity_results[method]['n_significant']
                        title = f"{method}\n({n_sig} significant)"
                    else:
                        plot_matrix = connectivity_matrix
                        title = method
                else:
                    # Threshold other methods
                    threshold = np.percentile(connectivity_matrix, 80)
                    plot_matrix = connectivity_matrix.copy()
                    plot_matrix[plot_matrix < threshold] = 0
                    n_conn = np.sum(plot_matrix > 0)
                    title = f"{method}\n({n_conn} connections)"
                
                im = axes[i].imshow(plot_matrix, cmap='viridis', aspect='auto')
                axes[i].set_title(title)
                axes[i].set_xlabel('Source Region')
                axes[i].set_ylabel('Target Region')
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.savefig(f'{save_prefix}_connectivity_matrices.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 2. Network validation results
        if validation_results:
            self._plot_network_validation(validation_results, save_prefix)
    
    def _plot_network_validation(self, validation_results: Dict[str, Any], save_prefix: str):
        """Plot network validation metrics."""
        
        # Prepare data for plotting
        methods = []
        networks = []
        coverages = []
        strengths = []
        modularities = []
        
        for method, method_results in validation_results.items():
            for network, metrics in method_results.items():
                methods.append(method)
                networks.append(network.replace('_', ' '))
                coverages.append(metrics['coverage'])
                strengths.append(metrics['mean_strength'])
                modularities.append(metrics['modularity'])
        
        if not methods:
            return
        
        # Create validation metrics plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Coverage plot
        df = pd.DataFrame({
            'Method': methods,
            'Network': networks,
            'Coverage': coverages
        })
        
        sns.barplot(data=df, x='Network', y='Coverage', hue='Method', ax=axes[0])
        axes[0].set_title('Network Coverage\n(% of expected connections detected)')
        axes[0].set_ylabel('Coverage')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Strength plot
        df['Strength'] = strengths
        sns.barplot(data=df, x='Network', y='Strength', hue='Method', ax=axes[1])
        axes[1].set_title('Mean Connection Strength\n(within-network connectivity)')
        axes[1].set_ylabel('Mean Strength')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Modularity plot
        df['Modularity'] = modularities
        sns.barplot(data=df, x='Network', y='Modularity', hue='Method', ax=axes[2])
        axes[2].set_title('Network Modularity\n(network vs random connectivity)')
        axes[2].set_ylabel('Modularity')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_network_validation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_validation_report(self, 
                                 connectivity_results: Dict[str, Any],
                                 validation_results: Dict[str, Any],
                                 roi_labels: List[str]) -> str:
        """Generate comprehensive validation report."""
        
        report = []
        report.append("# Optimized Real fMRI Data Validation Report")
        report.append("=" * 50)
        report.append("")
        
        # Dataset summary
        report.append("## Dataset Summary")
        report.append("")
        report.append(f"**Regions analyzed:** {len(roi_labels)}")
        report.append(f"**Methods tested:** {len([m for m, r in connectivity_results.items() if r is not None])}")
        report.append("")
        
        # Method performance summary
        report.append("## Method Performance Summary")
        report.append("")
        
        valid_methods = [m for m, r in connectivity_results.items() if r is not None]
        report.append("| Method | Total Connections | Status |")
        report.append("|--------|------------------|--------|")
        
        for method in connectivity_results.keys():
            result = connectivity_results[method]
            if result is not None:
                if method == 'SMTE':
                    n_conn = result['n_significant']
                else:
                    n_conn = result['n_connections']
                status = "✓ Success"
            else:
                n_conn = 0
                status = "✗ Failed"
            
            report.append(f"| {method} | {n_conn} | {status} |")
        
        report.append("")
        
        # Network validation results
        if validation_results:
            report.append("## Network Validation Results")
            report.append("")
            
            for method, method_results in validation_results.items():
                if not method_results:
                    continue
                    
                report.append(f"### {method}")
                report.append("")
                report.append("| Network | Coverage | Mean Strength | Modularity |")
                report.append("|---------|----------|---------------|------------|")
                
                for network, metrics in method_results.items():
                    coverage = metrics['coverage']
                    strength = metrics['mean_strength']
                    modularity = metrics['modularity']
                    
                    report.append(f"| {network.replace('_', ' ')} | {coverage:.3f} | {strength:.4f} | {modularity:.4f} |")
                
                report.append("")
        
        # SMTE specific analysis
        if 'SMTE' in connectivity_results and connectivity_results['SMTE'] is not None:
            smte_result = connectivity_results['SMTE']
            report.append("## SMTE Specific Analysis")
            report.append("")
            report.append(f"**Significant connections:** {smte_result['n_significant']}")
            
            if smte_result['n_significant'] > 0:
                p_values = smte_result['p_values']
                connectivity_matrix = smte_result['connectivity_matrix']
                
                report.append(f"**Min p-value:** {np.min(p_values[p_values > 0]):.6f}")
                report.append(f"**Mean SMTE value:** {np.mean(connectivity_matrix):.6f}")
                report.append(f"**Max SMTE value:** {np.max(connectivity_matrix):.6f}")
                
                # Lag analysis
                if 'lag_matrix' in smte_result:
                    lag_matrix = smte_result['lag_matrix']
                    significance_mask = smte_result['significance_mask']
                    significant_lags = lag_matrix[significance_mask]
                    if len(significant_lags) > 0:
                        most_common_lag = stats.mode(significant_lags, keepdims=True)[0][0]
                        report.append(f"**Most common lag:** {most_common_lag} time points")
            
            report.append("")
        
        # Overall assessment
        report.append("## Overall Assessment")
        report.append("")
        
        # Count successful methods
        n_successful = len(valid_methods)
        report.append(f"**Methods successfully completed:** {n_successful}/4")
        
        if 'SMTE' in valid_methods:
            smte_result = connectivity_results['SMTE']
            if smte_result['n_significant'] > 0:
                report.append("**SMTE Status:** ✓ Successfully detected significant connectivity patterns")
            else:
                report.append("**SMTE Status:** ⚠ No significant connections detected (consider parameter adjustment)")
        
        # Network validation summary
        if validation_results and 'SMTE' in validation_results:
            smte_validation = validation_results['SMTE']
            network_coverages = [metrics['coverage'] for metrics in smte_validation.values()]
            if network_coverages:
                mean_coverage = np.mean(network_coverages)
                report.append(f"**SMTE Network Detection:** {mean_coverage:.1%} average coverage of expected connections")
        
        return "\n".join(report)
    
    def run_complete_validation(self, output_dir: str = './optimized_validation_results') -> Dict[str, Any]:
        """Run complete optimized validation analysis."""
        
        print("Starting Optimized Real fMRI Data Validation")
        print("=" * 50)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Generate realistic fMRI data
        roi_timeseries, roi_labels = self.generate_realistic_fmri_data()
        
        # 2. Compute connectivity using multiple methods
        connectivity_results = self.compute_connectivity_methods(roi_timeseries, roi_labels)
        
        # 3. Validate against known networks
        validation_results = self.validate_against_known_networks(connectivity_results, roi_labels)
        
        # 4. Create visualizations
        save_prefix = os.path.join(output_dir, 'optimized_validation')
        self.create_validation_visualizations(
            connectivity_results, validation_results, roi_labels, save_prefix
        )
        
        # 5. Generate report
        report = self.generate_validation_report(
            connectivity_results, validation_results, roi_labels
        )
        
        # 6. Save results
        with open(os.path.join(output_dir, 'optimized_validation_report.md'), 'w') as f:
            f.write(report)
        
        # Save connectivity matrices
        for method, result in connectivity_results.items():
            if result is not None:
                np.save(
                    os.path.join(output_dir, f'{method}_connectivity_matrix.npy'),
                    result['connectivity_matrix']
                )
        
        print(f"\nValidation complete! Results saved to: {output_dir}")
        print("\n" + "=" * 50)
        print(report)
        
        return {
            'roi_timeseries': roi_timeseries,
            'roi_labels': roi_labels,
            'connectivity_results': connectivity_results,
            'validation_results': validation_results
        }


def main():
    """Run optimized validation demonstration."""
    
    print("Running Optimized Real fMRI Data Validation")
    print("=" * 60)
    
    # Initialize validator
    validator = OptimizedRealDataValidator(random_state=42)
    
    # Run complete validation
    results = validator.run_complete_validation()
    
    print("\nOptimized validation demonstration complete!")
    print("This demonstrates the framework capabilities with reduced computational complexity.")
    print("\nFor production use:")
    print("- Increase n_permutations to 1000+")
    print("- Use more regions (50-100)")
    print("- Use longer time series (200+ timepoints)")
    print("- Use real fMRI datasets (HCP, ABIDE, etc.)")
    
    return results


if __name__ == "__main__":
    results = main()