#!/usr/bin/env python3
"""
Real fMRI Data Analysis Framework for SMTE Connectivity.
This script provides tools for analyzing actual fMRI datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nibabel as nib
import os
import warnings
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
from scipy import stats
import networkx as nx

from voxel_smte_connectivity_corrected import VoxelSMTEConnectivity
from baseline_comparison import BaselineConnectivityMethods

warnings.filterwarnings('ignore')


class RealFMRIAnalyzer:
    """
    Framework for analyzing real fMRI data with SMTE and baseline methods.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Initialize SMTE analyzer with research-grade parameters
        self.smte_analyzer = VoxelSMTEConnectivity(
            n_symbols=6,
            symbolizer='ordinal',
            ordinal_order=3,
            max_lag=5,
            alpha=0.01,
            n_permutations=1000,
            random_state=random_state,
            memory_efficient=True
        )
        
        self.baseline_methods = {
            'Pearson_Correlation': self._compute_pearson,
            'Lagged_Correlation': self._compute_lagged_correlation,
            'Mutual_Information': self._compute_mutual_information,
            'Partial_Correlation': self._compute_partial_correlation,
        }
        
    def load_fmri_dataset(self, fmri_path: str, mask_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load and validate real fMRI dataset.
        
        Parameters:
        -----------
        fmri_path : str
            Path to 4D fMRI NIfTI file
        mask_path : str, optional
            Path to brain mask
            
        Returns:
        --------
        Dict with dataset information
        """
        print(f"Loading fMRI dataset: {fmri_path}")
        
        # Load fMRI data
        if not os.path.exists(fmri_path):
            raise FileNotFoundError(f"fMRI file not found: {fmri_path}")
            
        fmri_img = nib.load(fmri_path)
        fmri_data = fmri_img.get_fdata()
        
        if len(fmri_data.shape) != 4:
            raise ValueError(f"Expected 4D fMRI data, got {len(fmri_data.shape)}D")
        
        # Load or create mask
        if mask_path and os.path.exists(mask_path):
            mask_img = nib.load(mask_path)
            mask = mask_img.get_fdata().astype(bool)
            print(f"Loaded brain mask: {mask_path}")
        else:
            # Create simple mask based on mean signal
            mean_signal = np.mean(fmri_data, axis=3)
            mask = mean_signal > (0.1 * np.max(mean_signal))
            print("Created automatic brain mask")
        
        # Basic validation
        n_voxels = np.sum(mask)
        n_timepoints = fmri_data.shape[3]
        
        print(f"Dataset info:")
        print(f"  - Spatial dimensions: {fmri_data.shape[:3]}")
        print(f"  - Time points: {n_timepoints}")
        print(f"  - Brain voxels: {n_voxels}")
        print(f"  - Voxel size: {fmri_img.header.get_zooms()[:3]} mm")
        
        # Check for minimum requirements
        if n_timepoints < 100:
            print(f"WARNING: Short time series ({n_timepoints} points). Recommend ≥100 for reliable SMTE.")
        
        if n_voxels > 10000:
            print(f"WARNING: Large dataset ({n_voxels} voxels). Consider ROI-based analysis or chunking.")
        
        dataset_info = {
            'fmri_data': fmri_data,
            'mask': mask,
            'n_voxels': n_voxels,
            'n_timepoints': n_timepoints,
            'affine': fmri_img.affine,
            'header': fmri_img.header,
            'file_path': fmri_path
        }
        
        return dataset_info
    
    def extract_roi_timeseries(self, dataset_info: Dict[str, Any], 
                              atlas_path: Optional[str] = None,
                              roi_coords: Optional[List[Tuple]] = None,
                              roi_radius: int = 3) -> Tuple[np.ndarray, List[str]]:
        """
        Extract ROI time series from fMRI data.
        
        Parameters:
        -----------
        dataset_info : Dict
            Dataset information from load_fmri_dataset
        atlas_path : str, optional
            Path to atlas file (NIfTI)
        roi_coords : List[Tuple], optional
            List of (x, y, z) coordinates for spherical ROIs
        roi_radius : int
            Radius for spherical ROIs (in voxels)
            
        Returns:
        --------
        Tuple of (roi_timeseries, roi_labels)
        """
        fmri_data = dataset_info['fmri_data']
        mask = dataset_info['mask']
        
        if atlas_path and os.path.exists(atlas_path):
            # Use atlas-based ROIs
            print(f"Using atlas: {atlas_path}")
            atlas_img = nib.load(atlas_path)
            atlas_data = atlas_img.get_fdata()
            
            # Get unique ROI labels
            roi_labels_unique = np.unique(atlas_data[atlas_data > 0])
            roi_timeseries = []
            roi_labels = []
            
            for roi_label in roi_labels_unique:
                roi_mask = (atlas_data == roi_label) & mask
                if np.sum(roi_mask) > 10:  # Minimum 10 voxels
                    roi_ts = np.mean(fmri_data[roi_mask], axis=0)
                    roi_timeseries.append(roi_ts)
                    roi_labels.append(f"ROI_{int(roi_label)}")
            
            roi_timeseries = np.array(roi_timeseries)
            
        elif roi_coords:
            # Use coordinate-based spherical ROIs
            print(f"Creating {len(roi_coords)} spherical ROIs")
            roi_timeseries = []
            roi_labels = []
            
            for i, (x, y, z) in enumerate(roi_coords):
                # Create spherical ROI
                xx, yy, zz = np.ogrid[:fmri_data.shape[0], :fmri_data.shape[1], :fmri_data.shape[2]]
                sphere_mask = ((xx - x)**2 + (yy - y)**2 + (zz - z)**2) <= roi_radius**2
                roi_mask = sphere_mask & mask
                
                if np.sum(roi_mask) > 5:
                    roi_ts = np.mean(fmri_data[roi_mask], axis=0)
                    roi_timeseries.append(roi_ts)
                    roi_labels.append(f"ROI_{i+1}_({x},{y},{z})")
            
            roi_timeseries = np.array(roi_timeseries)
            
        else:
            # Default: use random voxel sampling
            print("Using random voxel sampling (50 voxels)")
            voxel_coords = np.where(mask)
            n_voxels_total = len(voxel_coords[0])
            
            # Sample voxels
            n_sample = min(50, n_voxels_total)
            sample_indices = np.random.choice(n_voxels_total, size=n_sample, replace=False)
            
            roi_timeseries = []
            roi_labels = []
            
            for i in range(n_sample):
                idx = sample_indices[i]
                x, y, z = voxel_coords[0][idx], voxel_coords[1][idx], voxel_coords[2][idx]
                roi_ts = fmri_data[x, y, z, :]
                roi_timeseries.append(roi_ts)
                roi_labels.append(f"Voxel_{i+1}_({x},{y},{z})")
            
            roi_timeseries = np.array(roi_timeseries)
        
        # Preprocessing
        print(f"Extracted {len(roi_timeseries)} ROI time series")
        
        # Detrend and standardize
        from scipy.signal import detrend
        roi_timeseries = np.array([detrend(ts) for ts in roi_timeseries])
        
        scaler = StandardScaler()
        roi_timeseries = scaler.fit_transform(roi_timeseries.T).T
        
        return roi_timeseries, roi_labels
    
    def analyze_connectivity_patterns(self, 
                                    roi_timeseries: np.ndarray,
                                    roi_labels: List[str],
                                    methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze connectivity patterns using multiple methods.
        
        Parameters:
        -----------
        roi_timeseries : np.ndarray
            ROI time series data (n_rois, n_timepoints)
        roi_labels : List[str]
            ROI labels
        methods : List[str], optional
            Methods to use (default: all)
            
        Returns:
        --------
        Dict with connectivity results
        """
        if methods is None:
            methods = ['SMTE'] + list(self.baseline_methods.keys())
        
        print(f"Computing connectivity using {len(methods)} methods...")
        
        results = {}
        
        for method in methods:
            print(f"  Computing {method}...")
            
            try:
                if method == 'SMTE':
                    # Use SMTE
                    symbolic_data = self.smte_analyzer.symbolize_timeseries(roi_timeseries)
                    self.smte_analyzer.symbolic_data = symbolic_data
                    connectivity_matrix, lag_matrix = self.smte_analyzer.compute_voxel_connectivity_matrix()
                    
                    # Statistical testing (reduced permutations for speed)
                    print(f"    Running statistical testing...")
                    self.smte_analyzer.n_permutations = 500  # Reduced for real data
                    p_values = self.smte_analyzer.statistical_testing(connectivity_matrix)
                    significance_mask = self.smte_analyzer.fdr_correction(p_values)
                    
                    results[method] = {
                        'connectivity_matrix': connectivity_matrix,
                        'p_values': p_values,
                        'significance_mask': significance_mask,
                        'lag_matrix': lag_matrix,
                        'n_significant': np.sum(significance_mask)
                    }
                    
                else:
                    # Use baseline method
                    connectivity_matrix = self.baseline_methods[method](roi_timeseries)
                    
                    results[method] = {
                        'connectivity_matrix': connectivity_matrix,
                        'n_connections': np.sum(connectivity_matrix > np.median(connectivity_matrix))
                    }
                    
            except Exception as e:
                print(f"    WARNING: {method} failed: {str(e)}")
                results[method] = None
        
        return results
    
    def _compute_pearson(self, data: np.ndarray) -> np.ndarray:
        """Compute Pearson correlation matrix."""
        return BaselineConnectivityMethods.pearson_correlation(data)
    
    def _compute_lagged_correlation(self, data: np.ndarray) -> np.ndarray:
        """Compute maximum lagged correlation."""
        n_rois = data.shape[0]
        max_corr_matrix = np.zeros((n_rois, n_rois))
        
        for lag in range(1, 6):
            corr_matrix = BaselineConnectivityMethods.pearson_correlation(data, lag)
            max_corr_matrix = np.maximum(max_corr_matrix, corr_matrix)
        
        return max_corr_matrix
    
    def _compute_mutual_information(self, data: np.ndarray) -> np.ndarray:
        """Compute mutual information matrix."""
        return BaselineConnectivityMethods.mutual_information(data)
    
    def _compute_partial_correlation(self, data: np.ndarray) -> np.ndarray:
        """Compute partial correlation matrix."""
        return BaselineConnectivityMethods.partial_correlation(data)
    
    def analyze_network_properties(self, connectivity_results: Dict[str, Any],
                                 roi_labels: List[str]) -> Dict[str, Any]:
        """
        Analyze network properties for each connectivity method.
        """
        network_analysis = {}
        
        for method, result in connectivity_results.items():
            if result is None:
                continue
                
            print(f"Analyzing network properties for {method}...")
            
            connectivity_matrix = result['connectivity_matrix']
            
            if method == 'SMTE':
                # Use significance mask for SMTE
                significance_mask = result['significance_mask']
                binary_matrix = significance_mask.astype(int)
                weighted_matrix = connectivity_matrix * significance_mask
            else:
                # Use threshold for other methods
                threshold = np.percentile(connectivity_matrix, 95)
                binary_matrix = (connectivity_matrix > threshold).astype(int)
                weighted_matrix = connectivity_matrix * binary_matrix
            
            # Create network graph
            G = nx.from_numpy_array(binary_matrix, create_using=nx.DiGraph())
            
            # Basic properties
            properties = {
                'n_nodes': G.number_of_nodes(),
                'n_edges': G.number_of_edges(),
                'density': nx.density(G),
                'mean_connectivity_strength': np.mean(weighted_matrix[weighted_matrix > 0]) if np.any(weighted_matrix > 0) else 0.0
            }
            
            # Centrality measures (if graph has edges)
            if G.number_of_edges() > 0:
                properties['in_degree_centrality'] = nx.in_degree_centrality(G)
                properties['out_degree_centrality'] = nx.out_degree_centrality(G)
                
                # Find hub regions
                in_centrality = properties['in_degree_centrality']
                out_centrality = properties['out_degree_centrality']
                
                # Top 3 hubs
                top_in_hubs = sorted(in_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
                top_out_hubs = sorted(out_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
                
                properties['top_input_hubs'] = [(roi_labels[idx], score) for idx, score in top_in_hubs]
                properties['top_output_hubs'] = [(roi_labels[idx], score) for idx, score in top_out_hubs]
            
            network_analysis[method] = properties
        
        return network_analysis
    
    def create_connectivity_visualizations(self, 
                                         connectivity_results: Dict[str, Any],
                                         roi_labels: List[str],
                                         save_prefix: str = 'real_fmri_connectivity'):
        """
        Create visualizations for real fMRI connectivity analysis.
        """
        n_methods = len([r for r in connectivity_results.values() if r is not None])
        
        if n_methods == 0:
            print("No valid results to visualize")
            return
        
        # Create subplot layout
        cols = min(3, n_methods)
        rows = (n_methods + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if n_methods == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        method_idx = 0
        
        for method, result in connectivity_results.items():
            if result is None:
                continue
                
            row = method_idx // cols
            col = method_idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            connectivity_matrix = result['connectivity_matrix']
            
            if method == 'SMTE':
                # Show only significant connections
                significance_mask = result['significance_mask']
                plot_matrix = connectivity_matrix * significance_mask
                title = f"{method}\n({result['n_significant']} significant connections)"
            else:
                # Show thresholded connections
                threshold = np.percentile(connectivity_matrix, 95)
                plot_matrix = connectivity_matrix.copy()
                plot_matrix[plot_matrix < threshold] = 0
                n_connections = np.sum(plot_matrix > 0)
                title = f"{method}\n({n_connections} connections)"
            
            # Create heatmap
            im = ax.imshow(plot_matrix, cmap='viridis', aspect='auto')
            ax.set_title(title)
            ax.set_xlabel('Source ROI')
            ax.set_ylabel('Target ROI')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            method_idx += 1
        
        # Hide unused subplots
        for idx in range(method_idx, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create network comparison plot
        self._plot_network_comparison(connectivity_results, roi_labels, save_prefix)
    
    def _plot_network_comparison(self, connectivity_results: Dict[str, Any],
                               roi_labels: List[str], save_prefix: str):
        """Plot network properties comparison."""
        
        # Extract network metrics
        methods = []
        densities = []
        n_connections = []
        mean_strengths = []
        
        for method, result in connectivity_results.items():
            if result is None:
                continue
                
            methods.append(method)
            
            connectivity_matrix = result['connectivity_matrix']
            
            if method == 'SMTE':
                significance_mask = result['significance_mask']
                binary_matrix = significance_mask
                weighted_matrix = connectivity_matrix * significance_mask
            else:
                threshold = np.percentile(connectivity_matrix, 95)
                binary_matrix = connectivity_matrix > threshold
                weighted_matrix = connectivity_matrix * binary_matrix
            
            # Calculate metrics
            n_nodes = connectivity_matrix.shape[0]
            n_edges = np.sum(binary_matrix)
            density = n_edges / (n_nodes * (n_nodes - 1))
            mean_strength = np.mean(weighted_matrix[weighted_matrix > 0]) if np.any(weighted_matrix > 0) else 0.0
            
            densities.append(density)
            n_connections.append(n_edges)
            mean_strengths.append(mean_strength)
        
        # Create comparison plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Network density
        axes[0].bar(methods, densities)
        axes[0].set_title('Network Density')
        axes[0].set_ylabel('Density')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Number of connections
        axes[1].bar(methods, n_connections)
        axes[1].set_title('Number of Connections')
        axes[1].set_ylabel('Count')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Mean connection strength
        axes[2].bar(methods, mean_strengths)
        axes[2].set_title('Mean Connection Strength')
        axes[2].set_ylabel('Strength')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_network_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_real_data_report(self, 
                                connectivity_results: Dict[str, Any],
                                network_analysis: Dict[str, Any],
                                roi_labels: List[str],
                                dataset_info: Dict[str, Any]) -> str:
        """Generate comprehensive report for real fMRI data analysis."""
        
        report = []
        report.append("# Real fMRI Data Connectivity Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Dataset Information
        report.append("## Dataset Information")
        report.append("")
        report.append(f"**File:** {os.path.basename(dataset_info['file_path'])}")
        report.append(f"**Spatial Dimensions:** {dataset_info['fmri_data'].shape[:3]}")
        report.append(f"**Time Points:** {dataset_info['n_timepoints']}")
        report.append(f"**Brain Voxels:** {dataset_info['n_voxels']}")
        report.append(f"**ROIs Analyzed:** {len(roi_labels)}")
        report.append("")
        
        # Connectivity Results Summary
        report.append("## Connectivity Analysis Results")
        report.append("")
        
        valid_methods = [method for method, result in connectivity_results.items() if result is not None]
        
        report.append("| Method | Connections | Density | Mean Strength |")
        report.append("|--------|-------------|---------|---------------|")
        
        for method in valid_methods:
            result = connectivity_results[method]
            network_props = network_analysis.get(method, {})
            
            if method == 'SMTE':
                n_connections = result['n_significant']
            else:
                connectivity_matrix = result['connectivity_matrix']
                threshold = np.percentile(connectivity_matrix, 95)
                n_connections = np.sum(connectivity_matrix > threshold)
            
            density = network_props.get('density', 0.0)
            mean_strength = network_props.get('mean_connectivity_strength', 0.0)
            
            report.append(f"| {method} | {n_connections} | {density:.4f} | {mean_strength:.4f} |")
        
        report.append("")
        
        # SMTE Specific Results
        if 'SMTE' in connectivity_results and connectivity_results['SMTE'] is not None:
            smte_result = connectivity_results['SMTE']
            report.append("## SMTE Specific Analysis")
            report.append("")
            report.append(f"**Significant Connections:** {smte_result['n_significant']}")
            
            # Statistical summary
            p_values = smte_result['p_values']
            connectivity_matrix = smte_result['connectivity_matrix']
            
            report.append(f"**Min p-value:** {np.min(p_values[p_values > 0]):.6f}")
            report.append(f"**Mean SMTE value:** {np.mean(connectivity_matrix):.6f}")
            report.append(f"**Max SMTE value:** {np.max(connectivity_matrix):.6f}")
            
            # Lag analysis
            lag_matrix = smte_result['lag_matrix']
            significant_lags = lag_matrix[smte_result['significance_mask']]
            if len(significant_lags) > 0:
                report.append(f"**Most common lag:** {stats.mode(significant_lags, keepdims=True)[0][0]} time points")
            
            report.append("")
        
        # Hub Analysis
        report.append("## Network Hub Analysis")
        report.append("")
        
        for method, network_props in network_analysis.items():
            if 'top_input_hubs' in network_props and 'top_output_hubs' in network_props:
                report.append(f"### {method}")
                report.append("")
                report.append("**Top Input Hubs:**")
                for roi_name, centrality in network_props['top_input_hubs']:
                    report.append(f"- {roi_name}: {centrality:.4f}")
                
                report.append("")
                report.append("**Top Output Hubs:**")
                for roi_name, centrality in network_props['top_output_hubs']:
                    report.append(f"- {roi_name}: {centrality:.4f}")
                
                report.append("")
        
        # Recommendations
        report.append("## Analysis Recommendations")
        report.append("")
        
        if dataset_info['n_timepoints'] < 150:
            report.append("- **Consider longer scans:** Short time series may limit SMTE reliability")
        
        if len(roi_labels) < 20:
            report.append("- **Expand ROI coverage:** More regions provide richer connectivity patterns")
        
        if 'SMTE' in connectivity_results and connectivity_results['SMTE'] is not None:
            n_sig = connectivity_results['SMTE']['n_significant']
            if n_sig == 0:
                report.append("- **No significant SMTE connections:** Consider adjusting significance threshold or parameters")
            elif n_sig > len(roi_labels) * 5:
                report.append("- **Many significant connections:** Consider more stringent significance threshold")
        
        return "\n".join(report)
    
    def run_complete_real_data_analysis(self,
                                      fmri_path: str,
                                      mask_path: Optional[str] = None,
                                      atlas_path: Optional[str] = None,
                                      roi_coords: Optional[List[Tuple]] = None,
                                      output_dir: str = './real_fmri_results') -> Dict[str, Any]:
        """
        Run complete analysis on real fMRI data.
        
        Parameters:
        -----------
        fmri_path : str
            Path to fMRI NIfTI file
        mask_path : str, optional
            Path to brain mask
        atlas_path : str, optional
            Path to atlas file
        roi_coords : List[Tuple], optional
            ROI coordinates
        output_dir : str
            Output directory
            
        Returns:
        --------
        Dict with complete analysis results
        """
        print("Starting complete real fMRI data analysis...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Load dataset
        dataset_info = self.load_fmri_dataset(fmri_path, mask_path)
        
        # 2. Extract ROI time series
        roi_timeseries, roi_labels = self.extract_roi_timeseries(
            dataset_info, atlas_path, roi_coords
        )
        
        # 3. Analyze connectivity
        connectivity_results = self.analyze_connectivity_patterns(
            roi_timeseries, roi_labels
        )
        
        # 4. Network analysis
        network_analysis = self.analyze_network_properties(
            connectivity_results, roi_labels
        )
        
        # 5. Create visualizations
        save_prefix = os.path.join(output_dir, 'real_fmri')
        self.create_connectivity_visualizations(
            connectivity_results, roi_labels, save_prefix
        )
        
        # 6. Generate report
        report = self.generate_real_data_report(
            connectivity_results, network_analysis, roi_labels, dataset_info
        )
        
        # 7. Save results
        with open(os.path.join(output_dir, 'real_fmri_analysis_report.md'), 'w') as f:
            f.write(report)
        
        # Save connectivity matrices
        for method, result in connectivity_results.items():
            if result is not None:
                np.save(
                    os.path.join(output_dir, f'{method}_connectivity_matrix.npy'),
                    result['connectivity_matrix']
                )
        
        print(f"\nAnalysis complete! Results saved to: {output_dir}")
        print("\n" + "="*50)
        print(report)
        
        return {
            'dataset_info': dataset_info,
            'roi_labels': roi_labels,
            'connectivity_results': connectivity_results,
            'network_analysis': network_analysis
        }


def demo_with_synthetic_fmri():
    """
    Demonstrate the framework with synthetic fMRI-like data.
    """
    print("Creating demo with synthetic fMRI-like data...")
    
    # Create realistic synthetic fMRI data
    n_rois = 20
    n_timepoints = 200
    TR = 2.0  # seconds
    
    # Generate realistic fMRI signals
    t = np.arange(n_timepoints) * TR
    
    # Create base signals with fMRI characteristics
    roi_signals = []
    
    for i in range(n_rois):
        # Low frequency component (task/rest)
        base_freq = 0.01 + 0.02 * np.random.rand()  # 0.01-0.03 Hz
        base_signal = np.sin(2 * np.pi * base_freq * t)
        
        # Add physiological noise
        respiratory = 0.1 * np.sin(2 * np.pi * 0.3 * t)  # ~0.3 Hz
        cardiac = 0.05 * np.sin(2 * np.pi * 1.0 * t)     # ~1 Hz
        
        # Add white noise
        noise = 0.5 * np.random.randn(n_timepoints)
        
        # Combine
        signal = base_signal + respiratory + cardiac + noise
        roi_signals.append(signal)
    
    # Add some connectivity patterns
    # Default mode network-like connectivity
    roi_signals[1] += 0.3 * roi_signals[0]  # PCC -> mPFC
    roi_signals[2] += 0.2 * roi_signals[0]  # PCC -> Angular
    
    # Executive network
    roi_signals[5] += 0.4 * roi_signals[4]  # DLPFC -> FEF
    
    roi_timeseries = np.array(roi_signals)
    roi_labels = [f"ROI_{i+1}" for i in range(n_rois)]
    
    # Analyze with framework
    analyzer = RealFMRIAnalyzer()
    connectivity_results = analyzer.analyze_connectivity_patterns(roi_timeseries, roi_labels)
    network_analysis = analyzer.analyze_network_properties(connectivity_results, roi_labels)
    
    # Create visualizations
    analyzer.create_connectivity_visualizations(
        connectivity_results, roi_labels, 'demo_synthetic_fmri'
    )
    
    print("Demo analysis complete!")
    return connectivity_results, network_analysis


if __name__ == "__main__":
    # Run demo with synthetic data
    print("Running Real fMRI Analysis Framework Demo")
    print("=" * 50)
    
    connectivity_results, network_analysis = demo_with_synthetic_fmri()
    
    print("\nFramework ready for real fMRI data!")
    print("To use with your data:")
    print("  analyzer = RealFMRIAnalyzer()")
    print("  results = analyzer.run_complete_real_data_analysis(")
    print("      fmri_path='your_fmri_data.nii.gz',")
    print("      mask_path='your_mask.nii.gz',")
    print("      output_dir='your_results/'")
    print("  )")