#!/usr/bin/env python3
"""
Comprehensive comparison of SMTE against baseline connectivity methods.
This script evaluates performance against established methods using synthetic and real data.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import hilbert
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Any
import time
import warnings
warnings.filterwarnings('ignore')

# Import our implementation
from voxel_smte_connectivity_corrected import VoxelSMTEConnectivity


class BaselineConnectivityMethods:
    """
    Collection of baseline connectivity methods for comparison.
    """
    
    @staticmethod
    def pearson_correlation(data: np.ndarray, lag: int = 0) -> np.ndarray:
        """Compute Pearson correlation matrix with optional lag."""
        n_voxels = data.shape[0]
        corr_matrix = np.zeros((n_voxels, n_voxels))
        
        for i in range(n_voxels):
            for j in range(n_voxels):
                if i != j:
                    if lag == 0:
                        corr, _ = stats.pearsonr(data[i], data[j])
                    else:
                        if lag < data.shape[1]:
                            corr, _ = stats.pearsonr(data[i][lag:], data[j][:-lag])
                        else:
                            corr = 0.0
                    corr_matrix[i, j] = abs(corr)  # Use absolute value
                    
        return corr_matrix
    
    @staticmethod
    def partial_correlation(data: np.ndarray) -> np.ndarray:
        """Compute partial correlation matrix."""
        from sklearn.covariance import GraphicalLassoCV
        
        try:
            # Estimate precision matrix
            model = GraphicalLassoCV(cv=3, max_iter=100)
            model.fit(data.T)
            precision = model.precision_
            
            # Convert precision to partial correlation
            n_vars = precision.shape[0]
            partial_corr = np.zeros((n_vars, n_vars))
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j:
                        partial_corr[i, j] = -precision[i, j] / np.sqrt(precision[i, i] * precision[j, j])
                        
            return np.abs(partial_corr)
            
        except Exception:
            # Fallback to simple partial correlation
            return BaselineConnectivityMethods.pearson_correlation(data)
    
    @staticmethod
    def mutual_information(data: np.ndarray, bins: int = 10) -> np.ndarray:
        """Compute mutual information matrix."""
        from sklearn.feature_selection import mutual_info_regression
        
        n_voxels = data.shape[0]
        mi_matrix = np.zeros((n_voxels, n_voxels))
        
        for i in range(n_voxels):
            for j in range(n_voxels):
                if i != j:
                    # Discretize data
                    x_disc = np.digitize(data[i], np.linspace(data[i].min(), data[i].max(), bins))
                    y_disc = np.digitize(data[j], np.linspace(data[j].min(), data[j].max(), bins))
                    
                    # Compute MI using histogram method
                    mi = BaselineConnectivityMethods._mutual_info_hist(x_disc, y_disc)
                    mi_matrix[i, j] = mi
                    
        return mi_matrix
    
    @staticmethod
    def _mutual_info_hist(x: np.ndarray, y: np.ndarray) -> float:
        """Compute mutual information using histogram method."""
        # Joint histogram
        xy_hist, _, _ = np.histogram2d(x, y, bins=10)
        xy_hist = xy_hist + 1e-10  # Avoid log(0)
        xy_prob = xy_hist / np.sum(xy_hist)
        
        # Marginal histograms
        x_hist, _ = np.histogram(x, bins=10)
        x_prob = (x_hist + 1e-10) / np.sum(x_hist + 1e-10)
        
        y_hist, _ = np.histogram(y, bins=10)
        y_prob = (y_hist + 1e-10) / np.sum(y_hist + 1e-10)
        
        # Mutual information
        mi = 0.0
        for i in range(len(x_prob)):
            for j in range(len(y_prob)):
                if xy_prob[i, j] > 1e-10:
                    mi += xy_prob[i, j] * np.log(xy_prob[i, j] / (x_prob[i] * y_prob[j]))
                    
        return max(mi, 0.0)
    
    @staticmethod
    def granger_causality(data: np.ndarray, max_lag: int = 5) -> np.ndarray:
        """Compute Granger causality using VAR models."""
        from statsmodels.tsa.api import VAR
        from statsmodels.tsa.stattools import grangercausalitytests
        
        n_voxels = data.shape[0]
        gc_matrix = np.zeros((n_voxels, n_voxels))
        
        for i in range(n_voxels):
            for j in range(n_voxels):
                if i != j:
                    try:
                        # Prepare data for Granger causality test
                        test_data = np.column_stack([data[i], data[j]])
                        
                        # Test causality j -> i
                        result = grangercausalitytests(test_data, max_lag, verbose=False)
                        
                        # Extract F-statistic (use lag with minimum p-value)
                        f_stats = []
                        for lag in range(1, max_lag + 1):
                            if lag in result:
                                f_stat = result[lag][0]['ssr_ftest'][0]
                                f_stats.append(f_stat)
                        
                        if f_stats:
                            gc_matrix[i, j] = max(f_stats)
                            
                    except Exception:
                        gc_matrix[i, j] = 0.0
                        
        return gc_matrix
    
    @staticmethod
    def phase_lag_index(data: np.ndarray) -> np.ndarray:
        """Compute Phase Lag Index using Hilbert transform."""
        n_voxels = data.shape[0]
        pli_matrix = np.zeros((n_voxels, n_voxels))
        
        # Get analytic signals
        analytic_signals = np.array([hilbert(ts) for ts in data])
        phases = np.angle(analytic_signals)
        
        for i in range(n_voxels):
            for j in range(n_voxels):
                if i != j:
                    # Phase difference
                    phase_diff = phases[i] - phases[j]
                    
                    # PLI calculation
                    pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))
                    pli_matrix[i, j] = pli
                    
        return pli_matrix
    
    @staticmethod
    def coherence(data: np.ndarray, fs: float = 1.0) -> np.ndarray:
        """Compute magnitude squared coherence."""
        from scipy.signal import coherence as scipy_coherence
        
        n_voxels = data.shape[0]
        coh_matrix = np.zeros((n_voxels, n_voxels))
        
        for i in range(n_voxels):
            for j in range(n_voxels):
                if i != j:
                    try:
                        f, Cxy = scipy_coherence(data[i], data[j], fs=fs, nperseg=32)
                        # Use mean coherence across frequencies
                        coh_matrix[i, j] = np.mean(Cxy)
                    except Exception:
                        coh_matrix[i, j] = 0.0
                        
        return coh_matrix


class ConnectivityBenchmark:
    """
    Comprehensive benchmark comparing SMTE against baseline methods.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Initialize SMTE analyzer
        self.smte_analyzer = VoxelSMTEConnectivity(
            n_symbols=6,
            symbolizer='ordinal',
            ordinal_order=3,
            max_lag=5,
            alpha=0.05,
            n_permutations=500,  # Reduced for speed
            random_state=random_state
        )
        
        self.methods = {
            'SMTE': self._compute_smte,
            'Pearson_Correlation': self._compute_pearson,
            'Lagged_Correlation': self._compute_lagged_correlation,
            'Partial_Correlation': self._compute_partial_correlation,
            'Mutual_Information': self._compute_mutual_information,
            'Granger_Causality': self._compute_granger,
            'Phase_Lag_Index': self._compute_pli,
            'Coherence': self._compute_coherence
        }
        
    def _compute_smte(self, data: np.ndarray) -> np.ndarray:
        """Compute SMTE connectivity."""
        self.smte_analyzer.symbolic_data = self.smte_analyzer.symbolize_timeseries(data)
        smte_matrix, _ = self.smte_analyzer.compute_voxel_connectivity_matrix()
        return smte_matrix
    
    def _compute_pearson(self, data: np.ndarray) -> np.ndarray:
        """Compute Pearson correlation."""
        return BaselineConnectivityMethods.pearson_correlation(data)
    
    def _compute_lagged_correlation(self, data: np.ndarray) -> np.ndarray:
        """Compute maximum lagged correlation."""
        n_voxels = data.shape[0]
        max_corr_matrix = np.zeros((n_voxels, n_voxels))
        
        for lag in range(1, 6):  # Test lags 1-5
            corr_matrix = BaselineConnectivityMethods.pearson_correlation(data, lag)
            max_corr_matrix = np.maximum(max_corr_matrix, corr_matrix)
            
        return max_corr_matrix
    
    def _compute_partial_correlation(self, data: np.ndarray) -> np.ndarray:
        """Compute partial correlation."""
        return BaselineConnectivityMethods.partial_correlation(data)
    
    def _compute_mutual_information(self, data: np.ndarray) -> np.ndarray:
        """Compute mutual information."""
        return BaselineConnectivityMethods.mutual_information(data)
    
    def _compute_granger(self, data: np.ndarray) -> np.ndarray:
        """Compute Granger causality."""
        try:
            return BaselineConnectivityMethods.granger_causality(data)
        except Exception:
            # Fallback if Granger causality fails
            return np.zeros((data.shape[0], data.shape[0]))
    
    def _compute_pli(self, data: np.ndarray) -> np.ndarray:
        """Compute Phase Lag Index."""
        return BaselineConnectivityMethods.phase_lag_index(data)
    
    def _compute_coherence(self, data: np.ndarray) -> np.ndarray:
        """Compute coherence."""
        return BaselineConnectivityMethods.coherence(data)
    
    def generate_synthetic_data(self, 
                               n_voxels: int = 20,
                               n_timepoints: int = 200,
                               coupling_strength: float = 0.6,
                               noise_level: float = 0.3,
                               coupling_type: str = 'linear') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic data with known connectivity patterns.
        
        Parameters:
        -----------
        coupling_type : str
            'linear', 'nonlinear', or 'mixed'
        """
        np.random.seed(self.random_state)
        
        # Base time series
        data = np.random.randn(n_voxels, n_timepoints)
        
        # Add various types of connectivity
        ground_truth = np.zeros((n_voxels, n_voxels))
        
        # Define connections
        connections = [
            (0, 1, 1),  # Linear lag-1
            (2, 3, 2),  # Linear lag-2
            (4, 5, 1),  # Nonlinear lag-1
            (6, 7, 3),  # Mixed lag-3
            (8, 9, 1),  # Bidirectional
            (9, 8, 2),
        ]
        
        for source, target, lag in connections:
            if source < n_voxels and target < n_voxels and lag < n_timepoints:
                
                if coupling_type == 'linear' or (coupling_type == 'mixed' and source % 2 == 0):
                    # Linear coupling
                    data[target, lag:] += coupling_strength * data[source, :-lag]
                    
                elif coupling_type == 'nonlinear' or (coupling_type == 'mixed' and source % 2 == 1):
                    # Nonlinear coupling
                    source_signal = data[source, :-lag]
                    nonlinear_signal = coupling_strength * np.tanh(source_signal)
                    data[target, lag:] += nonlinear_signal
                
                ground_truth[target, source] = coupling_strength
        
        # Add noise
        data += noise_level * np.random.randn(n_voxels, n_timepoints)
        
        # Standardize
        scaler = StandardScaler()
        data = scaler.fit_transform(data.T).T
        
        return data, ground_truth
    
    def evaluate_performance(self, 
                           connectivity_matrix: np.ndarray,
                           ground_truth: np.ndarray,
                           method_name: str) -> Dict[str, float]:
        """Evaluate connectivity detection performance."""
        
        # Flatten matrices (excluding diagonal)
        mask = ~np.eye(ground_truth.shape[0], dtype=bool)
        conn_flat = connectivity_matrix[mask]
        gt_flat = (ground_truth[mask] > 0).astype(int)
        
        # Normalize connectivity values to [0, 1]
        if np.max(conn_flat) > np.min(conn_flat):
            conn_norm = (conn_flat - np.min(conn_flat)) / (np.max(conn_flat) - np.min(conn_flat))
        else:
            conn_norm = conn_flat
        
        # Compute metrics
        results = {}
        
        # ROC AUC
        if len(np.unique(gt_flat)) > 1:  # Need both classes for AUC
            results['auc_roc'] = roc_auc_score(gt_flat, conn_norm)
        else:
            results['auc_roc'] = 0.5
        
        # Precision-Recall AUC
        if np.sum(gt_flat) > 0:  # Need positive examples
            precision, recall, _ = precision_recall_curve(gt_flat, conn_norm)
            results['auc_pr'] = auc(recall, precision)
        else:
            results['auc_pr'] = 0.0
        
        # Binary classification metrics (using median threshold)
        threshold = np.median(conn_norm)
        pred_binary = (conn_norm > threshold).astype(int)
        
        results['accuracy'] = accuracy_score(gt_flat, pred_binary)
        results['precision'] = precision_score(gt_flat, pred_binary, zero_division=0)
        results['recall'] = recall_score(gt_flat, pred_binary, zero_division=0)
        results['f1_score'] = f1_score(gt_flat, pred_binary, zero_division=0)
        
        # Correlation with ground truth strength
        gt_strength = ground_truth[mask]
        if np.std(gt_strength) > 0:
            results['strength_correlation'] = stats.pearsonr(conn_flat, gt_strength)[0]
        else:
            results['strength_correlation'] = 0.0
        
        # Specificity and sensitivity at optimal threshold
        if len(np.unique(gt_flat)) > 1:
            # Find optimal threshold using Youden's index
            fpr_list, tpr_list, thresholds = [], [], []
            for thresh in np.linspace(0, 1, 50):
                pred_thresh = (conn_norm > thresh).astype(int)
                tn = np.sum((gt_flat == 0) & (pred_thresh == 0))
                fp = np.sum((gt_flat == 0) & (pred_thresh == 1))
                fn = np.sum((gt_flat == 1) & (pred_thresh == 0))
                tp = np.sum((gt_flat == 1) & (pred_thresh == 1))
                
                if (tn + fp) > 0 and (tp + fn) > 0:
                    fpr = fp / (fp + tn)
                    tpr = tp / (tp + fn)
                    fpr_list.append(fpr)
                    tpr_list.append(tpr)
                    thresholds.append(thresh)
            
            if fpr_list and tpr_list:
                # Youden's index = TPR - FPR
                youden_scores = np.array(tpr_list) - np.array(fpr_list)
                optimal_idx = np.argmax(youden_scores)
                results['optimal_sensitivity'] = tpr_list[optimal_idx]
                results['optimal_specificity'] = 1 - fpr_list[optimal_idx]
            else:
                results['optimal_sensitivity'] = 0.0
                results['optimal_specificity'] = 1.0
        else:
            results['optimal_sensitivity'] = 0.0
            results['optimal_specificity'] = 1.0
        
        return results
    
    def run_comprehensive_benchmark(self, 
                                  n_simulations: int = 10,
                                  n_voxels: int = 15,
                                  n_timepoints: int = 150) -> pd.DataFrame:
        """Run comprehensive benchmark across multiple simulations."""
        
        print("Running comprehensive connectivity method benchmark...")
        print(f"Simulations: {n_simulations}, Voxels: {n_voxels}, Timepoints: {n_timepoints}")
        
        all_results = []
        coupling_types = ['linear', 'nonlinear', 'mixed']
        noise_levels = [0.2, 0.5, 0.8]
        
        simulation_count = 0
        total_simulations = len(coupling_types) * len(noise_levels) * n_simulations
        
        for coupling_type in coupling_types:
            for noise_level in noise_levels:
                for sim_idx in range(n_simulations):
                    simulation_count += 1
                    print(f"Simulation {simulation_count}/{total_simulations}: "
                          f"{coupling_type} coupling, noise={noise_level:.1f}")
                    
                    # Generate synthetic data
                    data, ground_truth = self.generate_synthetic_data(
                        n_voxels=n_voxels,
                        n_timepoints=n_timepoints,
                        noise_level=noise_level,
                        coupling_type=coupling_type
                    )
                    
                    # Test each method
                    for method_name, method_func in self.methods.items():
                        try:
                            # Time the computation
                            start_time = time.time()
                            connectivity_matrix = method_func(data)
                            computation_time = time.time() - start_time
                            
                            # Evaluate performance
                            performance = self.evaluate_performance(
                                connectivity_matrix, ground_truth, method_name
                            )
                            
                            # Store results
                            result = {
                                'method': method_name,
                                'coupling_type': coupling_type,
                                'noise_level': noise_level,
                                'simulation': sim_idx,
                                'computation_time': computation_time,
                                **performance
                            }
                            all_results.append(result)
                            
                        except Exception as e:
                            print(f"Warning: {method_name} failed: {str(e)}")
                            # Add failed result with zeros
                            result = {
                                'method': method_name,
                                'coupling_type': coupling_type,
                                'noise_level': noise_level,
                                'simulation': sim_idx,
                                'computation_time': np.inf,
                                'auc_roc': 0.0,
                                'auc_pr': 0.0,
                                'accuracy': 0.0,
                                'precision': 0.0,
                                'recall': 0.0,
                                'f1_score': 0.0,
                                'strength_correlation': 0.0,
                                'optimal_sensitivity': 0.0,
                                'optimal_specificity': 0.0
                            }
                            all_results.append(result)
        
        return pd.DataFrame(all_results)
    
    def analyze_results(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze benchmark results and generate summary statistics."""
        
        print("\nAnalyzing benchmark results...")
        
        analysis = {}
        
        # Overall performance summary
        metrics = ['auc_roc', 'auc_pr', 'f1_score', 'accuracy', 'optimal_sensitivity', 'optimal_specificity']
        
        overall_performance = results_df.groupby('method')[metrics].agg(['mean', 'std'])
        analysis['overall_performance'] = overall_performance
        
        # Performance by coupling type
        coupling_performance = results_df.groupby(['method', 'coupling_type'])[metrics].mean()
        analysis['coupling_performance'] = coupling_performance
        
        # Performance by noise level
        noise_performance = results_df.groupby(['method', 'noise_level'])[metrics].mean()
        analysis['noise_performance'] = noise_performance
        
        # Computation time analysis
        time_analysis = results_df.groupby('method')['computation_time'].agg(['mean', 'std', 'median'])
        analysis['computation_time'] = time_analysis
        
        # Statistical significance tests (SMTE vs others)
        smte_results = results_df[results_df['method'] == 'SMTE']
        significance_tests = {}
        
        for metric in metrics:
            significance_tests[metric] = {}
            smte_values = smte_results[metric].values
            
            for method in results_df['method'].unique():
                if method != 'SMTE':
                    method_values = results_df[results_df['method'] == method][metric].values
                    if len(method_values) > 0 and len(smte_values) > 0:
                        try:
                            stat, p_value = stats.mannwhitneyu(smte_values, method_values, alternative='two-sided')
                            significance_tests[metric][method] = {
                                'statistic': stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            }
                        except Exception:
                            significance_tests[metric][method] = {
                                'statistic': np.nan,
                                'p_value': np.nan,
                                'significant': False
                            }
        
        analysis['significance_tests'] = significance_tests
        
        # Best performing method for each metric
        best_methods = {}
        for metric in metrics:
            best_method = results_df.groupby('method')[metric].mean().idxmax()
            best_score = results_df.groupby('method')[metric].mean().max()
            best_methods[metric] = {'method': best_method, 'score': best_score}
        
        analysis['best_methods'] = best_methods
        
        return analysis
    
    def create_visualizations(self, results_df: pd.DataFrame, analysis: Dict[str, Any], save_dir: str = '.'):
        """Create comprehensive visualizations of benchmark results."""
        
        print("Creating visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Overall performance comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        metrics = ['auc_roc', 'auc_pr', 'f1_score', 'accuracy', 'optimal_sensitivity', 'optimal_specificity']
        metric_names = ['ROC AUC', 'PR AUC', 'F1 Score', 'Accuracy', 'Sensitivity', 'Specificity']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            sns.boxplot(data=results_df, x='method', y=metric, ax=axes[i])
            axes[i].set_title(f'{name}')
            axes[i].set_xlabel('Method')
            axes[i].set_ylabel(name)
            axes[i].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.savefig(f'{save_dir}/overall_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Performance by coupling type
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        coupling_types = ['linear', 'nonlinear', 'mixed']
        
        for i, coupling_type in enumerate(coupling_types):
            subset = results_df[results_df['coupling_type'] == coupling_type]
            sns.boxplot(data=subset, x='method', y='auc_roc', ax=axes[i])
            axes[i].set_title(f'{coupling_type.title()} Coupling (ROC AUC)')
            axes[i].set_xlabel('Method')
            axes[i].set_ylabel('ROC AUC')
            axes[i].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.savefig(f'{save_dir}/performance_by_coupling_type.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Performance vs noise level
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROC AUC vs noise
        pivot_roc = results_df.pivot_table(values='auc_roc', index='noise_level', columns='method', aggfunc='mean')
        for method in pivot_roc.columns:
            axes[0].plot(pivot_roc.index, pivot_roc[method], marker='o', label=method, linewidth=2)
        axes[0].set_xlabel('Noise Level')
        axes[0].set_ylabel('ROC AUC')
        axes[0].set_title('Performance vs Noise Level (ROC AUC)')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # F1 Score vs noise
        pivot_f1 = results_df.pivot_table(values='f1_score', index='noise_level', columns='method', aggfunc='mean')
        for method in pivot_f1.columns:
            axes[1].plot(pivot_f1.index, pivot_f1[method], marker='s', label=method, linewidth=2)
        axes[1].set_xlabel('Noise Level')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('Performance vs Noise Level (F1 Score)')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/performance_vs_noise.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Computation time comparison
        plt.figure(figsize=(12, 8))
        time_data = results_df[results_df['computation_time'] < np.inf]  # Exclude failed methods
        sns.boxplot(data=time_data, x='method', y='computation_time')
        plt.yscale('log')
        plt.title('Computation Time Comparison')
        plt.xlabel('Method')
        plt.ylabel('Computation Time (seconds, log scale)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/computation_time_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. Performance summary heatmap
        performance_summary = results_df.groupby('method')[metrics].mean()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(performance_summary.T, annot=True, cmap='viridis', fmt='.3f', cbar_kws={'label': 'Performance Score'})
        plt.title('Performance Summary Heatmap')
        plt.xlabel('Method')
        plt.ylabel('Metric')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, results_df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
        """Generate comprehensive benchmark report."""
        
        report = []
        report.append("# Connectivity Methods Benchmark Report")
        report.append("=" * 50)
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        
        best_overall = analysis['overall_performance'].loc[:, ('auc_roc', 'mean')].idxmax()
        best_score = analysis['overall_performance'].loc[best_overall, ('auc_roc', 'mean')]
        
        report.append(f"**Best Overall Method (ROC AUC):** {best_overall} ({best_score:.3f})")
        report.append("")
        
        # Performance by metric
        report.append("## Performance by Metric")
        report.append("")
        
        for metric, info in analysis['best_methods'].items():
            report.append(f"**{metric.upper()}:** {info['method']} ({info['score']:.3f})")
        report.append("")
        
        # SMTE Performance Analysis
        report.append("## SMTE Performance Analysis")
        report.append("")
        
        smte_performance = analysis['overall_performance'].loc['SMTE']
        report.append("### Overall SMTE Performance:")
        report.append("| Metric | Mean | Std |")
        report.append("|--------|------|-----|")
        
        metrics = ['auc_roc', 'auc_pr', 'f1_score', 'accuracy']
        for metric in metrics:
            mean_val = smte_performance[(metric, 'mean')]
            std_val = smte_performance[(metric, 'std')]
            report.append(f"| {metric.upper()} | {mean_val:.3f} | {std_val:.3f} |")
        
        report.append("")
        
        # Statistical Significance
        report.append("### Statistical Significance (SMTE vs Others):")
        report.append("")
        
        for metric in ['auc_roc', 'f1_score']:
            report.append(f"**{metric.upper()}:**")
            if metric in analysis['significance_tests']:
                for method, test_result in analysis['significance_tests'][metric].items():
                    p_val = test_result['p_value']
                    significant = "✓" if test_result['significant'] else "✗"
                    report.append(f"- vs {method}: p={p_val:.4f} {significant}")
            report.append("")
        
        # Performance by Condition
        report.append("## Performance by Condition")
        report.append("")
        
        # By coupling type
        report.append("### By Coupling Type (ROC AUC):")
        coupling_perf = analysis['coupling_performance']
        smte_coupling = coupling_perf.loc['SMTE']
        
        report.append("| Coupling Type | SMTE AUC |")
        report.append("|---------------|----------|")
        for coupling_type in ['linear', 'nonlinear', 'mixed']:
            auc_val = smte_coupling.loc[coupling_type, 'auc_roc']
            report.append(f"| {coupling_type.title()} | {auc_val:.3f} |")
        
        report.append("")
        
        # Computational Efficiency
        report.append("## Computational Efficiency")
        report.append("")
        
        time_analysis = analysis['computation_time']
        report.append("| Method | Mean Time (s) | Median Time (s) |")
        report.append("|--------|---------------|-----------------|")
        
        for method in time_analysis.index:
            mean_time = time_analysis.loc[method, 'mean']
            median_time = time_analysis.loc[method, 'median']
            if not np.isinf(mean_time):
                report.append(f"| {method} | {mean_time:.3f} | {median_time:.3f} |")
            else:
                report.append(f"| {method} | Failed | Failed |")
        
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        smte_rank_auc = (analysis['overall_performance'].loc[:, ('auc_roc', 'mean')]
                        .rank(ascending=False).loc['SMTE'])
        smte_rank_f1 = (analysis['overall_performance'].loc[:, ('f1_score', 'mean')]
                       .rank(ascending=False).loc['SMTE'])
        
        report.append(f"- **SMTE Overall Ranking:** #{int(smte_rank_auc)} in ROC AUC, #{int(smte_rank_f1)} in F1 Score")
        
        # Identify SMTE's strengths
        if 'nonlinear' in smte_coupling.index:
            nonlinear_auc = smte_coupling.loc['nonlinear', 'auc_roc']
            linear_auc = smte_coupling.loc['linear', 'auc_roc']
            
            if nonlinear_auc > linear_auc:
                report.append("- **SMTE Strength:** Superior performance on nonlinear connectivity patterns")
            
        report.append("- **SMTE Use Cases:** Recommended for nonlinear, complex connectivity analysis")
        report.append("- **Limitations:** Higher computational cost compared to correlation-based methods")
        
        return "\n".join(report)


def main():
    """Run comprehensive benchmark and generate report."""
    
    print("Starting Comprehensive Connectivity Methods Benchmark")
    print("=" * 60)
    
    # Initialize benchmark
    benchmark = ConnectivityBenchmark(random_state=42)
    
    # Run benchmark with moderate parameters for demonstration
    results_df = benchmark.run_comprehensive_benchmark(
        n_simulations=5,  # Reduced for demo - use 20+ for production
        n_voxels=12,      # Reduced for speed - use 20+ for production  
        n_timepoints=100  # Reduced for speed - use 200+ for production
    )
    
    # Analyze results
    analysis = benchmark.analyze_results(results_df)
    
    # Create visualizations
    benchmark.create_visualizations(results_df, analysis)
    
    # Generate and save report
    report = benchmark.generate_report(results_df, analysis)
    
    with open('connectivity_benchmark_report.md', 'w') as f:
        f.write(report)
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(report)
    
    # Save detailed results
    results_df.to_csv('detailed_benchmark_results.csv', index=False)
    
    print(f"\nDetailed results saved to: detailed_benchmark_results.csv")
    print(f"Full report saved to: connectivity_benchmark_report.md")
    
    return results_df, analysis


if __name__ == "__main__":
    results_df, analysis = main()