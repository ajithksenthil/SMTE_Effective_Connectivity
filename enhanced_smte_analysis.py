#!/usr/bin/env python3
"""
Enhanced SMTE Analysis - Optimizing parameters and understanding performance characteristics.
This script investigates SMTE performance issues and provides optimized implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from typing import Dict, List, Tuple, Any
import time
import warnings
warnings.filterwarnings('ignore')

from voxel_smte_connectivity_corrected import VoxelSMTEConnectivity
from baseline_comparison import BaselineConnectivityMethods, ConnectivityBenchmark


class EnhancedSMTEAnalysis:
    """
    Enhanced analysis to understand and optimize SMTE performance.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def parameter_optimization_study(self, 
                                   data: np.ndarray, 
                                   ground_truth: np.ndarray) -> pd.DataFrame:
        """
        Comprehensive parameter optimization for SMTE.
        """
        print("Running SMTE parameter optimization study...")
        
        results = []
        
        # Parameter ranges to test
        symbol_counts = [3, 4, 5, 6, 8, 10]
        symbolizers = ['uniform', 'quantile', 'ordinal']
        ordinal_orders = [2, 3, 4]
        max_lags = [2, 3, 5, 7]
        
        total_combinations = len(symbol_counts) * len(symbolizers) * len(max_lags)
        current_combination = 0
        
        for n_symbols in symbol_counts:
            for symbolizer in symbolizers:
                for max_lag in max_lags:
                    current_combination += 1
                    
                    # Handle ordinal patterns specially
                    if symbolizer == 'ordinal':
                        for ordinal_order in ordinal_orders:
                            if ordinal_order <= 4:  # Avoid too many symbols
                                try:
                                    analyzer = VoxelSMTEConnectivity(
                                        n_symbols=np.math.factorial(ordinal_order),
                                        symbolizer=symbolizer,
                                        ordinal_order=ordinal_order,
                                        max_lag=max_lag,
                                        alpha=0.05,
                                        n_permutations=100,  # Reduced for speed
                                        random_state=self.random_state
                                    )
                                    
                                    # Compute SMTE
                                    start_time = time.time()
                                    symbolic_data = analyzer.symbolize_timeseries(data)
                                    analyzer.symbolic_data = symbolic_data
                                    smte_matrix, _ = analyzer.compute_voxel_connectivity_matrix()
                                    computation_time = time.time() - start_time
                                    
                                    # Evaluate performance
                                    performance = self._evaluate_performance(smte_matrix, ground_truth)
                                    
                                    result = {
                                        'n_symbols': np.math.factorial(ordinal_order),
                                        'symbolizer': symbolizer,
                                        'ordinal_order': ordinal_order,
                                        'max_lag': max_lag,
                                        'computation_time': computation_time,
                                        **performance
                                    }
                                    results.append(result)
                                    
                                except Exception as e:
                                    print(f"Failed: {symbolizer}, order={ordinal_order}, lag={max_lag}: {e}")
                    else:
                        try:
                            analyzer = VoxelSMTEConnectivity(
                                n_symbols=n_symbols,
                                symbolizer=symbolizer,
                                max_lag=max_lag,
                                alpha=0.05,
                                n_permutations=100,
                                random_state=self.random_state
                            )
                            
                            # Compute SMTE
                            start_time = time.time()
                            symbolic_data = analyzer.symbolize_timeseries(data)
                            analyzer.symbolic_data = symbolic_data
                            smte_matrix, _ = analyzer.compute_voxel_connectivity_matrix()
                            computation_time = time.time() - start_time
                            
                            # Evaluate performance
                            performance = self._evaluate_performance(smte_matrix, ground_truth)
                            
                            result = {
                                'n_symbols': n_symbols,
                                'symbolizer': symbolizer,
                                'ordinal_order': np.nan,
                                'max_lag': max_lag,
                                'computation_time': computation_time,
                                **performance
                            }
                            results.append(result)
                            
                        except Exception as e:
                            print(f"Failed: {symbolizer}, symbols={n_symbols}, lag={max_lag}: {e}")
                    
                    print(f"Progress: {current_combination}/{total_combinations} parameter combinations tested")
        
        return pd.DataFrame(results)
    
    def _evaluate_performance(self, connectivity_matrix: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
        """Evaluate connectivity detection performance."""
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
        
        # Flatten matrices (excluding diagonal)
        mask = ~np.eye(ground_truth.shape[0], dtype=bool)
        conn_flat = connectivity_matrix[mask]
        gt_flat = (ground_truth[mask] > 0).astype(int)
        
        # Normalize connectivity values
        if np.max(conn_flat) > np.min(conn_flat):
            conn_norm = (conn_flat - np.min(conn_flat)) / (np.max(conn_flat) - np.min(conn_flat))
        else:
            conn_norm = conn_flat
        
        results = {}
        
        # ROC AUC
        if len(np.unique(gt_flat)) > 1:
            results['auc_roc'] = roc_auc_score(gt_flat, conn_norm)
        else:
            results['auc_roc'] = 0.5
        
        # Precision-Recall AUC
        if np.sum(gt_flat) > 0:
            precision, recall, _ = precision_recall_curve(gt_flat, conn_norm)
            results['auc_pr'] = auc(recall, precision)
        else:
            results['auc_pr'] = 0.0
        
        # Mean connectivity strength for true connections
        true_connections = ground_truth[mask] > 0
        if np.sum(true_connections) > 0:
            results['true_connection_strength'] = np.mean(conn_flat[true_connections])
        else:
            results['true_connection_strength'] = 0.0
        
        # Mean connectivity strength for false connections
        false_connections = ground_truth[mask] == 0
        if np.sum(false_connections) > 0:
            results['false_connection_strength'] = np.mean(conn_flat[false_connections])
        else:
            results['false_connection_strength'] = 0.0
        
        # Signal-to-noise ratio
        if results['false_connection_strength'] > 0:
            results['snr'] = results['true_connection_strength'] / results['false_connection_strength']
        else:
            results['snr'] = np.inf if results['true_connection_strength'] > 0 else 1.0
        
        return results
    
    def coupling_strength_analysis(self) -> pd.DataFrame:
        """Analyze SMTE performance across different coupling strengths."""
        print("Analyzing SMTE performance across coupling strengths...")
        
        results = []
        coupling_strengths = [0.2, 0.4, 0.6, 0.8, 1.0]
        noise_levels = [0.2, 0.5, 0.8]
        
        for coupling_strength in coupling_strengths:
            for noise_level in noise_levels:
                for sim_idx in range(3):  # Reduced for speed
                    # Generate data
                    data, ground_truth = self._generate_optimized_synthetic_data(
                        coupling_strength=coupling_strength,
                        noise_level=noise_level,
                        n_voxels=10,
                        n_timepoints=200
                    )
                    
                    # Test optimized SMTE
                    analyzer = VoxelSMTEConnectivity(
                        n_symbols=6,
                        symbolizer='ordinal',
                        ordinal_order=3,
                        max_lag=5,
                        random_state=self.random_state + sim_idx
                    )
                    
                    symbolic_data = analyzer.symbolize_timeseries(data)
                    analyzer.symbolic_data = symbolic_data
                    smte_matrix, _ = analyzer.compute_voxel_connectivity_matrix()
                    
                    performance = self._evaluate_performance(smte_matrix, ground_truth)
                    
                    # Compare with lagged correlation
                    lagged_corr = BaselineConnectivityMethods.pearson_correlation(data, lag=1)
                    corr_performance = self._evaluate_performance(lagged_corr, ground_truth)
                    
                    result = {
                        'coupling_strength': coupling_strength,
                        'noise_level': noise_level,
                        'simulation': sim_idx,
                        'smte_auc': performance['auc_roc'],
                        'smte_snr': performance['snr'],
                        'corr_auc': corr_performance['auc_roc'],
                        'corr_snr': corr_performance['snr'],
                        'smte_true_strength': performance['true_connection_strength'],
                        'smte_false_strength': performance['false_connection_strength']
                    }
                    results.append(result)
        
        return pd.DataFrame(results)
    
    def _generate_optimized_synthetic_data(self, 
                                         coupling_strength: float = 0.6,
                                         noise_level: float = 0.3,
                                         n_voxels: int = 15,
                                         n_timepoints: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data optimized for SMTE detection."""
        np.random.seed(self.random_state)
        
        # Start with white noise
        data = np.random.randn(n_voxels, n_timepoints)
        ground_truth = np.zeros((n_voxels, n_voxels))
        
        # Add more diverse coupling patterns that SMTE should detect
        connections = [
            (0, 1, 1, 'threshold'),    # Threshold coupling
            (2, 3, 2, 'quadratic'),    # Quadratic coupling  
            (4, 5, 1, 'sigmoid'),      # Sigmoid coupling
            (6, 7, 3, 'linear'),       # Linear coupling
        ]
        
        for source, target, lag, coupling_type in connections:
            if source < n_voxels and target < n_voxels:
                source_signal = data[source, :-lag] if lag > 0 else data[source]
                
                if coupling_type == 'linear':
                    coupled_signal = coupling_strength * source_signal
                elif coupling_type == 'quadratic':
                    coupled_signal = coupling_strength * np.sign(source_signal) * source_signal**2
                elif coupling_type == 'threshold':
                    coupled_signal = coupling_strength * (source_signal > 0).astype(float)
                elif coupling_type == 'sigmoid':
                    coupled_signal = coupling_strength * (2 / (1 + np.exp(-2*source_signal)) - 1)
                else:
                    coupled_signal = coupling_strength * source_signal
                
                # Add coupling with appropriate delay
                if lag > 0:
                    data[target, lag:] += coupled_signal
                else:
                    data[target] += coupled_signal
                
                ground_truth[target, source] = coupling_strength
        
        # Add noise
        data += noise_level * np.random.randn(n_voxels, n_timepoints)
        
        # Standardize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data = scaler.fit_transform(data.T).T
        
        return data, ground_truth
    
    def nonlinear_coupling_comparison(self) -> pd.DataFrame:
        """Compare methods specifically on nonlinear coupling detection."""
        print("Comparing methods on nonlinear coupling detection...")
        
        results = []
        
        # Generate data with purely nonlinear couplings
        for sim_idx in range(5):
            data, ground_truth = self._generate_nonlinear_data(sim_idx)
            
            methods = {
                'SMTE_Ordinal': lambda d: self._compute_optimized_smte(d, 'ordinal'),
                'SMTE_Uniform': lambda d: self._compute_optimized_smte(d, 'uniform'),
                'Mutual_Information': BaselineConnectivityMethods.mutual_information,
                'Lagged_Correlation': lambda d: BaselineConnectivityMethods.pearson_correlation(d, lag=1),
                'Phase_Lag_Index': BaselineConnectivityMethods.phase_lag_index
            }
            
            for method_name, method_func in methods.items():
                try:
                    connectivity_matrix = method_func(data)
                    performance = self._evaluate_performance(connectivity_matrix, ground_truth)
                    
                    result = {
                        'method': method_name,
                        'simulation': sim_idx,
                        **performance
                    }
                    results.append(result)
                    
                except Exception as e:
                    print(f"Method {method_name} failed: {e}")
        
        return pd.DataFrame(results)
    
    def _generate_nonlinear_data(self, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate data with purely nonlinear couplings."""
        np.random.seed(seed)
        
        n_voxels = 12
        n_timepoints = 200
        
        # Base signals with different characteristics
        data = np.random.randn(n_voxels, n_timepoints)
        ground_truth = np.zeros((n_voxels, n_voxels))
        
        # Nonlinear couplings only
        coupling_strength = 0.8
        
        # 0 -> 1: Threshold coupling
        threshold_signal = (data[0, :-1] > 0.5).astype(float) * coupling_strength
        data[1, 1:] += threshold_signal
        ground_truth[1, 0] = coupling_strength
        
        # 2 -> 3: Quadratic coupling
        quad_signal = coupling_strength * np.sign(data[2, :-2]) * data[2, :-2]**2
        data[3, 2:] += quad_signal
        ground_truth[3, 2] = coupling_strength
        
        # 4 -> 5: Sigmoid coupling
        sigmoid_signal = coupling_strength * np.tanh(data[4, :-1])
        data[5, 1:] += sigmoid_signal
        ground_truth[5, 4] = coupling_strength
        
        # 6 -> 7: XOR-like coupling
        xor_signal = coupling_strength * np.sign(data[6, :-1] * data[6, :-2])
        data[7, 1:] += xor_signal[:-1]
        ground_truth[7, 6] = coupling_strength
        
        # Add noise
        data += 0.3 * np.random.randn(n_voxels, n_timepoints)
        
        # Standardize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data = scaler.fit_transform(data.T).T
        
        return data, ground_truth
    
    def _compute_optimized_smte(self, data: np.ndarray, symbolizer: str) -> np.ndarray:
        """Compute SMTE with optimized parameters."""
        analyzer = VoxelSMTEConnectivity(
            n_symbols=6 if symbolizer == 'ordinal' else 8,
            symbolizer=symbolizer,
            ordinal_order=3 if symbolizer == 'ordinal' else None,
            max_lag=5,
            random_state=self.random_state
        )
        
        symbolic_data = analyzer.symbolize_timeseries(data)
        analyzer.symbolic_data = symbolic_data
        smte_matrix, _ = analyzer.compute_voxel_connectivity_matrix()
        
        return smte_matrix
    
    def create_optimization_visualizations(self, 
                                         param_results: pd.DataFrame,
                                         coupling_results: pd.DataFrame,
                                         nonlinear_results: pd.DataFrame):
        """Create visualizations for optimization analysis."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Parameter optimization heatmap
        if not param_results.empty:
            # Best parameters by symbolizer
            best_by_symbolizer = param_results.groupby('symbolizer')['auc_roc'].max()
            ax = axes[0, 0]
            best_by_symbolizer.plot(kind='bar', ax=ax)
            ax.set_title('Best ROC AUC by Symbolizer')
            ax.set_ylabel('ROC AUC')
            ax.tick_params(axis='x', rotation=45)
        
        # 2. Coupling strength analysis
        if not coupling_results.empty:
            ax = axes[0, 1]
            pivot_data = coupling_results.pivot_table(
                values='smte_auc', 
                index='coupling_strength', 
                columns='noise_level', 
                aggfunc='mean'
            )
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=ax)
            ax.set_title('SMTE Performance vs Coupling Strength & Noise')
            ax.set_xlabel('Noise Level')
            ax.set_ylabel('Coupling Strength')
        
        # 3. SMTE vs Correlation comparison
        if not coupling_results.empty:
            ax = axes[0, 2]
            ax.scatter(coupling_results['corr_auc'], coupling_results['smte_auc'], alpha=0.6)
            ax.plot([0, 1], [0, 1], 'r--', alpha=0.5)
            ax.set_xlabel('Lagged Correlation AUC')
            ax.set_ylabel('SMTE AUC')
            ax.set_title('SMTE vs Lagged Correlation')
        
        # 4. Nonlinear coupling performance
        if not nonlinear_results.empty:
            ax = axes[1, 0]
            sns.boxplot(data=nonlinear_results, x='method', y='auc_roc', ax=ax)
            ax.set_title('Nonlinear Coupling Detection')
            ax.set_ylabel('ROC AUC')
            ax.tick_params(axis='x', rotation=45)
        
        # 5. Signal-to-noise ratio analysis
        if not coupling_results.empty:
            ax = axes[1, 1]
            coupling_results.groupby('coupling_strength')['smte_snr'].mean().plot(ax=ax, marker='o')
            ax.set_title('SMTE Signal-to-Noise Ratio')
            ax.set_xlabel('Coupling Strength')
            ax.set_ylabel('SNR')
            ax.grid(True, alpha=0.3)
        
        # 6. Computation time vs performance
        if not param_results.empty:
            ax = axes[1, 2]
            valid_data = param_results[param_results['computation_time'] < 10]  # Remove outliers
            ax.scatter(valid_data['computation_time'], valid_data['auc_roc'], 
                      c=valid_data['n_symbols'], cmap='plasma', alpha=0.6)
            ax.set_xlabel('Computation Time (s)')
            ax.set_ylabel('ROC AUC')
            ax.set_title('Performance vs Computation Time')
            plt.colorbar(ax.collections[0], ax=ax, label='Number of Symbols')
        
        plt.tight_layout()
        plt.savefig('smte_optimization_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_optimization_report(self, 
                                   param_results: pd.DataFrame,
                                   coupling_results: pd.DataFrame,
                                   nonlinear_results: pd.DataFrame) -> str:
        """Generate comprehensive optimization report."""
        
        report = []
        report.append("# SMTE Optimization Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Best parameters
        if not param_results.empty:
            best_config = param_results.loc[param_results['auc_roc'].idxmax()]
            report.append("## Optimal SMTE Configuration")
            report.append("")
            report.append(f"**Best ROC AUC:** {best_config['auc_roc']:.3f}")
            report.append(f"**Symbolizer:** {best_config['symbolizer']}")
            report.append(f"**Number of Symbols:** {best_config['n_symbols']}")
            if not pd.isna(best_config['ordinal_order']):
                report.append(f"**Ordinal Order:** {int(best_config['ordinal_order'])}")
            report.append(f"**Max Lag:** {int(best_config['max_lag'])}")
            report.append(f"**Computation Time:** {best_config['computation_time']:.3f}s")
            report.append("")
        
        # Performance by symbolizer
        if not param_results.empty:
            report.append("## Performance by Symbolization Method")
            report.append("")
            symbolizer_performance = param_results.groupby('symbolizer')['auc_roc'].agg(['mean', 'max', 'std'])
            report.append("| Symbolizer | Mean AUC | Max AUC | Std AUC |")
            report.append("|------------|----------|---------|---------|")
            for symbolizer in symbolizer_performance.index:
                mean_auc = symbolizer_performance.loc[symbolizer, 'mean']
                max_auc = symbolizer_performance.loc[symbolizer, 'max']
                std_auc = symbolizer_performance.loc[symbolizer, 'std']
                report.append(f"| {symbolizer} | {mean_auc:.3f} | {max_auc:.3f} | {std_auc:.3f} |")
            report.append("")
        
        # Coupling strength analysis
        if not coupling_results.empty:
            report.append("## Coupling Strength Analysis")
            report.append("")
            
            # Performance by coupling strength
            coupling_perf = coupling_results.groupby('coupling_strength')['smte_auc'].mean()
            report.append("### SMTE Performance vs Coupling Strength:")
            report.append("| Coupling Strength | Mean AUC |")
            report.append("|-------------------|----------|")
            for strength in coupling_perf.index:
                auc = coupling_perf[strength]
                report.append(f"| {strength} | {auc:.3f} |")
            report.append("")
            
            # SMTE vs Correlation comparison
            smte_better = np.sum(coupling_results['smte_auc'] > coupling_results['corr_auc'])
            total_comparisons = len(coupling_results)
            report.append(f"**SMTE superior to lagged correlation:** {smte_better}/{total_comparisons} cases ({100*smte_better/total_comparisons:.1f}%)")
            report.append("")
        
        # Nonlinear coupling analysis
        if not nonlinear_results.empty:
            report.append("## Nonlinear Coupling Detection")
            report.append("")
            
            nonlinear_performance = nonlinear_results.groupby('method')['auc_roc'].mean().sort_values(ascending=False)
            report.append("### Performance Ranking (Nonlinear Couplings Only):")
            report.append("| Rank | Method | Mean AUC |")
            report.append("|------|--------|----------|")
            for rank, (method, auc) in enumerate(nonlinear_performance.items(), 1):
                report.append(f"| {rank} | {method} | {auc:.3f} |")
            report.append("")
            
            # SMTE ranking
            smte_methods = [method for method in nonlinear_performance.index if 'SMTE' in method]
            if smte_methods:
                best_smte = max(smte_methods, key=lambda x: nonlinear_performance[x])
                smte_rank = list(nonlinear_performance.index).index(best_smte) + 1
                report.append(f"**Best SMTE variant:** {best_smte} (Rank #{smte_rank})")
                report.append("")
        
        # Key insights
        report.append("## Key Insights")
        report.append("")
        
        if not param_results.empty:
            # Best symbolizer
            best_symbolizer = param_results.groupby('symbolizer')['auc_roc'].mean().idxmax()
            report.append(f"- **Best symbolization method:** {best_symbolizer}")
            
            # Computation efficiency
            time_performance = param_results.groupby('symbolizer')['computation_time'].mean()
            fastest_symbolizer = time_performance.idxmin()
            report.append(f"- **Most efficient symbolizer:** {fastest_symbolizer} ({time_performance[fastest_symbolizer]:.3f}s)")
        
        if not coupling_results.empty:
            # Threshold analysis
            strong_coupling = coupling_results[coupling_results['coupling_strength'] >= 0.6]
            weak_coupling = coupling_results[coupling_results['coupling_strength'] <= 0.4]
            
            if not strong_coupling.empty and not weak_coupling.empty:
                strong_performance = strong_coupling['smte_auc'].mean()
                weak_performance = weak_coupling['smte_auc'].mean()
                report.append(f"- **Strong coupling performance:** {strong_performance:.3f} AUC")
                report.append(f"- **Weak coupling performance:** {weak_performance:.3f} AUC")
        
        # Recommendations
        report.append("")
        report.append("## Recommendations for SMTE Usage")
        report.append("")
        
        if not param_results.empty:
            best_config = param_results.loc[param_results['auc_roc'].idxmax()]
            report.append(f"1. **Use {best_config['symbolizer']} symbolization** for optimal performance")
            report.append(f"2. **Set max_lag to {int(best_config['max_lag'])}** for this data type")
            
        if not nonlinear_results.empty:
            smte_vs_others = nonlinear_results[nonlinear_results['method'].str.contains('SMTE')]['auc_roc'].mean()
            other_methods = nonlinear_results[~nonlinear_results['method'].str.contains('SMTE')]['auc_roc'].mean()
            
            if smte_vs_others > other_methods:
                report.append("3. **SMTE shows advantage** for nonlinear coupling detection")
            else:
                report.append("3. **Consider alternative methods** for linear coupling detection")
        
        report.append("4. **Increase permutations to 1000+** for final analyses")
        report.append("5. **Use longer time series (200+ points)** when possible")
        
        return "\n".join(report)


def main():
    """Run enhanced SMTE analysis."""
    print("Starting Enhanced SMTE Analysis")
    print("=" * 50)
    
    # Initialize analysis
    analysis = EnhancedSMTEAnalysis(random_state=42)
    
    # Generate test data
    benchmark = ConnectivityBenchmark(random_state=42)
    data, ground_truth = benchmark.generate_synthetic_data(
        n_voxels=15,
        n_timepoints=150,
        coupling_type='mixed',
        noise_level=0.4
    )
    
    # Run parameter optimization (reduced scope for demo)
    print("Running parameter optimization...")
    param_results = analysis.parameter_optimization_study(data, ground_truth)
    
    # Run coupling strength analysis
    print("Running coupling strength analysis...")
    coupling_results = analysis.coupling_strength_analysis()
    
    # Run nonlinear coupling comparison
    print("Running nonlinear coupling comparison...")
    nonlinear_results = analysis.nonlinear_coupling_comparison()
    
    # Create visualizations
    print("Creating visualizations...")
    analysis.create_optimization_visualizations(param_results, coupling_results, nonlinear_results)
    
    # Generate report
    report = analysis.generate_optimization_report(param_results, coupling_results, nonlinear_results)
    
    # Save results
    param_results.to_csv('smte_parameter_optimization.csv', index=False)
    coupling_results.to_csv('smte_coupling_analysis.csv', index=False)
    nonlinear_results.to_csv('smte_nonlinear_analysis.csv', index=False)
    
    with open('smte_optimization_report.md', 'w') as f:
        f.write(report)
    
    print("\n" + "=" * 50)
    print("ENHANCED SMTE ANALYSIS COMPLETE")
    print("=" * 50)
    print(report)
    
    return param_results, coupling_results, nonlinear_results


if __name__ == "__main__":
    param_results, coupling_results, nonlinear_results = main()