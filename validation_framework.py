#!/usr/bin/env python3
"""
Validation Framework for SMTE Improvements
This module provides comprehensive testing to ensure improvements don't break existing functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import roc_auc_score
import time
import warnings
from pathlib import Path

from voxel_smte_connectivity_corrected import VoxelSMTEConnectivity

warnings.filterwarnings('ignore')


class SMTEValidationFramework:
    """
    Comprehensive validation framework for SMTE improvements.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Reference implementation for comparison
        self.reference_smte = VoxelSMTEConnectivity(
            n_symbols=6,
            symbolizer='ordinal',
            ordinal_order=3,
            max_lag=5,
            alpha=0.05,
            n_permutations=100,  # Reduced for testing
            random_state=random_state
        )
        
    def generate_test_datasets(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Generate multiple test datasets with known properties."""
        
        datasets = {}
        
        # 1. Simple linear coupling
        datasets['linear'] = self._generate_linear_coupling()
        
        # 2. Nonlinear coupling
        datasets['nonlinear'] = self._generate_nonlinear_coupling()
        
        # 3. Multi-lag coupling
        datasets['multilag'] = self._generate_multilag_coupling()
        
        # 4. Realistic fMRI-like data
        datasets['fmri_like'] = self._generate_fmri_like_data()
        
        # 5. No coupling (null data)
        datasets['null'] = self._generate_null_data()
        
        return datasets
    
    def _generate_linear_coupling(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate data with simple linear coupling."""
        n_regions = 10
        n_timepoints = 150
        
        data = np.random.randn(n_regions, n_timepoints)
        ground_truth = np.zeros((n_regions, n_regions))
        
        # Add linear couplings
        data[1, 2:] += 0.6 * data[0, :-2]  # 0 -> 1 with lag 2
        data[3, 1:] += 0.5 * data[2, :-1]  # 2 -> 3 with lag 1
        data[5, 3:] += 0.4 * data[4, :-3]  # 4 -> 5 with lag 3
        
        ground_truth[1, 0] = 0.6
        ground_truth[3, 2] = 0.5
        ground_truth[5, 4] = 0.4
        
        return data, ground_truth
    
    def _generate_nonlinear_coupling(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate data with nonlinear coupling."""
        n_regions = 8
        n_timepoints = 150
        
        data = np.random.randn(n_regions, n_timepoints)
        ground_truth = np.zeros((n_regions, n_regions))
        
        # Add nonlinear couplings
        data[1, 2:] += 0.5 * np.tanh(data[0, :-2])  # 0 -> 1 nonlinear
        data[3, 1:] += 0.4 * np.sin(data[2, :-1])   # 2 -> 3 nonlinear
        
        ground_truth[1, 0] = 0.5
        ground_truth[3, 2] = 0.4
        
        return data, ground_truth
    
    def _generate_multilag_coupling(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate data with multiple lag relationships."""
        n_regions = 12
        n_timepoints = 200
        
        data = np.random.randn(n_regions, n_timepoints)
        ground_truth = np.zeros((n_regions, n_regions))
        
        # Multiple lag couplings
        for lag in range(1, 6):
            source = lag - 1
            target = lag + 5
            if target < n_regions:
                strength = 0.6 - 0.1 * lag
                data[target, lag:] += strength * data[source, :-lag]
                ground_truth[target, source] = strength
        
        return data, ground_truth
    
    def _generate_fmri_like_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic fMRI-like data with network structure."""
        n_regions = 15
        n_timepoints = 120
        TR = 2.0
        
        t = np.arange(n_timepoints) * TR
        data = []
        
        for i in range(n_regions):
            # Base neural signal
            base_freq = 0.01 + 0.02 * np.random.rand()
            neural = np.sin(2 * np.pi * base_freq * t)
            
            # Physiological noise
            respiratory = 0.1 * np.sin(2 * np.pi * 0.3 * t)
            cardiac = 0.05 * np.sin(2 * np.pi * 1.0 * t)
            thermal = 0.3 * np.random.randn(n_timepoints)
            
            signal = neural + respiratory + cardiac + thermal
            data.append(signal)
        
        data = np.array(data)
        ground_truth = np.zeros((n_regions, n_regions))
        
        # Add known network connections
        connections = [(0, 1, 2), (2, 3, 1), (4, 5, 3), (1, 6, 1)]
        
        for source, target, lag in connections:
            strength = 0.4 + 0.2 * np.random.rand()
            data[target, lag:] += strength * data[source, :-lag]
            ground_truth[target, source] = strength
        
        return data, ground_truth
    
    def _generate_null_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate null data with no coupling."""
        n_regions = 8
        n_timepoints = 100
        
        data = np.random.randn(n_regions, n_timepoints)
        ground_truth = np.zeros((n_regions, n_regions))
        
        return data, ground_truth
    
    def validate_implementation(self, 
                              improved_smte: VoxelSMTEConnectivity,
                              implementation_name: str) -> Dict[str, Any]:
        """
        Validate an improved SMTE implementation against reference.
        """
        
        print(f"\nValidating {implementation_name}...")
        print("-" * 50)
        
        datasets = self.generate_test_datasets()
        validation_results = {
            'implementation_name': implementation_name,
            'datasets': {},
            'summary': {}
        }
        
        for dataset_name, (data, ground_truth) in datasets.items():
            print(f"Testing on {dataset_name} dataset...")
            
            dataset_result = self._validate_on_dataset(
                data, ground_truth, improved_smte, dataset_name
            )
            validation_results['datasets'][dataset_name] = dataset_result
        
        # Compute summary statistics
        validation_results['summary'] = self._compute_validation_summary(
            validation_results['datasets']
        )
        
        # Check for regressions
        regression_check = self._check_for_regressions(validation_results)
        validation_results['regression_check'] = regression_check
        
        return validation_results
    
    def _validate_on_dataset(self, 
                           data: np.ndarray,
                           ground_truth: np.ndarray,
                           improved_smte: VoxelSMTEConnectivity,
                           dataset_name: str) -> Dict[str, Any]:
        """Validate on a single dataset."""
        
        result = {
            'dataset_name': dataset_name,
            'data_shape': data.shape,
            'n_true_connections': np.sum(ground_truth > 0)
        }
        
        try:
            # Test reference implementation
            print(f"  Running reference SMTE...")
            start_time = time.time()
            
            ref_symbolic = self.reference_smte.symbolize_timeseries(data)
            self.reference_smte.symbolic_data = ref_symbolic
            ref_connectivity, ref_lags = self.reference_smte.compute_voxel_connectivity_matrix()
            
            ref_time = time.time() - start_time
            
            # Test improved implementation
            print(f"  Running improved SMTE...")
            start_time = time.time()
            
            imp_symbolic = improved_smte.symbolize_timeseries(data)
            improved_smte.symbolic_data = imp_symbolic
            imp_connectivity, imp_lags = improved_smte.compute_voxel_connectivity_matrix()
            
            imp_time = time.time() - start_time
            
            # Compute performance metrics
            result.update({
                'reference_time': ref_time,
                'improved_time': imp_time,
                'speedup': ref_time / imp_time if imp_time > 0 else np.inf,
                'reference_performance': self._compute_performance_metrics(ref_connectivity, ground_truth),
                'improved_performance': self._compute_performance_metrics(imp_connectivity, ground_truth),
                'connectivity_correlation': np.corrcoef(ref_connectivity.flatten(), imp_connectivity.flatten())[0, 1],
                'max_connectivity_diff': np.max(np.abs(ref_connectivity - imp_connectivity)),
                'mean_connectivity_diff': np.mean(np.abs(ref_connectivity - imp_connectivity))
            })
            
            result['success'] = True
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            result.update({
                'success': False,
                'error': str(e),
                'reference_performance': {'auc': 0.0},
                'improved_performance': {'auc': 0.0}
            })
        
        return result
    
    def _compute_performance_metrics(self, 
                                   connectivity_matrix: np.ndarray,
                                   ground_truth: np.ndarray) -> Dict[str, float]:
        """Compute performance metrics against ground truth."""
        
        # Flatten matrices (exclude diagonal)
        mask = ~np.eye(ground_truth.shape[0], dtype=bool)
        conn_flat = connectivity_matrix[mask]
        gt_flat = (ground_truth[mask] > 0).astype(int)
        
        # Normalize connectivity values
        if np.max(conn_flat) > np.min(conn_flat):
            conn_norm = (conn_flat - np.min(conn_flat)) / (np.max(conn_flat) - np.min(conn_flat))
        else:
            conn_norm = conn_flat
        
        metrics = {}
        
        # ROC AUC
        if len(np.unique(gt_flat)) > 1:
            metrics['auc'] = roc_auc_score(gt_flat, conn_norm)
        else:
            metrics['auc'] = 0.5
        
        # Top-k accuracy
        if np.sum(gt_flat) > 0:
            k = min(10, np.sum(gt_flat))
            top_k_indices = np.argsort(conn_norm)[-k:]
            metrics['top_k_accuracy'] = np.sum(gt_flat[top_k_indices]) / k
        else:
            metrics['top_k_accuracy'] = 0.0
        
        # Mean connectivity strength for true connections
        if np.sum(gt_flat) > 0:
            metrics['true_connection_strength'] = np.mean(conn_norm[gt_flat == 1])
        else:
            metrics['true_connection_strength'] = 0.0
        
        return metrics
    
    def _compute_validation_summary(self, dataset_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute summary statistics across all datasets."""
        
        summary = {
            'total_datasets': len(dataset_results),
            'successful_datasets': 0,
            'failed_datasets': 0,
            'mean_performance_improvement': 0.0,
            'mean_speedup': 0.0,
            'performance_improvements': [],
            'speedups': []
        }
        
        for dataset_name, result in dataset_results.items():
            if result['success']:
                summary['successful_datasets'] += 1
                
                # Performance improvement
                ref_auc = result['reference_performance']['auc']
                imp_auc = result['improved_performance']['auc']
                
                if ref_auc > 0:
                    improvement = (imp_auc - ref_auc) / ref_auc
                    summary['performance_improvements'].append(improvement)
                
                # Speedup
                if 'speedup' in result and not np.isinf(result['speedup']):
                    summary['speedups'].append(result['speedup'])
                    
            else:
                summary['failed_datasets'] += 1
        
        # Compute means
        if summary['performance_improvements']:
            summary['mean_performance_improvement'] = np.mean(summary['performance_improvements'])
        
        if summary['speedups']:
            summary['mean_speedup'] = np.mean(summary['speedups'])
        
        return summary
    
    def _check_for_regressions(self, validation_results: Dict[str, Any]) -> Dict[str, bool]:
        """Check for performance regressions."""
        
        checks = {
            'no_failures': validation_results['summary']['failed_datasets'] == 0,
            'performance_maintained': validation_results['summary']['mean_performance_improvement'] >= -0.05,  # Allow 5% degradation
            'reasonable_speed': validation_results['summary']['mean_speedup'] >= 0.1,  # At least 10% of original speed
            'numerical_stability': True  # Will be set based on correlation checks
        }
        
        # Check numerical stability
        correlations = []
        for dataset_result in validation_results['datasets'].values():
            if dataset_result['success'] and 'connectivity_correlation' in dataset_result:
                correlations.append(dataset_result['connectivity_correlation'])
        
        if correlations:
            min_correlation = np.min(correlations)
            checks['numerical_stability'] = min_correlation > 0.9  # High correlation required
        
        return checks
    
    def create_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Create a comprehensive validation report."""
        
        report = []
        report.append(f"# Validation Report: {validation_results['implementation_name']}")
        report.append("=" * 60)
        report.append("")
        
        # Summary
        summary = validation_results['summary']
        report.append("## Summary")
        report.append("")
        report.append(f"**Total datasets tested:** {summary['total_datasets']}")
        report.append(f"**Successful:** {summary['successful_datasets']}")
        report.append(f"**Failed:** {summary['failed_datasets']}")
        report.append(f"**Mean performance improvement:** {summary['mean_performance_improvement']:.2%}")
        report.append(f"**Mean speedup:** {summary['mean_speedup']:.2f}x")
        report.append("")
        
        # Regression checks
        regression = validation_results['regression_check']
        report.append("## Regression Checks")
        report.append("")
        
        for check_name, passed in regression.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            report.append(f"- **{check_name.replace('_', ' ').title()}:** {status}")
        
        report.append("")
        
        # Overall status
        all_passed = all(regression.values())
        overall_status = "✅ VALIDATION PASSED" if all_passed else "❌ VALIDATION FAILED"
        report.append(f"## Overall Status: {overall_status}")
        report.append("")
        
        # Detailed results
        report.append("## Detailed Results")
        report.append("")
        report.append("| Dataset | Success | AUC Ref | AUC Imp | Improvement | Speedup |")
        report.append("|---------|---------|---------|---------|-------------|---------|")
        
        for dataset_name, result in validation_results['datasets'].items():
            if result['success']:
                ref_auc = result['reference_performance']['auc']
                imp_auc = result['improved_performance']['auc']
                improvement = ((imp_auc - ref_auc) / ref_auc * 100) if ref_auc > 0 else 0
                speedup = result.get('speedup', 0)
                
                report.append(f"| {dataset_name} | ✅ | {ref_auc:.3f} | {imp_auc:.3f} | {improvement:+.1f}% | {speedup:.2f}x |")
            else:
                report.append(f"| {dataset_name} | ❌ | - | - | - | - |")
        
        report.append("")
        
        return "\n".join(report)


def run_validation_demo():
    """Demonstrate the validation framework."""
    
    print("SMTE Validation Framework Demo")
    print("=" * 50)
    
    # Create validation framework
    validator = SMTEValidationFramework()
    
    # Test against reference implementation (should be identical)
    reference_copy = VoxelSMTEConnectivity(
        n_symbols=6,
        symbolizer='ordinal',
        ordinal_order=3,
        max_lag=5,
        alpha=0.05,
        n_permutations=100,
        random_state=42
    )
    
    # Run validation
    results = validator.validate_implementation(reference_copy, "Reference Copy")
    
    # Generate report
    report = validator.create_validation_report(results)
    print(report)
    
    return results


if __name__ == "__main__":
    results = run_validation_demo()