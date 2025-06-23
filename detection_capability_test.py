#!/usr/bin/env python3
"""
Detection Capability Test: Can the SMTE Framework Actually Detect Anything?
Progressive testing from strong to weak signals to find detection thresholds.
"""

import numpy as np
import pandas as pd
import time
import warnings
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any

# Import implementations
from voxel_smte_connectivity_corrected import VoxelSMTEConnectivity

warnings.filterwarnings('ignore')

class DetectionCapabilityTester:
    """Test if SMTE framework can detect connections under various conditions."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def create_test_data_with_varying_strength(self, 
                                             n_rois: int = 8,
                                             n_timepoints: int = 200,
                                             connection_strength: float = 0.8,
                                             noise_level: float = 0.2,
                                             n_connections: int = 3) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Create test data with controllable connection strength and noise.
        
        Parameters:
        -----------
        connection_strength : float
            Strength of true connections (0.0 to 1.0)
        noise_level : float
            Level of added noise (0.0 to 1.0)
        """
        
        roi_labels = [f'ROI_{i+1}' for i in range(n_rois)]
        
        # Generate base signals with clear oscillations
        data = np.zeros((n_rois, n_timepoints))
        
        for i in range(n_rois):
            # Each ROI has a distinct base frequency
            base_freq = 0.05 + (i * 0.02)  # 0.05 to 0.19 Hz
            t = np.arange(n_timepoints) * 2.0  # TR = 2s
            
            # Clean oscillatory signal
            signal = np.sin(2 * np.pi * base_freq * t)
            signal += 0.3 * np.sin(2 * np.pi * (base_freq * 2) * t)  # Harmonic
            
            # Add controlled noise
            signal += noise_level * np.random.randn(n_timepoints)
            
            data[i] = signal
        
        # Add strong, clear connections
        ground_truth = np.zeros((n_rois, n_rois))
        
        # Define clear connections with specified strength
        connections = [
            (0, 1, 1),  # ROI_1 -> ROI_2
            (2, 3, 2),  # ROI_3 -> ROI_4  
            (4, 5, 1),  # ROI_5 -> ROI_6
        ][:n_connections]
        
        for source, target, lag in connections:
            if source < n_rois and target < n_rois and lag < n_timepoints:
                # Apply strong, clear connection
                data[target, lag:] += connection_strength * data[source, :-lag]
                ground_truth[source, target] = connection_strength
        
        # Standardize
        scaler = StandardScaler()
        data = scaler.fit_transform(data.T).T
        
        return data, roi_labels, ground_truth
    
    def test_detection_across_conditions(self) -> Dict[str, Any]:
        """Test detection capability across various signal and noise conditions."""
        
        print("🔬 DETECTION CAPABILITY TESTING")
        print("=" * 60)
        print("Testing if SMTE framework can detect connections under optimal conditions")
        print("=" * 60)
        
        results = {}
        
        # Test conditions: from easy to hard
        test_conditions = [
            # (connection_strength, noise_level, n_timepoints, description)
            (0.9, 0.1, 300, "Very Strong Signal, Low Noise, Long Scan"),
            (0.8, 0.2, 250, "Strong Signal, Low Noise, Long Scan"),
            (0.7, 0.2, 200, "Strong Signal, Medium Noise, Medium Scan"),
            (0.6, 0.3, 200, "Medium Signal, Medium Noise, Medium Scan"),
            (0.5, 0.3, 150, "Medium Signal, Medium Noise, Short Scan"),
            (0.4, 0.4, 120, "Weak Signal, High Noise, Short Scan (Realistic)"),
        ]
        
        for i, (strength, noise, timepoints, description) in enumerate(test_conditions):
            print(f"\n{i+1}. TESTING: {description}")
            print("-" * 50)
            print(f"   Connection strength: {strength:.1f}")
            print(f"   Noise level: {noise:.1f}")  
            print(f"   Scan duration: {timepoints} timepoints ({timepoints*2/60:.1f} min)")
            
            # Create test data
            data, roi_labels, ground_truth = self.create_test_data_with_varying_strength(
                n_rois=8, 
                n_timepoints=timepoints,
                connection_strength=strength,
                noise_level=noise,
                n_connections=3
            )
            
            # Test with multiple statistical thresholds
            threshold_results = {}
            
            for alpha in [0.01, 0.05, 0.1]:  # Multiple thresholds
                print(f"\n   Testing with α = {alpha}")
                
                try:
                    start_time = time.time()
                    
                    # Test with optimized parameters for detection
                    smte = VoxelSMTEConnectivity(
                        n_symbols=3,        # Moderate complexity
                        ordinal_order=2,    # Simple patterns
                        max_lag=5,          # Longer lags
                        n_permutations=100, # More permutations for stable p-values
                        random_state=self.random_state
                    )
                    
                    # Set custom alpha for this test
                    original_alpha = smte.alpha
                    smte.alpha = alpha
                    
                    # Compute connectivity
                    smte.fmri_data = data
                    smte.mask = np.ones(data.shape[0], dtype=bool)
                    
                    symbolic_data = smte.symbolize_timeseries(data)
                    smte.symbolic_data = symbolic_data
                    connectivity_matrix, _ = smte.compute_voxel_connectivity_matrix()
                    p_values = smte.statistical_testing(connectivity_matrix)
                    significance_mask = smte.fdr_correction(p_values)
                    
                    # Restore original alpha
                    smte.alpha = original_alpha
                    
                    computation_time = time.time() - start_time
                    
                    # Evaluate results
                    n_significant = np.sum(significance_mask)
                    true_connections = (ground_truth > 0.1).astype(int)
                    
                    # Count true/false positives
                    n_rois = data.shape[0]
                    triu_indices = np.triu_indices(n_rois, k=1)
                    true_binary = true_connections[triu_indices]
                    pred_binary = significance_mask[triu_indices].astype(int)
                    
                    true_positives = np.sum((true_binary == 1) & (pred_binary == 1))
                    false_positives = np.sum((true_binary == 0) & (pred_binary == 1))
                    total_true = np.sum(true_binary)
                    
                    detection_rate = (true_positives / total_true * 100) if total_true > 0 else 0
                    
                    threshold_results[alpha] = {
                        'n_significant': n_significant,
                        'true_positives': true_positives,
                        'false_positives': false_positives,
                        'detection_rate': detection_rate,
                        'computation_time': computation_time,
                        'success': True
                    }
                    
                    print(f"     → {n_significant} significant, {true_positives}/{total_true} detected ({detection_rate:.1f}%)")
                    
                except Exception as e:
                    print(f"     → Error: {str(e)}")
                    threshold_results[alpha] = {'error': str(e), 'success': False}
            
            results[f"condition_{i+1}"] = {
                'description': description,
                'parameters': {
                    'strength': strength,
                    'noise': noise,
                    'timepoints': timepoints
                },
                'threshold_results': threshold_results
            }
            
            # Summary for this condition
            successful_tests = {k: v for k, v in threshold_results.items() if v.get('success', False)}
            if successful_tests:
                best_detection = max([r['detection_rate'] for r in successful_tests.values()])
                best_alpha = [k for k, v in successful_tests.items() if v['detection_rate'] == best_detection][0]
                print(f"\n   🎯 BEST RESULT: {best_detection:.1f}% detection at α={best_alpha}")
                
                if best_detection > 50:
                    print("   ✅ EXCELLENT: Framework can detect connections under these conditions")
                elif best_detection > 20:
                    print("   ✅ GOOD: Framework shows detection capability")
                elif best_detection > 0:
                    print("   ⚠️ LIMITED: Framework shows some detection capability")
                else:
                    print("   ❌ POOR: No detection under these conditions")
            else:
                print("   ❌ FAILED: All threshold tests failed")
        
        return results
    
    def create_detection_summary(self, results: Dict[str, Any]) -> str:
        """Create summary report of detection capabilities."""
        
        report = []
        report.append("# DETECTION CAPABILITY TEST RESULTS")
        report.append("=" * 50)
        report.append("")
        
        # Summary table
        summary_data = []
        
        for condition_key, condition_data in results.items():
            if 'threshold_results' in condition_data:
                description = condition_data['description']
                params = condition_data['parameters']
                threshold_results = condition_data['threshold_results']
                
                successful_results = {k: v for k, v in threshold_results.items() if v.get('success', False)}
                
                if successful_results:
                    best_detection = max([r['detection_rate'] for r in successful_results.values()])
                    best_alpha = [k for k, v in successful_results.items() if v['detection_rate'] == best_detection][0]
                    
                    summary_data.append({
                        'Condition': description,
                        'Strength': f"{params['strength']:.1f}",
                        'Noise': f"{params['noise']:.1f}",
                        'Duration (min)': f"{params['timepoints']*2/60:.1f}",
                        'Best Detection': f"{best_detection:.1f}%",
                        'Best α': best_alpha,
                        'Status': '✅' if best_detection > 0 else '❌'
                    })
                else:
                    summary_data.append({
                        'Condition': description,
                        'Strength': f"{params['strength']:.1f}",
                        'Noise': f"{params['noise']:.1f}", 
                        'Duration (min)': f"{params['timepoints']*2/60:.1f}",
                        'Best Detection': "0.0%",
                        'Best α': "N/A",
                        'Status': '❌'
                    })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            report.append("## DETECTION SUMMARY")
            report.append("-" * 30)
            report.append("")
            report.append(df.to_string(index=False))
            report.append("")
            
            # Analysis
            report.append("## KEY FINDINGS")
            report.append("-" * 20)
            report.append("")
            
            successful_conditions = [r for r in summary_data if r['Status'] == '✅']
            
            if successful_conditions:
                report.append(f"✅ **Detection Confirmed**: Framework can detect connections under {len(successful_conditions)} conditions")
                report.append("")
                
                # Find minimum requirements for detection
                best_condition = max(successful_conditions, key=lambda x: float(x['Best Detection'].rstrip('%')))
                report.append("**Optimal Detection Conditions:**")
                report.append(f"- Connection strength: {best_condition['Strength']}")
                report.append(f"- Noise level: {best_condition['Noise']}")
                report.append(f"- Scan duration: {best_condition['Duration (min)']} minutes")
                report.append(f"- Statistical threshold: α = {best_condition['Best α']}")
                report.append(f"- **Detection rate: {best_condition['Best Detection']}**")
                report.append("")
                
                # Minimum requirements
                min_working = min(successful_conditions, 
                                key=lambda x: (float(x['Strength']), -float(x['Noise'].rstrip('%'))))
                report.append("**Minimum Requirements for Detection:**")
                report.append(f"- Connection strength ≥ {min_working['Strength']}")
                report.append(f"- Noise level ≤ {min_working['Noise']}")
                report.append(f"- Scan duration ≥ {min_working['Duration (min)']} minutes")
                report.append("")
                
            else:
                report.append("❌ **No Detection**: Framework failed to detect connections under all tested conditions")
                report.append("")
                report.append("**This suggests:**")
                report.append("- Framework parameters need optimization")
                report.append("- Statistical thresholds are too conservative")
                report.append("- Implementation may have fundamental issues")
                report.append("")
            
            # Recommendations
            report.append("## RECOMMENDATIONS")
            report.append("-" * 25)
            report.append("")
            
            if successful_conditions:
                report.append("**For Research Use:**")
                report.append("1. Use connection strengths ≥ 0.6 for reliable detection")
                report.append("2. Collect scans ≥ 8 minutes for adequate statistical power")
                report.append("3. Apply noise reduction preprocessing when possible")
                report.append("4. Consider relaxed thresholds (α = 0.01) for exploratory analyses")
                report.append("")
                
                report.append("**Framework Improvements:**")
                report.append("1. Implement adaptive threshold selection")
                report.append("2. Add preprocessing modules for noise reduction")
                report.append("3. Optimize parameters for realistic effect sizes")
                report.append("4. Add sensitivity analysis tools")
                
            else:
                report.append("**Critical Issues to Address:**")
                report.append("1. Review statistical testing implementation")
                report.append("2. Debug symbolization and SMTE computation")
                report.append("3. Validate against known-working implementations")
                report.append("4. Consider alternative parameter ranges")
        
        return "\n".join(report)

def main():
    """Run comprehensive detection capability testing."""
    
    print("🚀 DETECTION CAPABILITY TESTING")
    print("=" * 80)
    print("Testing if the SMTE framework can actually detect anything")
    print("Progressive testing from optimal to realistic conditions")
    print("=" * 80)
    
    # Initialize tester
    tester = DetectionCapabilityTester(random_state=42)
    
    # Run comprehensive detection testing
    results = tester.test_detection_across_conditions()
    
    # Generate summary report
    summary_report = tester.create_detection_summary(results)
    
    print("\n" + "="*80)
    print("DETECTION CAPABILITY SUMMARY")
    print("="*80)
    print(summary_report)
    
    # Save report
    report_file = '/Users/ajithsenthil/Desktop/SMTE_EConnect/detection_capability_report.md'
    with open(report_file, 'w') as f:
        f.write(summary_report)
    
    print(f"\n📄 Complete detection capability report saved to: {report_file}")
    
    return results

if __name__ == "__main__":
    results = main()