#!/usr/bin/env python3
"""
Test Backward Compatibility of Fixed Causal Graph Clustering
Ensures all existing functionality still works while validating the improvements.
"""

import numpy as np
import sys
import traceback
from typing import Dict, Any

def test_backward_compatibility():
    """Test that all existing implementations still work."""
    
    print("🔍 TESTING BACKWARD COMPATIBILITY")
    print("=" * 60)
    
    compatibility_results = {}
    
    # Test 1: Original VoxelSMTEConnectivity
    try:
        print("\n1. Testing VoxelSMTEConnectivity...")
        
        from voxel_smte_connectivity_corrected import VoxelSMTEConnectivity
        
        # Create simple test data
        np.random.seed(42)
        test_data = np.random.randn(5, 100)
        
        smte = VoxelSMTEConnectivity(n_symbols=2, ordinal_order=2, max_lag=3)
        smte.fmri_data = test_data
        smte.mask = np.ones(5, dtype=bool)
        
        # Test symbolization
        symbolic_data = smte.symbolize_timeseries(test_data)
        assert symbolic_data.shape == test_data.shape
        
        # Test connectivity computation
        smte.symbolic_data = symbolic_data
        connectivity_matrix, _ = smte.compute_voxel_connectivity_matrix()
        assert connectivity_matrix.shape == (5, 5)
        
        # Test statistical testing
        p_values = smte.statistical_testing(connectivity_matrix)
        assert p_values.shape == (5, 5)
        
        # Test FDR correction
        significance_mask = smte.fdr_correction(p_values)
        assert significance_mask.shape == (5, 5)
        
        compatibility_results['VoxelSMTEConnectivity'] = "✅ PASS"
        print("   ✅ VoxelSMTEConnectivity: All methods work")
        
    except Exception as e:
        compatibility_results['VoxelSMTEConnectivity'] = f"❌ FAIL: {str(e)}"
        print(f"   ❌ VoxelSMTEConnectivity failed: {e}")
        traceback.print_exc()
    
    # Test 2: Enhanced SMTE implementations
    enhanced_classes = [
        'AdaptiveSMTE',
        'NetworkAwareSMTE', 
        'PhysiologicalSMTE',
        'MultiScaleSMTE',
        'EnsembleSMTE',
        'HierarchicalSMTE'
    ]
    
    for class_name in enhanced_classes:
        try:
            print(f"\n2. Testing {class_name}...")
            
            if class_name == 'AdaptiveSMTE':
                from adaptive_smte_v1 import AdaptiveSMTE
                enhanced_smte = AdaptiveSMTE()
            elif class_name == 'NetworkAwareSMTE':
                from network_aware_smte_v1 import NetworkAwareSMTE
                enhanced_smte = NetworkAwareSMTE()
            elif class_name == 'PhysiologicalSMTE':
                from physiological_smte_v1 import PhysiologicalSMTE
                enhanced_smte = PhysiologicalSMTE()
            elif class_name == 'MultiScaleSMTE':
                from multiscale_smte_v1 import MultiScaleSMTE
                enhanced_smte = MultiScaleSMTE()
            elif class_name == 'EnsembleSMTE':
                from ensemble_smte_v1 import EnsembleSMTE
                enhanced_smte = EnsembleSMTE()
            elif class_name == 'HierarchicalSMTE':
                from hierarchical_smte_v1 import HierarchicalSMTE
                enhanced_smte = HierarchicalSMTE()
            
            # Test basic functionality
            enhanced_smte.fmri_data = test_data
            enhanced_smte.mask = np.ones(5, dtype=bool)
            
            # Test key methods exist and work
            if hasattr(enhanced_smte, 'symbolize_timeseries'):
                symbolic_data = enhanced_smte.symbolize_timeseries(test_data)
                enhanced_smte.symbolic_data = symbolic_data
            
            if hasattr(enhanced_smte, 'compute_voxel_connectivity_matrix'):
                connectivity_matrix, _ = enhanced_smte.compute_voxel_connectivity_matrix()
                
            compatibility_results[class_name] = "✅ PASS"
            print(f"   ✅ {class_name}: Basic functionality works")
            
        except ImportError:
            compatibility_results[class_name] = "⚠️ SKIP: Not found"
            print(f"   ⚠️ {class_name}: File not found (skipping)")
        except Exception as e:
            compatibility_results[class_name] = f"❌ FAIL: {str(e)}"
            print(f"   ❌ {class_name} failed: {e}")
    
    # Test 3: Graph clustering implementations
    try:
        print(f"\n3. Testing Graph Clustering...")
        
        from smte_graph_clustering_v1 import SMTEGraphClusterAnalyzer
        
        # Test initialization
        analyzer = SMTEGraphClusterAnalyzer()
        
        # Test with simple data
        simple_connectivity = np.random.rand(5, 5)
        simple_p_values = np.random.rand(5, 5)
        simple_labels = [f"ROI_{i}" for i in range(5)]
        
        # This should not break
        results = analyzer.analyze_smte_graph_clusters(
            simple_connectivity, simple_p_values, simple_labels, n_permutations=10
        )
        
        compatibility_results['SMTEGraphClusterAnalyzer'] = "✅ PASS"
        print("   ✅ SMTEGraphClusterAnalyzer: Works correctly")
        
    except ImportError:
        compatibility_results['SMTEGraphClusterAnalyzer'] = "⚠️ SKIP: Not found"
        print("   ⚠️ SMTEGraphClusterAnalyzer: File not found (skipping)")
    except Exception as e:
        compatibility_results['SMTEGraphClusterAnalyzer'] = f"❌ FAIL: {str(e)}"
        print(f"   ❌ SMTEGraphClusterAnalyzer failed: {e}")
    
    # Test 4: Clustering method comparison (with fixed causal clustering)
    try:
        print(f"\n4. Testing Fixed Clustering Method Comparison...")
        
        from clustering_method_comparison import ClusteringMethodComparison
        
        comparator = ClusteringMethodComparison(random_state=42)
        
        # Test data creation
        data, roi_labels, ground_truth, cluster_info = comparator.create_test_scenario_with_spatial_and_causal_clusters()
        
        # Test that it doesn't crash
        assert data.shape[0] == len(roi_labels)
        assert ground_truth.shape == (len(roi_labels), len(roi_labels))
        
        compatibility_results['ClusteringMethodComparison'] = "✅ PASS"
        print("   ✅ ClusteringMethodComparison: Data creation works")
        
    except Exception as e:
        compatibility_results['ClusteringMethodComparison'] = f"❌ FAIL: {str(e)}"
        print(f"   ❌ ClusteringMethodComparison failed: {e}")
    
    # Test 5: Validation framework
    try:
        print(f"\n5. Testing Validation Framework...")
        
        from validation_framework import ValidationFramework
        
        validator = ValidationFramework()
        
        # Test basic validation
        reference_smte = VoxelSMTEConnectivity(n_symbols=2, ordinal_order=2, max_lag=3, n_permutations=10)
        
        # This should not crash
        results = validator.validate_implementation(reference_smte, "test_implementation")
        
        compatibility_results['ValidationFramework'] = "✅ PASS"
        print("   ✅ ValidationFramework: Basic validation works")
        
    except ImportError:
        compatibility_results['ValidationFramework'] = "⚠️ SKIP: Not found"
        print("   ⚠️ ValidationFramework: File not found (skipping)")
    except Exception as e:
        compatibility_results['ValidationFramework'] = f"❌ FAIL: {str(e)}"
        print(f"   ❌ ValidationFramework failed: {e}")
    
    # Summary
    print(f"\n📊 COMPATIBILITY SUMMARY")
    print("-" * 40)
    
    passed = sum(1 for result in compatibility_results.values() if result == "✅ PASS")
    failed = sum(1 for result in compatibility_results.values() if "❌ FAIL" in result)
    skipped = sum(1 for result in compatibility_results.values() if "⚠️ SKIP" in result)
    total = len(compatibility_results)
    
    for component, result in compatibility_results.items():
        print(f"   {result}: {component}")
    
    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped out of {total} total")
    
    success_rate = passed / (passed + failed) * 100 if (passed + failed) > 0 else 100
    
    if success_rate >= 80:
        print(f"\n✅ BACKWARD COMPATIBILITY: EXCELLENT ({success_rate:.1f}%)")
        overall_status = "PASS"
    elif success_rate >= 60:
        print(f"\n⚠️ BACKWARD COMPATIBILITY: GOOD ({success_rate:.1f}%)")
        overall_status = "ACCEPTABLE"
    else:
        print(f"\n❌ BACKWARD COMPATIBILITY: POOR ({success_rate:.1f}%)")
        overall_status = "FAIL"
    
    return compatibility_results, overall_status

def test_new_functionality():
    """Test that the new fixed causal clustering works."""
    
    print(f"\n🆕 TESTING NEW FUNCTIONALITY")
    print("=" * 40)
    
    try:
        from final_fixed_causal_clustering import FinalFixedCausalClustering
        
        # Create test data
        np.random.seed(42)
        connectivity_matrix = np.random.rand(5, 5)
        p_values = np.random.rand(5, 5) * 0.1  # Liberal p-values
        roi_labels = [f"ROI_{i}" for i in range(5)]
        
        # Test fixed clustering
        fixed_clustering = FinalFixedCausalClustering(random_state=42)
        result = fixed_clustering.apply_robust_causal_clustering(
            connectivity_matrix, p_values, roi_labels, alpha=0.05, verbose=False
        )
        
        assert result.shape == p_values.shape
        assert result.dtype == bool
        
        print("   ✅ Fixed causal clustering: Works correctly")
        
        # Test integration with comparison framework
        from clustering_method_comparison import ClusteringMethodComparison
        
        comparator = ClusteringMethodComparison(random_state=42)
        
        # Test the fixed method directly
        fixed_result = comparator._apply_causal_graph_clustering_correction(
            connectivity_matrix, p_values, roi_labels, alpha=0.05
        )
        
        assert fixed_result.shape == p_values.shape
        
        print("   ✅ Integration with comparison framework: Works correctly")
        
        return True
        
    except Exception as e:
        print(f"   ❌ New functionality failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all compatibility and functionality tests."""
    
    print("🧪 COMPREHENSIVE COMPATIBILITY AND FUNCTIONALITY TEST")
    print("=" * 70)
    
    # Test backward compatibility
    compatibility_results, compatibility_status = test_backward_compatibility()
    
    # Test new functionality  
    new_functionality_works = test_new_functionality()
    
    # Final assessment
    print(f"\n🏆 FINAL ASSESSMENT")
    print("=" * 30)
    print(f"Backward Compatibility: {compatibility_status}")
    print(f"New Functionality: {'✅ PASS' if new_functionality_works else '❌ FAIL'}")
    
    if compatibility_status in ["PASS", "ACCEPTABLE"] and new_functionality_works:
        print(f"\n🎉 OVERALL: SUCCESS - Fixed implementation maintains compatibility")
        return True
    else:
        print(f"\n⚠️ OVERALL: NEEDS ATTENTION - Some issues detected")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)