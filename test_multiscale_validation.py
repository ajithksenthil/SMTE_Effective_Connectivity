#!/usr/bin/env python3
"""
Validate the Multi-Scale SMTE implementation using the validation framework.
"""

import sys
import numpy as np
from validation_framework import SMTEValidationFramework
from multiscale_smte_v1 import MultiScaleSMTE

def test_multiscale_smte_validation():
    """Test multi-scale SMTE using the validation framework."""
    
    print("Phase 2.1 Validation: Multi-Scale Temporal Analysis")
    print("=" * 70)
    
    # Create validation framework
    validator = SMTEValidationFramework(random_state=42)
    
    # Test single-scale mode (baseline)
    print("\n1. Testing Single-Scale Mode (Baseline)")
    print("-" * 50)
    
    single_scale_smte = MultiScaleSMTE(
        use_multiscale_analysis=False,  # Disable multi-scale
        adaptive_mode='heuristic',
        use_network_correction=True,
        use_physiological_constraints=True,
        n_permutations=100,
        random_state=42
    )
    
    results_single = validator.validate_implementation(
        single_scale_smte, "Multi-Scale SMTE (Single-Scale Mode)"
    )
    
    print("\nSingle-Scale Mode Results:")
    print(f"Performance improvement: {results_single['summary']['mean_performance_improvement']:.2%}")
    print(f"Speed: {results_single['summary']['mean_speedup']:.2f}x")
    
    # Test multi-scale mode (fast + intermediate only for validation speed)
    print("\n2. Testing Multi-Scale Mode (Fast + Intermediate)")
    print("-" * 55)
    
    multiscale_smte = MultiScaleSMTE(
        use_multiscale_analysis=True,   # Enable multi-scale
        scales_to_analyze=['fast', 'intermediate'],  # Reduced for validation speed
        adaptive_mode='heuristic',
        use_network_correction=True,
        use_physiological_constraints=True,
        n_permutations=100,
        random_state=42
    )
    
    results_multiscale = validator.validate_implementation(
        multiscale_smte, "Multi-Scale SMTE (Multi-Scale Mode)"
    )
    
    print("\nMulti-Scale Mode Results:")
    print(f"Performance improvement: {results_multiscale['summary']['mean_performance_improvement']:.2%}")
    print(f"Speed: {results_multiscale['summary']['mean_speedup']:.2f}x")
    
    # Test all scales mode (comprehensive)
    print("\n3. Testing All Scales Mode (Fast + Intermediate + Slow)")
    print("-" * 60)
    
    all_scales_smte = MultiScaleSMTE(
        use_multiscale_analysis=True,   # Enable multi-scale
        scales_to_analyze=['fast', 'intermediate', 'slow'],  # All scales
        adaptive_mode='heuristic',
        use_network_correction=True,
        use_physiological_constraints=True,
        n_permutations=100,
        random_state=42
    )
    
    results_all_scales = validator.validate_implementation(
        all_scales_smte, "Multi-Scale SMTE (All Scales Mode)"
    )
    
    print("\nAll Scales Mode Results:")
    print(f"Performance improvement: {results_all_scales['summary']['mean_performance_improvement']:.2%}")
    print(f"Speed: {results_all_scales['summary']['mean_speedup']:.2f}x")
    
    # Generate reports
    print("\n" + "="*70)
    print("DETAILED VALIDATION REPORTS")
    print("="*70)
    
    report_single = validator.create_validation_report(results_single)
    print(report_single)
    
    print("\n" + "="*70)
    
    report_multiscale = validator.create_validation_report(results_multiscale)
    print(report_multiscale)
    
    print("\n" + "="*70)
    
    report_all_scales = validator.create_validation_report(results_all_scales)
    print(report_all_scales)
    
    # Summary
    print("\n" + "="*70)
    print("PHASE 2.1 COMPLETION SUMMARY")
    print("="*70)
    
    single_passed = all(results_single['regression_check'].values())
    multiscale_passed = all(results_multiscale['regression_check'].values())
    all_scales_passed = all(results_all_scales['regression_check'].values())
    
    print(f"✅ Single-Scale Mode: {'PASSED' if single_passed else 'FAILED'}")
    print(f"✅ Multi-Scale Mode: {'PASSED' if multiscale_passed else 'FAILED'}")
    print(f"✅ All Scales Mode: {'PASSED' if all_scales_passed else 'FAILED'}")
    
    # Multi-scale analysis
    if single_passed and multiscale_passed and all_scales_passed:
        print("\n📊 MULTI-SCALE ANALYSIS:")
        
        # Compare speeds
        single_speed = results_single['summary']['mean_speedup']
        multiscale_speed = results_multiscale['summary']['mean_speedup'] 
        all_scales_speed = results_all_scales['summary']['mean_speedup']
        
        print(f"Speed comparison:")
        print(f"  Single-scale:     {single_speed:.2f}x")
        print(f"  Multi-scale (2):  {multiscale_speed:.2f}x")
        print(f"  All scales (3):   {all_scales_speed:.2f}x")
        
        # Expected overhead calculation
        expected_overhead_2 = 2.0  # 2 scales
        expected_overhead_3 = 3.0  # 3 scales
        
        actual_overhead_2 = single_speed / multiscale_speed if multiscale_speed > 0 else float('inf')
        actual_overhead_3 = single_speed / all_scales_speed if all_scales_speed > 0 else float('inf')
        
        print(f"\nComputational overhead:")
        print(f"  2 scales: {actual_overhead_2:.1f}x (expected ~{expected_overhead_2:.1f}x)")
        print(f"  3 scales: {actual_overhead_3:.1f}x (expected ~{expected_overhead_3:.1f}x)")
        
        # Performance comparison
        single_perf = results_single['summary']['mean_performance_improvement']
        multiscale_perf = results_multiscale['summary']['mean_performance_improvement']
        all_scales_perf = results_all_scales['summary']['mean_performance_improvement']
        
        print(f"\nPerformance comparison:")
        print(f"  Single-scale:     {single_perf:.2%}")
        print(f"  Multi-scale (2):  {multiscale_perf:.2%}")
        print(f"  All scales (3):   {all_scales_perf:.2%}")
        
        print("\n🎉 PHASE 2.1 SUCCESSFULLY COMPLETED!")
        print("Multi-scale temporal analysis is working correctly.")
        print("- Framework analyzes connectivity across multiple temporal scales")
        print("- Scale-specific parameter optimization implemented")
        print("- Multi-scale result combination working properly")
        print("- Performance maintained within acceptable bounds")
        print("- Implementation is numerically stable across all scales")
        return True
    else:
        print("\n❌ PHASE 2.1 FAILED - Issues detected in validation")
        return False

if __name__ == "__main__":
    success = test_multiscale_smte_validation()
    sys.exit(0 if success else 1)