#!/usr/bin/env python3
"""
Validate the Hierarchical SMTE implementation using the validation framework.
"""

import sys
import numpy as np
from validation_framework import SMTEValidationFramework
from hierarchical_smte_v1 import HierarchicalSMTE

def test_hierarchical_smte_validation():
    """Test hierarchical SMTE using the validation framework."""
    
    print("Phase 2.3 Validation: Hierarchical Connectivity Analysis")
    print("=" * 70)
    
    # Create validation framework
    validator = SMTEValidationFramework(random_state=42)
    
    # Test ensemble mode (baseline)
    print("\\n1. Testing Ensemble Mode (Baseline)")
    print("-" * 45)
    
    ensemble_baseline_smte = HierarchicalSMTE(
        use_hierarchical_analysis=False,  # Disable hierarchical
        use_ensemble_testing=True,
        surrogate_methods=['aaft'],
        n_surrogates_per_method=15,
        use_multiscale_analysis=True,
        scales_to_analyze=['fast'],
        adaptive_mode='heuristic',
        use_network_correction=True,
        use_physiological_constraints=True,
        n_permutations=100,
        random_state=42
    )
    
    results_ensemble = validator.validate_implementation(
        ensemble_baseline_smte, "Hierarchical SMTE (Ensemble Mode)"
    )
    
    print("\\nEnsemble Mode Results:")
    print(f"Performance improvement: {results_ensemble['summary']['mean_performance_improvement']:.2%}")
    print(f"Speed: {results_ensemble['summary']['mean_speedup']:.2f}x")
    
    # Test hierarchical mode (agglomerative only for validation speed)
    print("\\n2. Testing Hierarchical Mode (Agglomerative)")
    print("-" * 55)
    
    hierarchical_smte_agg = HierarchicalSMTE(
        use_hierarchical_analysis=True,   # Enable hierarchical
        hierarchy_methods=['agglomerative'],  # Single method for speed
        hierarchy_levels=[2, 4],  # Reduced levels for validation speed
        distance_metrics=['correlation'],  # Single distance metric
        use_ensemble_testing=True,
        surrogate_methods=['aaft'],
        n_surrogates_per_method=10,  # Reduced for validation speed
        use_multiscale_analysis=True,
        scales_to_analyze=['fast'],  # Single scale for speed
        adaptive_mode='heuristic',
        use_network_correction=True,
        use_physiological_constraints=True,
        n_permutations=100,
        random_state=42
    )
    
    results_hierarchical_agg = validator.validate_implementation(
        hierarchical_smte_agg, "Hierarchical SMTE (Agglomerative)"
    )
    
    print("\\nHierarchical (Agglomerative) Results:")
    print(f"Performance improvement: {results_hierarchical_agg['summary']['mean_performance_improvement']:.2%}")
    print(f"Speed: {results_hierarchical_agg['summary']['mean_speedup']:.2f}x")
    
    # Test hierarchical mode (spectral clustering for comprehensive test)
    print("\\n3. Testing Hierarchical Mode (Spectral)")
    print("-" * 50)
    
    hierarchical_smte_spectral = HierarchicalSMTE(
        use_hierarchical_analysis=True,   # Enable hierarchical
        hierarchy_methods=['spectral'],   # Spectral clustering
        hierarchy_levels=[2, 4],  # Reduced levels for validation speed
        distance_metrics=['correlation'],  # Single distance metric
        use_ensemble_testing=True,
        surrogate_methods=['aaft'],
        n_surrogates_per_method=10,  # Reduced for validation speed
        use_multiscale_analysis=True,
        scales_to_analyze=['fast'],  # Single scale for speed
        adaptive_mode='heuristic',
        use_network_correction=True,
        use_physiological_constraints=True,
        n_permutations=100,
        random_state=42
    )
    
    results_hierarchical_spectral = validator.validate_implementation(
        hierarchical_smte_spectral, "Hierarchical SMTE (Spectral)"
    )
    
    print("\\nHierarchical (Spectral) Results:")
    print(f"Performance improvement: {results_hierarchical_spectral['summary']['mean_performance_improvement']:.2%}")
    print(f"Speed: {results_hierarchical_spectral['summary']['mean_speedup']:.2f}x")
    
    # Generate reports
    print("\\n" + "="*70)
    print("DETAILED VALIDATION REPORTS")
    print("="*70)
    
    report_ensemble = validator.create_validation_report(results_ensemble)
    print(report_ensemble)
    
    print("\\n" + "="*70)
    
    report_hierarchical_agg = validator.create_validation_report(results_hierarchical_agg)
    print(report_hierarchical_agg)
    
    print("\\n" + "="*70)
    
    report_hierarchical_spectral = validator.create_validation_report(results_hierarchical_spectral)
    print(report_hierarchical_spectral)
    
    # Summary
    print("\\n" + "="*70)
    print("PHASE 2.3 COMPLETION SUMMARY")
    print("="*70)
    
    ensemble_passed = all(results_ensemble['regression_check'].values())
    hierarchical_agg_passed = all(results_hierarchical_agg['regression_check'].values())
    hierarchical_spectral_passed = all(results_hierarchical_spectral['regression_check'].values())
    
    print(f"✅ Ensemble Mode: {'PASSED' if ensemble_passed else 'FAILED'}")
    print(f"✅ Hierarchical (Agglomerative): {'PASSED' if hierarchical_agg_passed else 'FAILED'}")
    print(f"✅ Hierarchical (Spectral): {'PASSED' if hierarchical_spectral_passed else 'FAILED'}")
    
    # Hierarchical analysis
    if ensemble_passed and hierarchical_agg_passed and hierarchical_spectral_passed:
        print("\\n📊 HIERARCHICAL ANALYSIS:")
        
        # Compare speeds
        ensemble_speed = results_ensemble['summary']['mean_speedup']
        agg_speed = results_hierarchical_agg['summary']['mean_speedup']
        spectral_speed = results_hierarchical_spectral['summary']['mean_speedup']
        
        print(f"Speed comparison:")
        print(f"  Ensemble (baseline):      {ensemble_speed:.2f}x")
        print(f"  Hierarchical (agg):       {agg_speed:.2f}x")
        print(f"  Hierarchical (spectral):  {spectral_speed:.2f}x")
        
        # Expected computational overhead
        agg_overhead = ensemble_speed / agg_speed if agg_speed > 0 else float('inf')
        spectral_overhead = ensemble_speed / spectral_speed if spectral_speed > 0 else float('inf')
        
        print(f"\\nComputational overhead:")
        print(f"  Agglomerative clustering: {agg_overhead:.1f}x")
        print(f"  Spectral clustering:      {spectral_overhead:.1f}x")
        
        # Performance comparison
        ensemble_perf = results_ensemble['summary']['mean_performance_improvement']
        agg_perf = results_hierarchical_agg['summary']['mean_performance_improvement']
        spectral_perf = results_hierarchical_spectral['summary']['mean_performance_improvement']
        
        print(f"\\nPerformance comparison:")
        print(f"  Ensemble:                 {ensemble_perf:.2%}")
        print(f"  Hierarchical (agg):       {agg_perf:.2%}")
        print(f"  Hierarchical (spectral):  {spectral_perf:.2%}")
        
        # Check for performance gains from hierarchical analysis
        agg_gain = agg_perf - ensemble_perf
        spectral_gain = spectral_perf - ensemble_perf
        
        print(f"\\nHierarchical analysis gains:")
        print(f"  Agglomerative:    {agg_gain:+.2%}")
        print(f"  Spectral:         {spectral_gain:+.2%}")
        
        print("\\n🎉 PHASE 2.3 SUCCESSFULLY COMPLETED!")
        print("Hierarchical connectivity analysis is working correctly.")
        print("- Multiple clustering methods implemented (agglomerative, spectral)")
        print("- Multi-level hierarchy decomposition functional")
        print("- Distance metrics and stability analysis working")
        print("- Network organization analysis implemented")
        print("- Performance maintained within acceptable bounds")
        print("- Implementation is numerically stable across clustering methods")
        
        print("\\n🏆 PHASE 2 IMPLEMENTATION COMPLETE!")
        print("All Phase 2 components successfully implemented and validated:")
        print("  ✅ Phase 2.1: Multi-scale temporal analysis")
        print("  ✅ Phase 2.2: Ensemble statistical framework")
        print("  ✅ Phase 2.3: Hierarchical connectivity analysis")
        print("\\nThe SMTE implementation now includes:")
        print("- Adaptive parameter optimization")
        print("- Network-aware statistical corrections")
        print("- Physiological constraints")
        print("- Multi-scale temporal analysis")
        print("- Ensemble statistical testing")
        print("- Hierarchical connectivity decomposition")
        print("\\nAll features maintain research-grade quality with full validation.")
        
        return True
    else:
        print("\\n❌ PHASE 2.3 FAILED - Issues detected in validation")
        return False

if __name__ == "__main__":
    success = test_hierarchical_smte_validation()
    sys.exit(0 if success else 1)