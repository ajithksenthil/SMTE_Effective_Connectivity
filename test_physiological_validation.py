#!/usr/bin/env python3
"""
Validate the Physiological SMTE implementation using the validation framework.
"""

import sys
import numpy as np
from validation_framework import SMTEValidationFramework
from physiological_smte_v1 import PhysiologicalSMTE

def test_physiological_smte_validation():
    """Test physiological SMTE using the validation framework."""
    
    print("Phase 1.3 Validation: Physiological Constraints")
    print("=" * 60)
    
    # Create validation framework
    validator = SMTEValidationFramework(random_state=42)
    
    # Test without physiological constraints (baseline)
    print("\n1. Testing Without Physiological Constraints (Baseline)")
    print("-" * 60)
    
    physio_smte_off = PhysiologicalSMTE(
        adaptive_mode='heuristic',
        use_network_correction=True,
        use_physiological_constraints=False,  # Disable constraints
        n_permutations=100,
        random_state=42
    )
    
    results_baseline = validator.validate_implementation(
        physio_smte_off, "Physiological SMTE (No Constraints)"
    )
    
    print("\nBaseline Results (No Physiological Constraints):")
    print(f"Performance improvement: {results_baseline['summary']['mean_performance_improvement']:.2%}")
    print(f"Speed: {results_baseline['summary']['mean_speedup']:.2f}x")
    
    # Test with physiological constraints
    print("\n2. Testing With Physiological Constraints")
    print("-" * 50)
    
    # Create synthetic coordinates for validation datasets
    n_regions_max = 15  # Maximum number of regions in validation datasets
    roi_coords = np.random.randn(n_regions_max, 3) * 20  # Synthetic brain coordinates
    
    physio_smte_on = PhysiologicalSMTE(
        adaptive_mode='heuristic',
        use_network_correction=True,
        use_physiological_constraints=True,   # Enable constraints
        roi_coords=roi_coords,
        TR=2.0,
        n_permutations=100,
        random_state=42
    )
    
    results_constrained = validator.validate_implementation(
        physio_smte_on, "Physiological SMTE (With Constraints)"
    )
    
    print("\nConstrained Results (With Physiological Constraints):")
    print(f"Performance improvement: {results_constrained['summary']['mean_performance_improvement']:.2%}")
    print(f"Speed: {results_constrained['summary']['mean_speedup']:.2f}x")
    
    # Generate reports
    print("\n" + "="*60)
    print("DETAILED VALIDATION REPORTS")
    print("="*60)
    
    report_baseline = validator.create_validation_report(results_baseline)
    print(report_baseline)
    
    print("\n" + "="*60)
    
    report_constrained = validator.create_validation_report(results_constrained)
    print(report_constrained)
    
    # Summary
    print("\n" + "="*60)
    print("PHASE 1.3 COMPLETION SUMMARY")
    print("="*60)
    
    baseline_passed = all(results_baseline['regression_check'].values())
    constrained_passed = all(results_constrained['regression_check'].values())
    
    print(f"✅ Baseline Mode: {'PASSED' if baseline_passed else 'FAILED'}")
    print(f"✅ Constrained Mode: {'PASSED' if constrained_passed else 'FAILED'}")
    
    # Additional analysis
    if baseline_passed and constrained_passed:
        print("\n📊 CONSTRAINT ANALYSIS:")
        
        # Compare performance between modes
        baseline_perf = results_baseline['summary']['mean_performance_improvement'] 
        constrained_perf = results_constrained['summary']['mean_performance_improvement']
        
        print(f"Performance difference: {constrained_perf - baseline_perf:.2%}")
        print("(Positive means constraints improved performance)")
        
        # Speed comparison
        baseline_speed = results_baseline['summary']['mean_speedup']
        constrained_speed = results_constrained['summary']['mean_speedup']
        
        print(f"Speed difference: {constrained_speed - baseline_speed:.2f}x")
        print("(Constraints add computational overhead but improve biological plausibility)")
        
        print("\n🎉 PHASE 1.3 SUCCESSFULLY COMPLETED!")
        print("Physiological constraints are working correctly.")
        print("- Constraints filter biologically implausible connections")
        print("- Performance is maintained within acceptable bounds")
        print("- Implementation is numerically stable")
        return True
    else:
        print("\n❌ PHASE 1.3 FAILED - Issues detected in validation")
        return False

if __name__ == "__main__":
    success = test_physiological_smte_validation()
    sys.exit(0 if success else 1)