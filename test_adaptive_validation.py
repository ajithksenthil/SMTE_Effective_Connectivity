#!/usr/bin/env python3
"""
Validate the Adaptive SMTE implementation using the validation framework.
"""

import sys
import numpy as np
from validation_framework import SMTEValidationFramework
from adaptive_smte_v1 import AdaptiveSMTE

def test_adaptive_smte_validation():
    """Test adaptive SMTE using the validation framework."""
    
    print("Phase 1.1 Validation: Adaptive Parameter Selection")
    print("=" * 60)
    
    # Create validation framework
    validator = SMTEValidationFramework(random_state=42)
    
    # Test heuristic mode
    print("\n1. Testing Heuristic Mode")
    print("-" * 40)
    
    adaptive_heuristic = AdaptiveSMTE(
        adaptive_mode='heuristic',
        n_permutations=100,  # Standard for validation
        random_state=42
    )
    
    results_heuristic = validator.validate_implementation(
        adaptive_heuristic, "Adaptive SMTE (Heuristic)"
    )
    
    print("\nHeuristic Mode Results:")
    print(f"Performance improvement: {results_heuristic['summary']['mean_performance_improvement']:.2%}")
    print(f"Speed: {results_heuristic['summary']['mean_speedup']:.2f}x")
    
    # Test grid search mode
    print("\n2. Testing Grid Search Mode")
    print("-" * 40)
    
    adaptive_grid = AdaptiveSMTE(
        adaptive_mode='grid_search',
        quick_optimization=True,
        n_permutations=100,
        random_state=42
    )
    
    results_grid = validator.validate_implementation(
        adaptive_grid, "Adaptive SMTE (Grid Search)"
    )
    
    print("\nGrid Search Mode Results:")
    print(f"Performance improvement: {results_grid['summary']['mean_performance_improvement']:.2%}")
    print(f"Speed: {results_grid['summary']['mean_speedup']:.2f}x")
    
    # Generate reports
    print("\n" + "="*60)
    print("DETAILED VALIDATION REPORTS")
    print("="*60)
    
    report_heuristic = validator.create_validation_report(results_heuristic)
    print(report_heuristic)
    
    print("\n" + "="*60)
    
    report_grid = validator.create_validation_report(results_grid)
    print(report_grid)
    
    # Summary
    print("\n" + "="*60)
    print("PHASE 1.1 COMPLETION SUMMARY")
    print("="*60)
    
    heuristic_passed = all(results_heuristic['regression_check'].values())
    grid_passed = all(results_grid['regression_check'].values())
    
    print(f"✅ Heuristic Mode: {'PASSED' if heuristic_passed else 'FAILED'}")
    print(f"✅ Grid Search Mode: {'PASSED' if grid_passed else 'FAILED'}")
    
    if heuristic_passed and grid_passed:
        print("\n🎉 PHASE 1.1 SUCCESSFULLY COMPLETED!")
        print("Adaptive parameter selection is working correctly.")
        return True
    else:
        print("\n❌ PHASE 1.1 FAILED - Issues detected in validation")
        return False

if __name__ == "__main__":
    success = test_adaptive_smte_validation()
    sys.exit(0 if success else 1)