#!/usr/bin/env python3
"""
Validate the Network-Aware SMTE implementation using the validation framework.
"""

import sys
import numpy as np
from validation_framework import SMTEValidationFramework
from network_aware_smte_v1 import NetworkAwareSMTE

def test_network_aware_smte_validation():
    """Test network-aware SMTE using the validation framework."""
    
    print("Phase 1.2 Validation: Network-Aware Statistical Correction")
    print("=" * 70)
    
    # Create validation framework
    validator = SMTEValidationFramework(random_state=42)
    
    # Test standard mode (without network correction)
    print("\n1. Testing Standard Mode (for comparison)")
    print("-" * 50)
    
    network_aware_standard = NetworkAwareSMTE(
        adaptive_mode='heuristic',
        use_network_correction=False,  # Disable network correction
        n_permutations=100,
        random_state=42
    )
    
    results_standard = validator.validate_implementation(
        network_aware_standard, "Network-Aware SMTE (Standard)"
    )
    
    print("\nStandard Mode Results:")
    print(f"Performance improvement: {results_standard['summary']['mean_performance_improvement']:.2%}")
    print(f"Speed: {results_standard['summary']['mean_speedup']:.2f}x")
    
    # Test network-aware mode
    print("\n2. Testing Network-Aware Mode")
    print("-" * 40)
    
    # Create known networks for validation data
    known_networks = {
        'network_a': [0, 1, 2],
        'network_b': [3, 4, 5], 
        'network_c': [6, 7, 8]
    }
    
    network_aware_enhanced = NetworkAwareSMTE(
        adaptive_mode='heuristic',
        use_network_correction=True,   # Enable network correction
        known_networks=known_networks,
        n_permutations=100,
        random_state=42
    )
    
    results_enhanced = validator.validate_implementation(
        network_aware_enhanced, "Network-Aware SMTE (Enhanced)"
    )
    
    print("\nNetwork-Aware Mode Results:")
    print(f"Performance improvement: {results_enhanced['summary']['mean_performance_improvement']:.2%}")
    print(f"Speed: {results_enhanced['summary']['mean_speedup']:.2f}x")
    
    # Generate reports
    print("\n" + "="*70)
    print("DETAILED VALIDATION REPORTS")
    print("="*70)
    
    report_standard = validator.create_validation_report(results_standard)
    print(report_standard)
    
    print("\n" + "="*70)
    
    report_enhanced = validator.create_validation_report(results_enhanced)
    print(report_enhanced)
    
    # Summary
    print("\n" + "="*70)
    print("PHASE 1.2 COMPLETION SUMMARY")
    print("="*70)
    
    standard_passed = all(results_standard['regression_check'].values())
    enhanced_passed = all(results_enhanced['regression_check'].values())
    
    print(f"✅ Standard Mode: {'PASSED' if standard_passed else 'FAILED'}")
    print(f"✅ Network-Aware Mode: {'PASSED' if enhanced_passed else 'FAILED'}")
    
    if standard_passed and enhanced_passed:
        print("\n🎉 PHASE 1.2 SUCCESSFULLY COMPLETED!")
        print("Network-aware statistical correction is working correctly.")
        return True
    else:
        print("\n❌ PHASE 1.2 FAILED - Issues detected in validation")
        return False

if __name__ == "__main__":
    success = test_network_aware_smte_validation()
    sys.exit(0 if success else 1)