# REAL fMRI STUDY REPLICATION REPORT
## Enhanced SMTE Framework Validation
================================================================================

## STUDY REPLICATION DETAILS
----------------------------------------
**Target Study**: Panikratova et al. (2020)
**Dataset**: OpenNeuro ds002422
**Analysis Method**: ENHANCED_SMTE
**Timestamp**: 2025-06-22 17:27:43

## METHODOLOGY
--------------------
**Target Region**: DLPFC
**ROIs Analyzed**: 10 regions
**ROI Labels**: DLPFC_L, DLPFC_R, mPFC, PCC, M1_L, M1_R, V1_L, V1_R, Cerebellum_L, Cerebellum_R
**SMTE Parameters**:
  - n_symbols: 3
  - ordinal_order: 2
  - max_lag: 3
  - n_permutations: 200
  - alpha: 0.05
  - use_fdr_correction: True

## VALIDATION RESULTS
------------------------------
## VALIDATION LOG
--------------------
[17:27:43] Dataset Check: CHECKING
    Looking for ds002422 data
[17:27:43] Dataset Check: NOT_FOUND
    Dataset not available locally
[17:27:43] Data Simulation: CREATING
    Generating realistic simulation based on study parameters
[17:27:43] Data Simulation: SUCCESS
    Created 20 subjects with 10 ROIs, 200 timepoints
[17:27:43] SMTE Analysis: STARTING
    Analyzing 20 subjects with enhanced SMTE
[17:27:43] Compatibility Check: VERIFYING
    Checking backward compatibility
[17:27:43] Compatibility Check: ERROR
    Compatibility issue: VoxelSMTEConnectivity.__init__() got an unexpected keyword argument 'use_fdr_correction'
[17:27:43] Results Validation: STARTING
    Validating analysis results
[17:27:43] Results Validation: FAILED
    No results to validate

## CONCLUSIONS
--------------------
⚠️ **REPLICATION INCOMPLETE**: Issues encountered during analysis.
   Review validation log for detailed error information.

## DATA AND CODE AVAILABILITY
----------------------------------------
**Dataset**: OpenNeuro ds002422
**Code**: Enhanced SMTE Framework (all implementations)
**Reproducibility**: Random seed 42, deterministic analysis
**Validation**: Comprehensive validation framework included