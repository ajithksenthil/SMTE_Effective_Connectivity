# Connectivity Methods Benchmark Report
==================================================

## Executive Summary

**Best Overall Method (ROC AUC):** Lagged_Correlation (0.952)

## Performance by Metric

**AUC_ROC:** Lagged_Correlation (0.952)
**AUC_PR:** Lagged_Correlation (0.732)
**F1_SCORE:** Lagged_Correlation (0.164)
**ACCURACY:** Granger_Causality (0.955)
**OPTIMAL_SENSITIVITY:** Lagged_Correlation (0.944)
**OPTIMAL_SPECIFICITY:** Granger_Causality (1.000)

## SMTE Performance Analysis

### Overall SMTE Performance:
| Metric | Mean | Std |
|--------|------|-----|
| AUC_ROC | 0.586 | 0.178 |
| AUC_PR | 0.211 | 0.226 |
| F1_SCORE | 0.096 | 0.046 |
| ACCURACY | 0.507 | 0.025 |

### Statistical Significance (SMTE vs Others):

**AUC_ROC:**
- vs Pearson_Correlation: p=0.0000 ✓
- vs Lagged_Correlation: p=0.0000 ✓
- vs Partial_Correlation: p=0.3555 ✗
- vs Mutual_Information: p=0.0204 ✓
- vs Granger_Causality: p=0.3335 ✗
- vs Phase_Lag_Index: p=0.0000 ✓
- vs Coherence: p=0.0000 ✓

**F1_SCORE:**
- vs Pearson_Correlation: p=0.0257 ✓
- vs Lagged_Correlation: p=0.0000 ✓
- vs Partial_Correlation: p=0.0000 ✓
- vs Mutual_Information: p=0.0014 ✓
- vs Granger_Causality: p=0.0000 ✓
- vs Phase_Lag_Index: p=0.0000 ✓
- vs Coherence: p=0.0000 ✓

## Performance by Condition

### By Coupling Type (ROC AUC):
| Coupling Type | SMTE AUC |
|---------------|----------|
| Linear | 0.635 |
| Nonlinear | 0.495 |
| Mixed | 0.627 |

## Computational Efficiency

| Method | Mean Time (s) | Median Time (s) |
|--------|---------------|-----------------|
| Coherence | 0.017 | 0.016 |
| Granger_Causality | 0.000 | 0.000 |
| Lagged_Correlation | 0.149 | 0.148 |
| Mutual_Information | 0.017 | 0.017 |
| Partial_Correlation | 0.027 | 0.027 |
| Pearson_Correlation | 0.030 | 0.030 |
| Phase_Lag_Index | 0.001 | 0.001 |
| SMTE | 0.074 | 0.074 |

## Recommendations

- **SMTE Overall Ranking:** #5 in ROC AUC, #5 in F1 Score
- **SMTE Use Cases:** Recommended for nonlinear, complex connectivity analysis
- **Limitations:** Higher computational cost compared to correlation-based methods