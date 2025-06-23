# Optimized Real fMRI Data Validation Report
==================================================

## Dataset Summary

**Regions analyzed:** 15
**Methods tested:** 4

## Method Performance Summary

| Method | Total Connections | Status |
|--------|------------------|--------|
| SMTE | 0 | ✓ Success |
| Lagged_Correlation | 12 | ✓ Success |
| Mutual_Information | 12 | ✓ Success |
| Partial_Correlation | 12 | ✓ Success |

## Network Validation Results

### SMTE

| Network | Coverage | Mean Strength | Modularity |
|---------|----------|---------------|------------|
| Default Mode Network | 0.800 | 0.1238 | 0.0305 |
| Executive Control Network | 0.250 | 0.1196 | 0.0538 |
| Sensorimotor Network | 0.750 | 0.1159 | 0.0200 |

### Lagged_Correlation

| Network | Coverage | Mean Strength | Modularity |
|---------|----------|---------------|------------|
| Default Mode Network | 0.600 | 0.4441 | 0.1210 |
| Executive Control Network | 0.750 | 0.4619 | 0.2882 |
| Sensorimotor Network | 1.000 | 0.5301 | 0.0685 |

### Mutual_Information

| Network | Coverage | Mean Strength | Modularity |
|---------|----------|---------------|------------|
| Default Mode Network | 0.000 | 0.2223 | 0.0155 |
| Executive Control Network | 0.500 | 0.2646 | 0.1626 |
| Sensorimotor Network | 0.750 | 0.3223 | 0.1137 |

### Partial_Correlation

| Network | Coverage | Mean Strength | Modularity |
|---------|----------|---------------|------------|
| Default Mode Network | 0.200 | 0.0004 | 0.0004 |
| Executive Control Network | 0.500 | 0.0337 | 0.0337 |
| Sensorimotor Network | 0.500 | 0.0911 | 0.0911 |

## SMTE Specific Analysis

**Significant connections:** 0

## Overall Assessment

**Methods successfully completed:** 4/4
**SMTE Status:** ⚠ No significant connections detected (consider parameter adjustment)
**SMTE Network Detection:** 60.0% average coverage of expected connections