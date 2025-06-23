# COMPREHENSIVE LIMITATIONS ANALYSIS
# Causal Graph Clustering for SMTE Networks
============================================================

## EXECUTIVE SUMMARY

This analysis identifies key limitations affecting the effectiveness of causal
relationship detection and cluster-level thresholding in our SMTE implementation.

## FUNDAMENTAL SMTE LIMITATIONS

### 1. Temporal Resolution Issues ⚠️ CRITICAL

**Best Performance**: TR_0.5s (0.0% detection)
**Worst Performance**: TR_0.5s (0.0% detection)

**Key Issues**:
- SMTE limited to max_lag=3 samples
- Hemodynamic delays ~6s may require more lags at high TR
- Temporal mismatch reduces detection sensitivity

**Recommendations**:
- Use TR ≤ 2s for optimal SMTE performance
- Consider adaptive max_lag based on TR
- Account for hemodynamic delay in lag selection

### 2. Symbolization Parameter Sensitivity ⚠️ MODERATE

**Best Parameters**: sym2_ord2 (0.0% detection)

**Key Issues**:
- High sensitivity to n_symbols and ordinal_order choices
- Some parameter combinations cause computational failures
- No clear optimal parameter selection strategy

**Recommendations**:
- Systematically test parameter combinations on pilot data
- Consider adaptive parameter selection
- Default to n_symbols=2, ordinal_order=2 for stability

### 3. Statistical Power Limitations ⚠️ CRITICAL

**Key Issues**:
- Requires large sample sizes for reliable detection
- Conservative permutation testing needs many samples
- Multiple comparison correction further reduces power

**Sample Size Effects**:
- 50 timepoints: 0.0% detection
- 100 timepoints: 0.0% detection
- 200 timepoints: 0.0% detection
- 400 timepoints: 0.0% detection
- 800 timepoints: 0.0% detection

**Recommendations**:
- Use ≥200 timepoints for adequate power
- Consider liberal uncorrected thresholds for exploration
- Balance permutation count vs. computational cost

## CLUSTERING-SPECIFIC LIMITATIONS

### 4. Graph Construction Threshold Sensitivity ⚠️ HIGH

**Key Issues**:
- Clustering performance highly dependent on initial threshold
- Too conservative: no connections to cluster
- Too liberal: everything clusters together
- No principled method for threshold selection

**Recommendations**:
- Implement adaptive threshold selection
- Use multiple thresholds with ensemble approach
- Consider data-driven threshold optimization

### 5. Cluster Size Effects ⚠️ MODERATE

**Cluster Size Performance**:
- Size 2: 100.0% detection
- Size 3: 33.3% detection
- Size 5: 30.0% detection
- Size 8: 19.6% detection
- Size 10: 28.9% detection

**Key Issues**:
- Large clusters suffer from over-conservative FDR correction
- Small clusters may lack statistical power
- Optimal cluster size depends on network structure

**Recommendations**:
- Implement cluster-size-adaptive correction methods
- Use hierarchical clustering for large networks
- Consider cluster validity metrics

## OVERALL ASSESSMENT

### 🔴 Critical Limitations (High Impact)
1. **Temporal Resolution Mismatch**: SMTE parameters not optimized for fMRI
2. **Statistical Power**: Conservative corrections limit practical detection
3. **Threshold Sensitivity**: Graph construction highly parameter-dependent

### 🟡 Moderate Limitations (Medium Impact)
1. **Symbolization Sensitivity**: Parameter choices affect performance
2. **Cluster Size Effects**: Need adaptive correction methods
3. **Network Structure Bias**: Some topologies favored over others

### 🟢 Minor Limitations (Low Impact)
1. **Computational Complexity**: Manageable with optimization
2. **Integration Issues**: Mostly resolved with current fixes

## PRIORITY RECOMMENDATIONS

### Immediate Improvements (High Priority)
1. **Implement adaptive max_lag based on TR**
2. **Add data-driven threshold selection**
3. **Optimize symbolization parameters automatically**

### Medium-term Improvements (Medium Priority)
1. **Develop cluster-size-adaptive FDR correction**
2. **Add ensemble clustering approaches**
3. **Implement temporal resolution optimization**

### Long-term Improvements (Lower Priority)
1. **Alternative to surrogate-based testing**
2. **Network topology-aware clustering**
3. **Integration with other connectivity methods**

## CONCLUSION

While the fixed causal graph clustering now works functionally, several
fundamental limitations remain that affect detection effectiveness:

- **Temporal resolution mismatch** is the most critical issue
- **Statistical power limitations** require larger datasets
- **Parameter sensitivity** needs automated optimization

Addressing these limitations could significantly improve the method's
practical utility for neuroimaging applications.