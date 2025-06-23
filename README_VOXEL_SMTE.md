# fMRI Voxel Effective Connectivity Analysis using Symbolic Matrix Transfer Entropy

This repository provides a complete implementation of voxel-wise effective connectivity analysis for fMRI data using Symbolic Matrix Transfer Entropy (SMTE). The implementation is designed for computational efficiency, statistical rigor, and ease of use with neuroimaging data.

## Overview

The implementation includes:

1. **Multiple Symbolization Methods**: Convert continuous fMRI time series to symbolic sequences
2. **Efficient SMTE Computation**: Parallel computation of transfer entropy between voxel pairs
3. **Statistical Significance Testing**: Permutation tests with FDR correction
4. **Network Analysis**: Graph-based connectivity analysis and visualization
5. **Memory-Efficient Processing**: Handle large datasets with chunked processing

## Key Features

### Symbolization Methods

- **Uniform Discretization**: Equal-width binning of time series values
- **Quantile Discretization**: Equal-frequency binning based on data distribution
- **Ordinal Patterns**: Rank-order patterns capturing temporal dynamics
- **Vector Quantization**: K-means clustering for symbolic representation

### Transfer Entropy Computation

- **Multiple Time Lags**: Automatically finds optimal lag for each voxel pair
- **Histogram-based Estimation**: Efficient entropy computation using joint histograms
- **Non-negative Guarantee**: Ensures theoretically valid transfer entropy values

### Statistical Testing

- **Permutation Tests**: Circular shuffling to generate null distributions
- **FDR Correction**: Benjamini-Hochberg correction for multiple comparisons
- **Significance Thresholding**: Robust identification of significant connections

## Installation

```bash
# Install required packages
pip install -r requirements.txt

# Or install individually
pip install numpy scipy nibabel scikit-learn networkx matplotlib seaborn numba pandas
```

## Usage

### Basic Usage

```python
from voxel_smte_connectivity import VoxelSMTEConnectivity

# Initialize analyzer
analyzer = VoxelSMTEConnectivity(
    n_symbols=5,           # Number of symbols for discretization
    symbolizer='uniform',  # Symbolization method
    max_lag=5,            # Maximum time lag to consider
    alpha=0.05,           # Statistical significance threshold
    n_permutations=1000,  # Number of permutations for testing
    n_jobs=-1,            # Use all available cores
    memory_efficient=True # Enable memory-efficient processing
)

# Run complete analysis
results = analyzer.run_complete_analysis(
    fmri_path='path/to/fmri_data.nii.gz',
    output_dir='output_directory/',
    mask_path='path/to/brain_mask.nii.gz',  # Optional
    visualize=True
)
```

### Step-by-Step Analysis

```python
# 1. Load fMRI data
analyzer.load_fmri_data('fmri_data.nii.gz', 'brain_mask.nii.gz')

# 2. Extract voxel time series
voxel_timeseries = analyzer.extract_voxel_timeseries()

# 3. Symbolize time series
symbolic_data = analyzer.symbolize_timeseries(voxel_timeseries)
analyzer.symbolic_data = symbolic_data

# 4. Compute connectivity matrix
smte_matrix, lag_matrix = analyzer.compute_voxel_connectivity_matrix()

# 5. Statistical testing
p_values = analyzer.statistical_testing(smte_matrix)

# 6. FDR correction
significance_mask = analyzer.fdr_correction(p_values)

# 7. Build connectivity graph
graph = analyzer.build_connectivity_graph(smte_matrix, significance_mask)

# 8. Analyze network properties
properties = analyzer.analyze_network_properties(graph)
```

### Analyzing Specific Voxels

```python
# Analyze only specific voxels (e.g., ROI-based analysis)
roi_voxel_indices = [10, 25, 50, 75, 100]  # Voxel indices within mask

results = analyzer.run_complete_analysis(
    fmri_path='fmri_data.nii.gz',
    output_dir='roi_analysis/',
    voxel_indices=roi_voxel_indices,
    visualize=True
)
```

## Method Details

### Symbolic Matrix Transfer Entropy (SMTE)

SMTE quantifies directed information flow between time series by:

1. **Symbolization**: Converting continuous time series to discrete symbol sequences
2. **Gram Matrix Construction**: Creating matrices that capture symbol co-occurrence patterns
3. **Matrix Entropy**: Computing von Neumann entropy of Gram matrices
4. **Transfer Entropy**: Measuring reduction in uncertainty when conditioning on source

The transfer entropy from time series Y to X at lag τ is:

```
SMTE(Y→X,τ) = H(X_t|X_{t-1}) - H(X_t|X_{t-1}, Y_{t-τ})
```

Where H denotes the matrix entropy and the optimal lag τ is selected by maximizing SMTE.

### Statistical Significance

Significance is assessed using:

1. **Permutation Testing**: Generate null distribution by circular shuffling
2. **P-value Computation**: Compare observed SMTE to null distribution
3. **FDR Correction**: Control false discovery rate using Benjamini-Hochberg procedure

### Memory Efficiency

For large datasets, the implementation uses:

- **Chunked Processing**: Process voxel pairs in smaller chunks
- **Parallel Computation**: Utilize multiple CPU cores
- **Optimized Data Structures**: Minimize memory footprint

## Output Files

The analysis generates several output files:

- `smte_matrix.npy`: Full SMTE connectivity matrix
- `p_values.npy`: Statistical significance p-values
- `significance_mask.npy`: Boolean mask of significant connections
- `connectivity_graph.graphml`: Network graph in GraphML format
- `network_properties.json`: Computed network metrics

## Visualization

The implementation provides several visualization options:

### Connectivity Matrix
```python
analyzer.visualize_connectivity_matrix(
    smte_matrix, 
    significance_mask, 
    title="Voxel-wise SMTE Connectivity"
)
```

### Network Properties
```python
properties = analyzer.analyze_network_properties(connectivity_graph)
print(f"Network density: {properties['density']:.4f}")
print(f"Number of hub nodes: {len(properties['hub_nodes'])}")
```

## Testing

Run the comprehensive test suite:

```bash
python test_voxel_connectivity.py
```

The test suite includes:

1. **Symbolization Methods**: Comparison of different symbolization approaches
2. **SMTE Computation**: Validation with synthetic data with known dependencies
3. **Full Pipeline**: End-to-end test with synthetic fMRI data
4. **Performance Benchmark**: Speed and memory usage analysis

## Performance Considerations

### Computational Complexity

- **Time Complexity**: O(N²·T·L) where N=voxels, T=timepoints, L=max_lag
- **Space Complexity**: O(N²) for storing connectivity matrices
- **Parallelization**: Near-linear speedup with multiple cores

### Optimization Tips

1. **Reduce Voxels**: Use brain masks or ROI-based analysis
2. **Chunk Processing**: Enable memory_efficient=True for large datasets
3. **Parallel Processing**: Set n_jobs to number of available cores
4. **Symbol Count**: Balance between resolution (more symbols) and computational cost

### Memory Usage

Approximate memory requirements:

- **1000 voxels**: ~8 GB RAM
- **5000 voxels**: ~200 GB RAM
- **10000 voxels**: ~800 GB RAM

Use chunked processing for large analyses.

## Applications

### Neuroimaging Studies

- **Effective Connectivity**: Identify directed connections between brain regions
- **Network Hubs**: Find highly connected brain areas
- **Connectivity Changes**: Compare connectivity between conditions or groups
- **Clinical Applications**: Study connectivity alterations in neurological/psychiatric disorders

### Methodological Advantages

1. **Model-Free**: No assumptions about underlying connectivity model
2. **Nonlinear Dependencies**: Captures complex, nonlinear relationships
3. **Directionality**: Provides information about direction of influence
4. **Statistical Rigor**: Robust significance testing with multiple comparison correction

## Citation

If you use this implementation in your research, please cite:

```
[Your paper/repository citation here]
```

## References

1. Symbolic Transfer Entropy: [Original methodology papers]
2. Matrix-based Transfer Entropy: [Theoretical foundations]
3. fMRI Connectivity Analysis: [Neuroimaging applications]

## License

[Specify your license here]

## Support

For questions, issues, or contributions:

- Open an issue on GitHub
- Contact: [your email]
- Documentation: [link to documentation]

## Changelog

### Version 1.0.0
- Initial implementation
- Multiple symbolization methods
- Statistical significance testing
- Network analysis tools
- Comprehensive test suite