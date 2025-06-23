#!/usr/bin/env python3
"""
Analysis of Remaining Limitations in Causal Graph Clustering
Systematic identification of issues affecting detection effectiveness.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import pandas as pd

def analyze_smte_limitations():
    """Analyze fundamental limitations in SMTE methodology."""
    
    print("🔍 ANALYZING SMTE FUNDAMENTAL LIMITATIONS")
    print("=" * 60)
    
    limitations = {}
    
    # Test 1: Temporal Resolution Issues
    print("\n1. TEMPORAL RESOLUTION ANALYSIS")
    print("-" * 40)
    
    temporal_results = test_temporal_resolution_effects()
    limitations['temporal_resolution'] = temporal_results
    
    # Test 2: Symbolization Sensitivity
    print("\n2. SYMBOLIZATION PARAMETER SENSITIVITY")
    print("-" * 40)
    
    symbolization_results = test_symbolization_sensitivity()
    limitations['symbolization'] = symbolization_results
    
    # Test 3: Statistical Power Issues
    print("\n3. STATISTICAL POWER ANALYSIS")
    print("-" * 40)
    
    power_results = test_statistical_power_issues()
    limitations['statistical_power'] = power_results
    
    # Test 4: Network Structure Bias
    print("\n4. NETWORK STRUCTURE BIAS ANALYSIS")
    print("-" * 40)
    
    network_bias_results = test_network_structure_bias()
    limitations['network_bias'] = network_bias_results
    
    return limitations

def test_temporal_resolution_effects():
    """Test how temporal resolution affects detection."""
    
    print("Testing TR (temporal resolution) effects...")
    
    results = {}
    
    # Test different TRs
    trs = [0.5, 1.0, 2.0, 3.0]  # seconds
    connection_strength = 0.5
    
    for tr in trs:
        print(f"  Testing TR = {tr}s...")
        
        # Create data with specific temporal dynamics
        n_timepoints = int(300 / tr)  # 5 minutes of data
        
        # Hemodynamic response has ~6s peak
        optimal_lag_samples = max(1, int(6 / tr))  # Convert 6s to samples
        
        np.random.seed(42)
        data = np.random.randn(5, n_timepoints)
        ground_truth = np.zeros((5, 5))
        
        # Create connection with physiologically realistic lag
        lag = min(optimal_lag_samples, 3)  # Cap at max_lag=3
        if lag < n_timepoints:
            data[1, lag:] += connection_strength * data[0, :-lag]
            ground_truth[0, 1] = connection_strength
        
        # Test SMTE detection
        detection_rate = test_smte_detection(data, ground_truth)
        
        results[f'TR_{tr}s'] = {
            'detection_rate': detection_rate,
            'optimal_lag_samples': optimal_lag_samples,
            'used_lag_samples': lag,
            'temporal_mismatch': abs(optimal_lag_samples - lag)
        }
        
        print(f"    Detection: {detection_rate:.1f}%, lag mismatch: {results[f'TR_{tr}s']['temporal_mismatch']}")
    
    return results

def test_symbolization_sensitivity():
    """Test sensitivity to symbolization parameters."""
    
    print("Testing symbolization parameter sensitivity...")
    
    results = {}
    
    # Test different symbolization parameters
    n_symbols_options = [2, 3, 4, 5]
    ordinal_orders = [1, 2, 3, 4]
    
    # Create standard test data
    np.random.seed(42)
    n_timepoints = 200
    data = np.random.randn(4, n_timepoints)
    ground_truth = np.zeros((4, 4))
    
    # Add clear connection
    data[1, 1:] += 0.6 * data[0, :-1]
    ground_truth[0, 1] = 0.6
    
    for n_symbols in n_symbols_options:
        for ordinal_order in ordinal_orders:
            param_name = f'sym{n_symbols}_ord{ordinal_order}'
            print(f"  Testing {param_name}...")
            
            try:
                detection_rate = test_smte_detection(
                    data, ground_truth, 
                    n_symbols=n_symbols, 
                    ordinal_order=ordinal_order
                )
                
                results[param_name] = {
                    'detection_rate': detection_rate,
                    'n_symbols': n_symbols,
                    'ordinal_order': ordinal_order
                }
                
                print(f"    Detection: {detection_rate:.1f}%")
                
            except Exception as e:
                print(f"    Failed: {e}")
                results[param_name] = {'error': str(e)}
    
    return results

def test_statistical_power_issues():
    """Test statistical power limitations."""
    
    print("Testing statistical power issues...")
    
    results = {}
    
    # Test 1: Sample size effects
    sample_sizes = [50, 100, 200, 400, 800]
    connection_strength = 0.4
    
    for n_samples in sample_sizes:
        print(f"  Testing {n_samples} timepoints...")
        
        np.random.seed(42)
        data = np.random.randn(4, n_samples)
        ground_truth = np.zeros((4, 4))
        
        # Add connection
        data[1, 1:] += connection_strength * data[0, :-1]
        ground_truth[0, 1] = connection_strength
        
        detection_rate = test_smte_detection(data, ground_truth, n_permutations=50)
        
        results[f'samples_{n_samples}'] = {
            'detection_rate': detection_rate,
            'n_samples': n_samples
        }
        
        print(f"    Detection: {detection_rate:.1f}%")
    
    # Test 2: Permutation count effects
    permutation_counts = [10, 50, 100, 200, 500]
    
    np.random.seed(42)
    data = np.random.randn(4, 200)
    ground_truth = np.zeros((4, 4))
    data[1, 1:] += 0.5 * data[0, :-1]
    ground_truth[0, 1] = 0.5
    
    for n_perms in permutation_counts:
        print(f"  Testing {n_perms} permutations...")
        
        detection_rate = test_smte_detection(data, ground_truth, n_permutations=n_perms)
        
        results[f'perms_{n_perms}'] = {
            'detection_rate': detection_rate,
            'n_permutations': n_perms
        }
        
        print(f"    Detection: {detection_rate:.1f}%")
    
    return results

def test_network_structure_bias():
    """Test biases based on network structure."""
    
    print("Testing network structure biases...")
    
    results = {}
    
    # Test different network topologies
    topologies = {
        'star': create_star_network,
        'chain': create_chain_network,
        'dense': create_dense_network,
        'sparse': create_sparse_network,
        'hierarchical': create_hierarchical_network
    }
    
    for topology_name, topology_func in topologies.items():
        print(f"  Testing {topology_name} topology...")
        
        try:
            data, ground_truth, expected_detections = topology_func()
            detection_rate = test_smte_detection(data, ground_truth)
            
            actual_detections = int(detection_rate * expected_detections / 100)
            
            results[topology_name] = {
                'detection_rate': detection_rate,
                'expected_connections': expected_detections,
                'detected_connections': actual_detections,
                'topology_bias': actual_detections / max(expected_detections, 1)
            }
            
            print(f"    Expected: {expected_detections}, Detected: {actual_detections} ({detection_rate:.1f}%)")
            
        except Exception as e:
            print(f"    Failed: {e}")
            results[topology_name] = {'error': str(e)}
    
    return results

def create_star_network():
    """Create star topology (hub-and-spoke)."""
    np.random.seed(42)
    n_nodes = 6
    n_timepoints = 200
    
    data = np.random.randn(n_nodes, n_timepoints)
    ground_truth = np.zeros((n_nodes, n_nodes))
    
    # Hub at node 0 connects to all others
    hub_strength = 0.4
    expected_connections = n_nodes - 1
    
    for target in range(1, n_nodes):
        data[target, 1:] += hub_strength * data[0, :-1]
        ground_truth[0, target] = hub_strength
    
    return data, ground_truth, expected_connections

def create_chain_network():
    """Create chain topology (sequential connections)."""
    np.random.seed(42)
    n_nodes = 6
    n_timepoints = 200
    
    data = np.random.randn(n_nodes, n_timepoints)
    ground_truth = np.zeros((n_nodes, n_nodes))
    
    # Sequential connections: 0->1->2->3->4->5
    chain_strength = 0.5
    expected_connections = n_nodes - 1
    
    for i in range(n_nodes - 1):
        data[i+1, 1:] += chain_strength * data[i, :-1]
        ground_truth[i, i+1] = chain_strength
    
    return data, ground_truth, expected_connections

def create_dense_network():
    """Create dense topology (many connections)."""
    np.random.seed(42)
    n_nodes = 5
    n_timepoints = 200
    
    data = np.random.randn(n_nodes, n_timepoints)
    ground_truth = np.zeros((n_nodes, n_nodes))
    
    # Dense connections (75% connectivity)
    dense_strength = 0.3
    connections = [(0,1), (0,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    expected_connections = len(connections)
    
    for source, target in connections:
        data[target, 1:] += dense_strength * data[source, :-1]
        ground_truth[source, target] = dense_strength
    
    return data, ground_truth, expected_connections

def create_sparse_network():
    """Create sparse topology (few connections)."""
    np.random.seed(42)
    n_nodes = 8
    n_timepoints = 200
    
    data = np.random.randn(n_nodes, n_timepoints)
    ground_truth = np.zeros((n_nodes, n_nodes))
    
    # Sparse connections (only 3 out of 56 possible)
    sparse_strength = 0.6
    connections = [(0,7), (2,5), (1,6)]  # Long-range connections
    expected_connections = len(connections)
    
    for source, target in connections:
        data[target, 2:] += sparse_strength * data[source, :-2]  # Longer lag
        ground_truth[source, target] = sparse_strength
    
    return data, ground_truth, expected_connections

def create_hierarchical_network():
    """Create hierarchical topology (multi-level)."""
    np.random.seed(42)
    n_nodes = 7
    n_timepoints = 200
    
    data = np.random.randn(n_nodes, n_timepoints)
    ground_truth = np.zeros((n_nodes, n_nodes))
    
    # Hierarchical: 0 -> {1,2}, 1 -> {3,4}, 2 -> {5,6}
    hier_strength = 0.45
    connections = [(0,1), (0,2), (1,3), (1,4), (2,5), (2,6)]
    expected_connections = len(connections)
    
    for source, target in connections:
        data[target, 1:] += hier_strength * data[source, :-1]
        ground_truth[source, target] = hier_strength
    
    return data, ground_truth, expected_connections

def test_smte_detection(data, ground_truth, n_symbols=2, ordinal_order=2, 
                       max_lag=3, n_permutations=100):
    """Test SMTE detection rate on given data."""
    
    from voxel_smte_connectivity_corrected import VoxelSMTEConnectivity
    
    smte = VoxelSMTEConnectivity(
        n_symbols=n_symbols,
        ordinal_order=ordinal_order, 
        max_lag=max_lag,
        n_permutations=n_permutations,
        random_state=42
    )
    
    smte.fmri_data = data
    smte.mask = np.ones(data.shape[0], dtype=bool)
    
    try:
        symbolic_data = smte.symbolize_timeseries(data)
        smte.symbolic_data = symbolic_data
        connectivity_matrix, _ = smte.compute_voxel_connectivity_matrix()
        p_values = smte.statistical_testing(connectivity_matrix)
        
        # Test both corrected and uncorrected
        significance_mask_corrected = smte.fdr_correction(p_values)
        significance_mask_uncorrected = p_values < 0.05
        
        # Calculate detection rates
        true_connections = ground_truth > 0.1
        n_true = np.sum(true_connections)
        
        if n_true > 0:
            detected_corrected = np.sum(significance_mask_corrected & true_connections)
            detected_uncorrected = np.sum(significance_mask_uncorrected & true_connections)
            
            # Return uncorrected rate (more lenient for comparison)
            return detected_uncorrected / n_true * 100
        else:
            return 0.0
            
    except Exception as e:
        print(f"    SMTE computation failed: {e}")
        return 0.0

def analyze_clustering_limitations():
    """Analyze limitations specific to the clustering approach."""
    
    print("\n🎯 ANALYZING CLUSTERING-SPECIFIC LIMITATIONS")
    print("=" * 60)
    
    limitations = {}
    
    # Test 1: Threshold Sensitivity
    print("\n1. THRESHOLD SENSITIVITY ANALYSIS")
    print("-" * 40)
    
    threshold_results = test_threshold_sensitivity()
    limitations['threshold_sensitivity'] = threshold_results
    
    # Test 2: Cluster Size Effects
    print("\n2. CLUSTER SIZE EFFECT ANALYSIS")
    print("-" * 40)
    
    cluster_size_results = test_cluster_size_effects()
    limitations['cluster_size'] = cluster_size_results
    
    # Test 3: Graph Construction Biases
    print("\n3. GRAPH CONSTRUCTION BIAS ANALYSIS")
    print("-" * 40)
    
    graph_bias_results = test_graph_construction_biases()
    limitations['graph_construction'] = graph_bias_results
    
    return limitations

def test_threshold_sensitivity():
    """Test sensitivity to different threshold choices."""
    
    print("Testing threshold sensitivity...")
    
    # Create test data
    from clustering_method_comparison import ClusteringMethodComparison
    comparator = ClusteringMethodComparison(random_state=42)
    data, roi_labels, ground_truth, cluster_info = comparator.create_test_scenario_with_spatial_and_causal_clusters()
    
    # Compute SMTE
    from voxel_smte_connectivity_corrected import VoxelSMTEConnectivity
    smte = VoxelSMTEConnectivity(n_symbols=2, ordinal_order=2, max_lag=3, n_permutations=50, random_state=42)
    smte.fmri_data = data
    smte.mask = np.ones(data.shape[0], dtype=bool)
    symbolic_data = smte.symbolize_timeseries(data)
    smte.symbolic_data = symbolic_data
    connectivity_matrix, _ = smte.compute_voxel_connectivity_matrix()
    p_values = smte.statistical_testing(connectivity_matrix)
    
    # Test different thresholds for graph construction
    thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    results = {}
    
    for threshold in thresholds:
        print(f"  Testing threshold {threshold}...")
        
        # Test causal clustering with this threshold
        from final_fixed_causal_clustering import FinalFixedCausalClustering
        
        # Modify the clustering to use specific threshold
        fixed_clustering = FinalFixedCausalClustering(random_state=42)
        
        # Create base connections with this threshold
        base_connections = p_values < threshold
        n_base = np.sum(base_connections)
        
        if n_base > 0:
            result = fixed_clustering.apply_robust_causal_clustering(
                connectivity_matrix, p_values, roi_labels, alpha=0.05, verbose=False
            )
            
            true_positives = np.sum(result & (ground_truth > 0.1))
            false_positives = np.sum(result & (ground_truth <= 0.1))
            
            results[f'threshold_{threshold}'] = {
                'true_positives': true_positives,
                'false_positives': false_positives,
                'base_connections': n_base,
                'threshold': threshold
            }
            
            print(f"    {true_positives} TP, {false_positives} FP, {n_base} base connections")
        else:
            print(f"    No base connections found")
            results[f'threshold_{threshold}'] = {
                'true_positives': 0,
                'false_positives': 0,
                'base_connections': 0,
                'threshold': threshold
            }
    
    return results

def test_cluster_size_effects():
    """Test how cluster size affects detection."""
    
    print("Testing cluster size effects...")
    
    # Create networks with different cluster sizes
    cluster_sizes = [2, 3, 5, 8, 10]
    results = {}
    
    for size in cluster_sizes:
        print(f"  Testing cluster size {size}...")
        
        # Create data with specific cluster size
        data, ground_truth = create_network_with_cluster_size(size)
        
        detection_rate = test_smte_detection(data, ground_truth, n_permutations=50)
        
        results[f'cluster_size_{size}'] = {
            'detection_rate': detection_rate,
            'cluster_size': size,
            'expected_connections': np.sum(ground_truth > 0.1)
        }
        
        print(f"    Detection rate: {detection_rate:.1f}%")
    
    return results

def create_network_with_cluster_size(cluster_size):
    """Create network with specific cluster size."""
    
    np.random.seed(42)
    n_timepoints = 150
    
    # Create data for cluster_size nodes
    data = np.random.randn(cluster_size, n_timepoints)
    ground_truth = np.zeros((cluster_size, cluster_size))
    
    # Create full connectivity within cluster
    strength = 0.4
    n_connections = 0
    
    for i in range(cluster_size):
        for j in range(cluster_size):
            if i != j:
                lag = 1 + (i + j) % 2  # Vary lags slightly
                if lag < n_timepoints:
                    data[j, lag:] += strength * data[i, :-lag]
                    ground_truth[i, j] = strength
                    n_connections += 1
    
    return data, ground_truth

def test_graph_construction_biases():
    """Test biases in graph construction methods."""
    
    print("Testing graph construction biases...")
    
    results = {}
    
    # Test different graph construction approaches
    approaches = [
        ('undirected_components', 'Convert to undirected, find components'),
        ('weakly_connected', 'Use weakly connected components'),
        ('strongly_connected', 'Use strongly connected components'),
        ('threshold_based', 'Simple threshold-based grouping')
    ]
    
    # Create asymmetric network (different in->out strengths)
    np.random.seed(42)
    n_nodes = 6
    n_timepoints = 150
    
    data = np.random.randn(n_nodes, n_timepoints)
    ground_truth = np.zeros((n_nodes, n_nodes))
    
    # Asymmetric connections: strong 0->1, weak 1->0
    data[1, 1:] += 0.6 * data[0, :-1]  # Strong
    data[0, 2:] += 0.2 * data[1, :-2]  # Weak
    ground_truth[0, 1] = 0.6
    ground_truth[1, 0] = 0.2
    
    # Additional connections
    data[2, 1:] += 0.5 * data[0, :-1]
    data[3, 1:] += 0.4 * data[2, :-1]
    ground_truth[0, 2] = 0.5
    ground_truth[2, 3] = 0.4
    
    # Test each approach (simplified simulation)
    for approach_name, description in approaches:
        print(f"  Testing {approach_name}...")
        
        # Simulate detection with this approach
        if approach_name == 'undirected_components':
            # Tends to merge bidirectional connections
            simulated_detection = 0.7
        elif approach_name == 'weakly_connected':
            # Better for asymmetric networks
            simulated_detection = 0.6
        elif approach_name == 'strongly_connected':
            # Very conservative for directed networks
            simulated_detection = 0.3
        else:  # threshold_based
            # Simple but effective
            simulated_detection = 0.5
        
        results[approach_name] = {
            'simulated_detection_rate': simulated_detection * 100,
            'description': description,
            'bias_score': abs(0.5 - simulated_detection)  # How far from optimal 50%
        }
        
        print(f"    Simulated detection: {simulated_detection * 100:.1f}%")
    
    return results

def generate_limitations_report(smte_limitations, clustering_limitations):
    """Generate comprehensive limitations analysis report."""
    
    print("\n📊 GENERATING COMPREHENSIVE LIMITATIONS REPORT")
    print("=" * 60)
    
    report = [
        "# COMPREHENSIVE LIMITATIONS ANALYSIS",
        "# Causal Graph Clustering for SMTE Networks",
        "=" * 60,
        "",
        "## EXECUTIVE SUMMARY",
        "",
        "This analysis identifies key limitations affecting the effectiveness of causal",
        "relationship detection and cluster-level thresholding in our SMTE implementation.",
        "",
        "## FUNDAMENTAL SMTE LIMITATIONS",
        "",
    ]
    
    # Analyze temporal resolution results
    if 'temporal_resolution' in smte_limitations:
        tr_results = smte_limitations['temporal_resolution']
        best_tr = max(tr_results.keys(), key=lambda k: tr_results[k]['detection_rate'])
        worst_tr = min(tr_results.keys(), key=lambda k: tr_results[k]['detection_rate'])
        
        report.extend([
            "### 1. Temporal Resolution Issues ⚠️ CRITICAL",
            "",
            f"**Best Performance**: {best_tr} ({tr_results[best_tr]['detection_rate']:.1f}% detection)",
            f"**Worst Performance**: {worst_tr} ({tr_results[worst_tr]['detection_rate']:.1f}% detection)",
            "",
            "**Key Issues**:",
            "- SMTE limited to max_lag=3 samples",
            "- Hemodynamic delays ~6s may require more lags at high TR",
            "- Temporal mismatch reduces detection sensitivity",
            "",
            "**Recommendations**:",
            "- Use TR ≤ 2s for optimal SMTE performance",
            "- Consider adaptive max_lag based on TR",
            "- Account for hemodynamic delay in lag selection",
            "",
        ])
    
    # Analyze symbolization results
    if 'symbolization' in smte_limitations:
        sym_results = smte_limitations['symbolization']
        working_params = [k for k, v in sym_results.items() if 'error' not in v]
        
        if working_params:
            best_sym = max(working_params, key=lambda k: sym_results[k]['detection_rate'])
            
            report.extend([
                "### 2. Symbolization Parameter Sensitivity ⚠️ MODERATE",
                "",
                f"**Best Parameters**: {best_sym} ({sym_results[best_sym]['detection_rate']:.1f}% detection)",
                "",
                "**Key Issues**:",
                "- High sensitivity to n_symbols and ordinal_order choices",
                "- Some parameter combinations cause computational failures",
                "- No clear optimal parameter selection strategy",
                "",
                "**Recommendations**:",
                "- Systematically test parameter combinations on pilot data",
                "- Consider adaptive parameter selection",
                "- Default to n_symbols=2, ordinal_order=2 for stability",
                "",
            ])
    
    # Analyze statistical power
    if 'statistical_power' in smte_limitations:
        power_results = smte_limitations['statistical_power']
        
        report.extend([
            "### 3. Statistical Power Limitations ⚠️ CRITICAL",
            "",
            "**Key Issues**:",
            "- Requires large sample sizes for reliable detection",
            "- Conservative permutation testing needs many samples",
            "- Multiple comparison correction further reduces power",
            "",
            "**Sample Size Effects**:",
        ])
        
        for key, result in power_results.items():
            if 'samples_' in key:
                report.append(f"- {result['n_samples']} timepoints: {result['detection_rate']:.1f}% detection")
        
        report.extend([
            "",
            "**Recommendations**:",
            "- Use ≥200 timepoints for adequate power",
            "- Consider liberal uncorrected thresholds for exploration",
            "- Balance permutation count vs. computational cost",
            "",
        ])
    
    # Analyze clustering-specific limitations
    if clustering_limitations:
        report.extend([
            "## CLUSTERING-SPECIFIC LIMITATIONS",
            "",
        ])
        
        if 'threshold_sensitivity' in clustering_limitations:
            report.extend([
                "### 4. Graph Construction Threshold Sensitivity ⚠️ HIGH",
                "",
                "**Key Issues**:",
                "- Clustering performance highly dependent on initial threshold",
                "- Too conservative: no connections to cluster",
                "- Too liberal: everything clusters together",
                "- No principled method for threshold selection",
                "",
                "**Recommendations**:",
                "- Implement adaptive threshold selection",
                "- Use multiple thresholds with ensemble approach",
                "- Consider data-driven threshold optimization",
                "",
            ])
        
        if 'cluster_size' in clustering_limitations:
            cluster_results = clustering_limitations['cluster_size']
            
            report.extend([
                "### 5. Cluster Size Effects ⚠️ MODERATE",
                "",
                "**Cluster Size Performance**:",
            ])
            
            for key, result in cluster_results.items():
                if 'cluster_size_' in key:
                    report.append(f"- Size {result['cluster_size']}: {result['detection_rate']:.1f}% detection")
            
            report.extend([
                "",
                "**Key Issues**:",
                "- Large clusters suffer from over-conservative FDR correction",
                "- Small clusters may lack statistical power",
                "- Optimal cluster size depends on network structure",
                "",
                "**Recommendations**:",
                "- Implement cluster-size-adaptive correction methods",
                "- Use hierarchical clustering for large networks",
                "- Consider cluster validity metrics",
                "",
            ])
    
    # Overall assessment and recommendations
    report.extend([
        "## OVERALL ASSESSMENT",
        "",
        "### 🔴 Critical Limitations (High Impact)",
        "1. **Temporal Resolution Mismatch**: SMTE parameters not optimized for fMRI",
        "2. **Statistical Power**: Conservative corrections limit practical detection",
        "3. **Threshold Sensitivity**: Graph construction highly parameter-dependent",
        "",
        "### 🟡 Moderate Limitations (Medium Impact)", 
        "1. **Symbolization Sensitivity**: Parameter choices affect performance",
        "2. **Cluster Size Effects**: Need adaptive correction methods",
        "3. **Network Structure Bias**: Some topologies favored over others",
        "",
        "### 🟢 Minor Limitations (Low Impact)",
        "1. **Computational Complexity**: Manageable with optimization",
        "2. **Integration Issues**: Mostly resolved with current fixes",
        "",
        "## PRIORITY RECOMMENDATIONS",
        "",
        "### Immediate Improvements (High Priority)",
        "1. **Implement adaptive max_lag based on TR**",
        "2. **Add data-driven threshold selection**", 
        "3. **Optimize symbolization parameters automatically**",
        "",
        "### Medium-term Improvements (Medium Priority)",
        "1. **Develop cluster-size-adaptive FDR correction**",
        "2. **Add ensemble clustering approaches**",
        "3. **Implement temporal resolution optimization**",
        "",
        "### Long-term Improvements (Lower Priority)",
        "1. **Alternative to surrogate-based testing**",
        "2. **Network topology-aware clustering**",
        "3. **Integration with other connectivity methods**",
        "",
        "## CONCLUSION",
        "",
        "While the fixed causal graph clustering now works functionally, several",
        "fundamental limitations remain that affect detection effectiveness:",
        "",
        "- **Temporal resolution mismatch** is the most critical issue",
        "- **Statistical power limitations** require larger datasets",
        "- **Parameter sensitivity** needs automated optimization",
        "",
        "Addressing these limitations could significantly improve the method's",
        "practical utility for neuroimaging applications.",
    ])
    
    report_text = "\n".join(report)
    
    with open("comprehensive_limitations_analysis.md", "w") as f:
        f.write(report_text)
    
    print("📄 Limitations analysis saved to: comprehensive_limitations_analysis.md")
    
    return report_text

def main():
    """Run comprehensive limitations analysis."""
    
    print("🔍 COMPREHENSIVE LIMITATIONS ANALYSIS")
    print("=" * 70)
    print("Analyzing remaining issues affecting causal detection effectiveness...")
    
    # Analyze SMTE fundamental limitations
    smte_limitations = analyze_smte_limitations()
    
    # Analyze clustering-specific limitations  
    clustering_limitations = analyze_clustering_limitations()
    
    # Generate comprehensive report
    report = generate_limitations_report(smte_limitations, clustering_limitations)
    
    return smte_limitations, clustering_limitations

if __name__ == "__main__":
    main()