#!/usr/bin/env python3
"""
Phase 1.2: Network-Aware Statistical Correction for SMTE
This module implements adaptive FDR correction based on network structure and connection types.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
import networkx as nx
from scipy import stats
from sklearn.preprocessing import StandardScaler
import logging

from adaptive_smte_v1 import AdaptiveSMTE

logging.basicConfig(level=logging.INFO)


class NetworkStructureAnalyzer:
    """
    Analyzes network structure to guide statistical correction.
    """
    
    def __init__(self):
        self.network_types = {
            'within_network': 'connections within known brain networks',
            'between_network': 'connections between different networks', 
            'hub_connections': 'connections involving network hubs',
            'peripheral_connections': 'connections involving peripheral nodes',
            'short_range': 'anatomically local connections',
            'long_range': 'anatomically distant connections'
        }
        
    def analyze_roi_structure(self, 
                            roi_labels: List[str],
                            roi_coords: Optional[np.ndarray] = None,
                            known_networks: Optional[Dict[str, List[int]]] = None) -> Dict[str, Any]:
        """
        Analyze ROI structure and create network organization.
        """
        
        n_rois = len(roi_labels)
        structure_info = {
            'n_rois': n_rois,
            'roi_labels': roi_labels,
            'network_assignments': {},
            'connection_types': np.zeros((n_rois, n_rois), dtype=int),
            'distance_matrix': None
        }
        
        # Assign ROIs to networks if known networks provided
        if known_networks:
            structure_info['network_assignments'] = self._assign_rois_to_networks(
                roi_labels, known_networks
            )
        else:
            # Create default network assignments based on naming conventions
            structure_info['network_assignments'] = self._infer_network_assignments(roi_labels)
        
        # Compute spatial distances if coordinates provided
        if roi_coords is not None:
            structure_info['distance_matrix'] = self._compute_spatial_distances(roi_coords)
        
        # Classify connection types
        structure_info['connection_types'] = self._classify_connection_types(
            structure_info['network_assignments'],
            structure_info['distance_matrix']
        )
        
        return structure_info
    
    def _assign_rois_to_networks(self, roi_labels: List[str], known_networks: Dict[str, List[int]]) -> Dict[int, str]:
        """Assign ROIs to known networks."""
        
        assignments = {}
        
        for network_name, roi_indices in known_networks.items():
            for roi_idx in roi_indices:
                if roi_idx < len(roi_labels):
                    assignments[roi_idx] = network_name
        
        # Assign unassigned ROIs to 'unassigned' network
        for i in range(len(roi_labels)):
            if i not in assignments:
                assignments[i] = 'unassigned'
        
        return assignments
    
    def _infer_network_assignments(self, roi_labels: List[str]) -> Dict[int, str]:
        """Infer network assignments from ROI labels."""
        
        # Common network keywords
        network_keywords = {
            'default': ['pcc', 'mPFC', 'angular', 'precuneus', 'default'],
            'executive': ['dlpfc', 'parietal', 'frontal', 'executive', 'control'],
            'salience': ['insula', 'acc', 'salience'],
            'sensorimotor': ['motor', 'sensory', 'm1', 's1', 'precentral', 'postcentral'],
            'visual': ['visual', 'occipital', 'v1', 'v2', 'calcarine'],
            'auditory': ['auditory', 'temporal', 'heschl', 'stg']
        }
        
        assignments = {}
        
        for i, label in enumerate(roi_labels):
            label_lower = label.lower()
            assigned = False
            
            for network, keywords in network_keywords.items():
                for keyword in keywords:
                    if keyword in label_lower:
                        assignments[i] = network
                        assigned = True
                        break
                if assigned:
                    break
            
            if not assigned:
                assignments[i] = 'unassigned'
        
        return assignments
    
    def _compute_spatial_distances(self, roi_coords: np.ndarray) -> np.ndarray:
        """Compute spatial distances between ROIs."""
        
        n_rois = roi_coords.shape[0]
        distance_matrix = np.zeros((n_rois, n_rois))
        
        for i in range(n_rois):
            for j in range(n_rois):
                if i != j:
                    distance = np.linalg.norm(roi_coords[i] - roi_coords[j])
                    distance_matrix[i, j] = distance
        
        return distance_matrix
    
    def _classify_connection_types(self, 
                                 network_assignments: Dict[int, str],
                                 distance_matrix: Optional[np.ndarray]) -> np.ndarray:
        """
        Classify connection types for statistical correction.
        
        Connection type codes:
        0: within-network, short-range
        1: within-network, long-range  
        2: between-network, short-range
        3: between-network, long-range
        4: hub connections
        5: peripheral connections
        """
        
        n_rois = len(network_assignments)
        connection_types = np.zeros((n_rois, n_rois), dtype=int)
        
        # Determine distance threshold if available
        distance_threshold = None
        if distance_matrix is not None:
            # Use median distance as threshold
            distance_threshold = np.median(distance_matrix[distance_matrix > 0])
        
        # Identify network hubs (nodes with many within-network connections)
        network_sizes = {}
        for roi_idx, network in network_assignments.items():
            if network not in network_sizes:
                network_sizes[network] = 0
            network_sizes[network] += 1
        
        hub_threshold = 0.3  # ROIs connecting to >30% of their network
        hub_rois = set()
        
        for i in range(n_rois):
            network_i = network_assignments[i]
            if network_i != 'unassigned':
                same_network_count = sum(1 for j in range(n_rois) 
                                       if j != i and network_assignments[j] == network_i)
                if same_network_count > hub_threshold * network_sizes[network_i]:
                    hub_rois.add(i)
        
        # Classify connections
        for i in range(n_rois):
            for j in range(n_rois):
                if i == j:
                    continue
                
                network_i = network_assignments[i]
                network_j = network_assignments[j]
                
                # Check if hub connection
                if i in hub_rois or j in hub_rois:
                    connection_types[i, j] = 4  # Hub connection
                    continue
                
                # Check within vs between network
                within_network = (network_i == network_j and network_i != 'unassigned')
                
                # Check distance if available
                if distance_matrix is not None and distance_threshold is not None:
                    is_short_range = distance_matrix[i, j] <= distance_threshold
                else:
                    is_short_range = True  # Default to short-range if no distance info
                
                # Assign connection type
                if within_network:
                    connection_types[i, j] = 0 if is_short_range else 1
                else:
                    connection_types[i, j] = 2 if is_short_range else 3
                    
                # Check for peripheral connections (few connections)
                if i not in hub_rois and j not in hub_rois:
                    # Count connections for each ROI
                    connections_i = np.sum(connection_types[i, :] > 0) + np.sum(connection_types[:, i] > 0)
                    connections_j = np.sum(connection_types[j, :] > 0) + np.sum(connection_types[:, j] > 0)
                    
                    if connections_i < 3 or connections_j < 3:  # Very few connections
                        connection_types[i, j] = 5  # Peripheral connection
        
        return connection_types


class NetworkAwareFDRCorrection:
    """
    Network-aware False Discovery Rate correction.
    """
    
    def __init__(self):
        # Different alpha levels for different connection types
        self.alpha_levels = {
            0: 0.05,   # within-network, short-range (standard)
            1: 0.03,   # within-network, long-range (slightly stricter)
            2: 0.01,   # between-network, short-range (stricter)
            3: 0.005,  # between-network, long-range (very strict)
            4: 0.10,   # hub connections (more lenient)
            5: 0.001   # peripheral connections (very strict)
        }
        
        # Connection type names for reporting
        self.connection_names = {
            0: 'within-network, short-range',
            1: 'within-network, long-range',
            2: 'between-network, short-range', 
            3: 'between-network, long-range',
            4: 'hub connections',
            5: 'peripheral connections'
        }
    
    def network_aware_fdr_correction(self, 
                                   p_values: np.ndarray,
                                   connection_types: np.ndarray,
                                   global_alpha: float = 0.05) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply network-aware FDR correction.
        """
        
        corrected_significance = np.zeros_like(p_values, dtype=bool)
        correction_info = {
            'global_alpha': global_alpha,
            'type_statistics': {},
            'total_tests': np.sum(~np.eye(p_values.shape[0], dtype=bool)),
            'total_significant': 0
        }
        
        # Apply correction for each connection type
        unique_types = np.unique(connection_types)
        
        for conn_type in unique_types:
            if conn_type == -1:  # Skip invalid types
                continue
                
            # Get connections of this type
            type_mask = (connection_types == conn_type)
            type_p_values = p_values[type_mask]
            
            if len(type_p_values) == 0:
                continue
            
            # Get alpha level for this connection type
            type_alpha = self.alpha_levels.get(conn_type, global_alpha)
            
            # Apply Benjamini-Hochberg FDR correction
            type_significant = self._benjamini_hochberg_correction(type_p_values, type_alpha)
            
            # Store results
            corrected_significance[type_mask] = type_significant
            
            # Store statistics
            correction_info['type_statistics'][conn_type] = {
                'connection_type': self.connection_names.get(conn_type, f'type_{conn_type}'),
                'n_tests': len(type_p_values),
                'n_significant': np.sum(type_significant),
                'alpha_used': type_alpha,
                'proportion_significant': np.sum(type_significant) / len(type_p_values),
                'min_p_value': np.min(type_p_values),
                'median_p_value': np.median(type_p_values)
            }
        
        # Overall statistics
        correction_info['total_significant'] = np.sum(corrected_significance)
        correction_info['overall_proportion'] = (
            correction_info['total_significant'] / correction_info['total_tests']
        )
        
        return corrected_significance, correction_info
    
    def _benjamini_hochberg_correction(self, p_values: np.ndarray, alpha: float) -> np.ndarray:
        """Apply Benjamini-Hochberg FDR correction."""
        
        if len(p_values) == 0:
            return np.array([], dtype=bool)
        
        # Sort p-values
        n_tests = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        # Benjamini-Hochberg critical values
        critical_values = (np.arange(1, n_tests + 1) / n_tests) * alpha
        
        # Find largest k such that P(k) <= (k/n) * alpha
        significant_indices = np.where(sorted_p_values <= critical_values)[0]
        
        if len(significant_indices) == 0:
            return np.zeros(n_tests, dtype=bool)
        
        # All tests up to the largest significant index are significant
        max_significant_index = np.max(significant_indices)
        significant_mask = np.zeros(n_tests, dtype=bool)
        significant_mask[sorted_indices[:max_significant_index + 1]] = True
        
        return significant_mask


class NetworkAwareSMTE(AdaptiveSMTE):
    """
    SMTE implementation with network-aware statistical correction.
    """
    
    def __init__(self, 
                 use_network_correction: bool = True,
                 roi_coords: Optional[np.ndarray] = None,
                 known_networks: Optional[Dict[str, List[int]]] = None,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.use_network_correction = use_network_correction
        self.roi_coords = roi_coords
        self.known_networks = known_networks
        
        # Initialize network analysis components
        self.network_analyzer = NetworkStructureAnalyzer()
        self.fdr_corrector = NetworkAwareFDRCorrection()
        
        # Store network structure info
        self.network_structure = None
        self.correction_info = None
        
    def analyze_network_structure(self, roi_labels: List[str]) -> Dict[str, Any]:
        """Analyze the network structure of ROIs."""
        
        self.network_structure = self.network_analyzer.analyze_roi_structure(
            roi_labels, self.roi_coords, self.known_networks
        )
        
        return self.network_structure
    
    def network_aware_statistical_testing(self, 
                                        connectivity_matrix: np.ndarray,
                                        roi_labels: List[str]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Perform statistical testing with network-aware correction.
        """
        
        # Analyze network structure if not done already
        if self.network_structure is None:
            self.analyze_network_structure(roi_labels)
        
        # Perform standard statistical testing to get p-values
        print("Computing p-values using standard permutation testing...")
        p_values = self.statistical_testing(connectivity_matrix)
        
        if self.use_network_correction:
            print("Applying network-aware FDR correction...")
            
            # Apply network-aware correction
            significance_mask, correction_info = self.fdr_corrector.network_aware_fdr_correction(
                p_values, self.network_structure['connection_types']
            )
            
            self.correction_info = correction_info
            
            print(f"Network-aware correction results:")
            print(f"  Total tests: {correction_info['total_tests']}")
            print(f"  Total significant: {correction_info['total_significant']}")
            print(f"  Overall proportion: {correction_info['overall_proportion']:.3f}")
            
            # Print per-connection-type results
            for conn_type, stats in correction_info['type_statistics'].items():
                print(f"  {stats['connection_type']}: "
                      f"{stats['n_significant']}/{stats['n_tests']} "
                      f"({stats['proportion_significant']:.3f}) "
                      f"at α={stats['alpha_used']}")
        
        else:
            print("Applying standard FDR correction...")
            significance_mask = self.fdr_correction(p_values)
            self.correction_info = {
                'method': 'standard_fdr',
                'total_significant': np.sum(significance_mask)
            }
        
        return p_values, significance_mask, self.correction_info
    
    def compute_network_aware_connectivity(self, 
                                         data: np.ndarray,
                                         roi_labels: List[str],
                                         ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute connectivity with network-aware statistical correction.
        """
        
        print("Computing network-aware SMTE connectivity...")
        
        # Adaptive parameter fitting
        if hasattr(self, 'fit_parameters'):
            optimization_result = self.fit_parameters(data, ground_truth, verbose=False)
        else:
            optimization_result = {}
        
        # Symbolize and compute connectivity
        symbolic_data = self.symbolize_timeseries(data)
        self.symbolic_data = symbolic_data
        connectivity_matrix, lag_matrix = self.compute_voxel_connectivity_matrix()
        
        # Network-aware statistical testing
        p_values, significance_mask, correction_info = self.network_aware_statistical_testing(
            connectivity_matrix, roi_labels
        )
        
        results = {
            'connectivity_matrix': connectivity_matrix,
            'lag_matrix': lag_matrix,
            'p_values': p_values,
            'significance_mask': significance_mask,
            'n_significant': np.sum(significance_mask),
            'network_structure': self.network_structure,
            'correction_info': correction_info,
            'optimization_result': optimization_result
        }
        
        return results
    
    def create_network_visualization(self, 
                                   results: Dict[str, Any],
                                   roi_labels: List[str],
                                   save_prefix: str = 'network_aware_smte'):
        """Create visualizations showing network-aware results."""
        
        connectivity_matrix = results['connectivity_matrix']
        significance_mask = results['significance_mask']
        connection_types = results['network_structure']['connection_types']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Raw connectivity matrix
        im1 = axes[0, 0].imshow(connectivity_matrix, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('Raw SMTE Connectivity')
        axes[0, 0].set_xlabel('Source ROI')
        axes[0, 0].set_ylabel('Target ROI') 
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        # 2. Significant connections only
        significant_connectivity = connectivity_matrix * significance_mask
        im2 = axes[0, 1].imshow(significant_connectivity, cmap='viridis', aspect='auto')
        axes[0, 1].set_title(f'Significant Connections ({results["n_significant"]})')
        axes[0, 1].set_xlabel('Source ROI')
        axes[0, 1].set_ylabel('Target ROI')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # 3. Connection types
        im3 = axes[1, 0].imshow(connection_types, cmap='tab10', aspect='auto')
        axes[1, 0].set_title('Connection Types')
        axes[1, 0].set_xlabel('Source ROI')
        axes[1, 0].set_ylabel('Target ROI')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # 4. P-values
        im4 = axes[1, 1].imshow(-np.log10(results['p_values'] + 1e-10), cmap='hot', aspect='auto')
        axes[1, 1].set_title('Statistical Significance (-log10 p)')
        axes[1, 1].set_xlabel('Source ROI')
        axes[1, 1].set_ylabel('Target ROI')
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_network_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


def test_network_aware_smte():
    """Test the network-aware SMTE implementation."""
    
    print("Testing Network-Aware SMTE Implementation")
    print("=" * 60)
    
    # Generate realistic test data with network structure
    np.random.seed(42)
    n_regions = 15
    n_timepoints = 120
    
    # Create ROI labels with network structure
    roi_labels = [
        # Default Mode Network
        'PCC', 'mPFC', 'Angular_L', 'Angular_R', 'Precuneus',
        # Executive Control Network  
        'DLPFC_L', 'DLPFC_R', 'Parietal_L', 'Parietal_R',
        # Sensorimotor Network
        'M1_L', 'M1_R', 'S1_L', 'S1_R',
        # Other regions
        'Insula', 'Temporal'
    ]
    
    # Define known networks
    known_networks = {
        'default': [0, 1, 2, 3, 4],           # PCC, mPFC, Angular, Precuneus
        'executive': [5, 6, 7, 8],             # DLPFC, Parietal
        'sensorimotor': [9, 10, 11, 12]        # Motor, Sensory
    }
    
    # Generate synthetic coordinates
    roi_coords = np.random.randn(n_regions, 3) * 10
    
    # Create realistic fMRI-like data
    data = []
    for i in range(n_regions):
        # Base signal
        base_signal = np.sin(2 * np.pi * 0.02 * np.arange(n_timepoints))
        noise = 0.5 * np.random.randn(n_timepoints)
        signal = base_signal + noise
        data.append(signal)
    
    data = np.array(data)
    
    # Add within-network connectivity
    # Default Mode Network connections
    data[1, 2:] += 0.4 * data[0, :-2]  # PCC -> mPFC
    data[2, 1:] += 0.3 * data[0, :-1]  # PCC -> Angular_L
    
    # Executive Network connections  
    data[6, 1:] += 0.5 * data[5, :-1]  # DLPFC_L -> DLPFC_R
    data[7, 2:] += 0.4 * data[5, :-2]  # DLPFC_L -> Parietal_L
    
    # Sensorimotor connections
    data[10, 1:] += 0.6 * data[9, :-1]  # M1_L -> M1_R
    data[11, 1:] += 0.3 * data[9, :-1]  # M1_L -> S1_L
    
    # Add some between-network connections (should be less significant)
    data[5, 3:] += 0.2 * data[0, :-3]  # PCC -> DLPFC_L (weak)
    
    # Standardize data
    scaler = StandardScaler()
    data = scaler.fit_transform(data.T).T
    
    # Test network-aware SMTE
    network_smte = NetworkAwareSMTE(
        adaptive_mode='heuristic',
        use_network_correction=True,
        roi_coords=roi_coords,
        known_networks=known_networks,
        n_permutations=100,  # Reduced for testing
        random_state=42
    )
    
    # Compute connectivity
    results = network_smte.compute_network_aware_connectivity(data, roi_labels)
    
    # Display results
    print(f"\nResults Summary:")
    print(f"Total significant connections: {results['n_significant']}")
    print(f"Network structure analyzed: {len(results['network_structure']['network_assignments'])} ROIs")
    
    if 'type_statistics' in results['correction_info']:
        print(f"\nConnection type breakdown:")
        for conn_type, stats in results['correction_info']['type_statistics'].items():
            print(f"  {stats['connection_type']}: "
                  f"{stats['n_significant']}/{stats['n_tests']} significant "
                  f"(α={stats['alpha_used']})")
    
    # Create visualization
    network_smte.create_network_visualization(results, roi_labels)
    
    return results


if __name__ == "__main__":
    results = test_network_aware_smte()