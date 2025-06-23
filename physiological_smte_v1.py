#!/usr/bin/env python3
"""
Phase 1.3: Physiological Constraints for SMTE
This module implements biologically-informed constraints on connectivity analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.preprocessing import StandardScaler
import logging

from network_aware_smte_v1 import NetworkAwareSMTE

logging.basicConfig(level=logging.INFO)


class PhysiologicalConstraints:
    """
    Implements physiological constraints for brain connectivity analysis.
    """
    
    def __init__(self, TR: float = 2.0):
        self.TR = TR  # Repetition time in seconds
        
        # Physiological parameter ranges (in TR units)
        self.constraints = {
            # Hemodynamic response function constraints
            'hemodynamic_delay': {
                'min_lag': 1,    # Minimum 1 TR (~2s)
                'max_lag': 3,    # Maximum 3 TR (~6s)
                'description': 'Hemodynamic response delay'
            },
            
            # Neural transmission delays
            'neural_transmission': {
                'min_lag': 0.5,  # Minimum 0.5 TR (~1s)
                'max_lag': 2.0,  # Maximum 2 TR (~4s)
                'description': 'Neural signal transmission'
            },
            
            # Network-specific constraints
            'visual_processing': {
                'min_lag': 0.5,
                'max_lag': 1.5,
                'description': 'Visual processing hierarchy'
            },
            
            'motor_control': {
                'min_lag': 1.0,
                'max_lag': 2.0, 
                'description': 'Motor command and feedback'
            },
            
            'cognitive_control': {
                'min_lag': 2.0,
                'max_lag': 4.0,
                'description': 'High-level cognitive processing'
            },
            
            'default_mode': {
                'min_lag': 1.0,
                'max_lag': 3.0,
                'description': 'Default mode network dynamics'
            }
        }
        
        # Anatomical distance constraints
        self.distance_constraints = {
            'local_connections': {
                'max_distance': 30.0,  # mm
                'expected_lag': (0.5, 2.0),
                'description': 'Local cortical connections'
            },
            
            'long_range_connections': {
                'min_distance': 30.0,  # mm
                'expected_lag': (1.0, 4.0),
                'description': 'Long-range cortical connections'
            },
            
            'subcortical_connections': {
                'expected_lag': (0.5, 1.5),
                'description': 'Subcortical-cortical connections'
            }
        }
        
        # Connectivity strength constraints
        self.strength_constraints = {
            'within_network': {
                'min_strength': 0.1,
                'description': 'Minimum within-network connectivity'
            },
            
            'between_network': {
                'max_strength': 0.5,
                'description': 'Maximum between-network connectivity'
            },
            
            'hub_connections': {
                'min_strength': 0.05,
                'description': 'Hub connectivity threshold'
            }
        }
    
    def classify_roi_types(self, roi_labels: List[str]) -> Dict[int, str]:
        """
        Classify ROIs into physiological categories.
        """
        
        roi_types = {}
        
        # Define classification keywords
        type_keywords = {
            'visual': ['v1', 'v2', 'v3', 'visual', 'occipital', 'calcarine', 'cuneus'],
            'motor': ['m1', 'motor', 'precentral', 'sma', 'supplementary'],
            'sensory': ['s1', 'sensory', 'postcentral', 'somatosensory'],
            'frontal': ['frontal', 'dlpfc', 'vlpfc', 'ofc', 'acc', 'mfg', 'ifg'],
            'parietal': ['parietal', 'ipl', 'spl', 'angular', 'supramarginal'],
            'temporal': ['temporal', 'stg', 'mtg', 'itg', 'superior', 'middle', 'inferior'],
            'limbic': ['limbic', 'hippocampus', 'amygdala', 'cingulate'],
            'subcortical': ['thalamus', 'caudate', 'putamen', 'pallidum', 'striatum'],
            'default_mode': ['pcc', 'precuneus', 'mpfc', 'angular'],
            'executive': ['dlpfc', 'executive', 'control'],
            'salience': ['insula', 'salience', 'acc']
        }
        
        for i, label in enumerate(roi_labels):
            label_lower = label.lower()
            assigned = False
            
            for roi_type, keywords in type_keywords.items():
                for keyword in keywords:
                    if keyword in label_lower:
                        roi_types[i] = roi_type
                        assigned = True
                        break
                if assigned:
                    break
            
            if not assigned:
                roi_types[i] = 'unclassified'
        
        return roi_types
    
    def apply_lag_constraints(self, 
                            lag_matrix: np.ndarray,
                            roi_types: Dict[int, str],
                            connection_types: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply physiological lag constraints to filter implausible connections.
        """
        
        n_rois = lag_matrix.shape[0]
        plausibility_mask = np.ones_like(lag_matrix, dtype=bool)
        
        for i in range(n_rois):
            for j in range(n_rois):
                if i == j:
                    continue
                
                source_type = roi_types.get(i, 'unclassified')
                target_type = roi_types.get(j, 'unclassified')
                lag = lag_matrix[i, j]
                
                # Determine appropriate constraint based on ROI types
                constraint_key = self._select_constraint_type(source_type, target_type)
                
                if constraint_key in self.constraints:
                    constraint = self.constraints[constraint_key]
                    min_lag = constraint['min_lag']
                    max_lag = constraint['max_lag']
                    
                    # Check if lag is within physiological range
                    if lag < min_lag or lag > max_lag:
                        plausibility_mask[i, j] = False
                
                # Additional specific constraints
                plausibility_mask[i, j] = self._apply_specific_constraints(
                    i, j, source_type, target_type, lag, plausibility_mask[i, j]
                )
        
        return plausibility_mask
    
    def _select_constraint_type(self, source_type: str, target_type: str) -> str:
        """Select appropriate constraint type based on ROI types."""
        
        # Priority-based constraint selection
        if source_type == 'visual' or target_type == 'visual':
            return 'visual_processing'
        elif source_type == 'motor' or target_type == 'motor':
            return 'motor_control'
        elif source_type in ['frontal', 'parietal', 'executive'] or target_type in ['frontal', 'parietal', 'executive']:
            return 'cognitive_control'
        elif source_type == 'default_mode' or target_type == 'default_mode':
            return 'default_mode'
        else:
            return 'neural_transmission'  # Default constraint
    
    def _apply_specific_constraints(self, 
                                  source_idx: int, 
                                  target_idx: int,
                                  source_type: str, 
                                  target_type: str,
                                  lag: float,
                                  current_plausibility: bool) -> bool:
        """Apply specific physiological constraints."""
        
        if not current_plausibility:
            return False
        
        # Motor-sensory feedback loops should have short lags
        if (source_type == 'motor' and target_type == 'sensory') or \
           (source_type == 'sensory' and target_type == 'motor'):
            if lag > 2.0:  # Too long for sensorimotor feedback
                return False
        
        # Visual hierarchy constraints (V1 -> V2 -> higher areas)
        if source_type == 'visual' and target_type == 'visual':
            if lag < 0.5 or lag > 2.0:  # Visual processing steps
                return False
        
        # Subcortical-cortical constraints
        if source_type == 'subcortical' or target_type == 'subcortical':
            if lag > 2.0:  # Subcortical connections should be fast
                return False
        
        # Cognitive control should not be too fast
        if source_type in ['frontal', 'executive'] and target_type in ['frontal', 'executive']:
            if lag < 1.0:  # Cognitive processing takes time
                return False
        
        return True
    
    def apply_distance_constraints(self, 
                                 connectivity_matrix: np.ndarray,
                                 lag_matrix: np.ndarray,
                                 roi_coords: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply anatomical distance constraints.
        """
        
        if roi_coords is None:
            # If no coordinates, return original matrix
            return np.ones_like(connectivity_matrix, dtype=bool)
        
        n_rois = connectivity_matrix.shape[0]
        distance_mask = np.ones_like(connectivity_matrix, dtype=bool)
        
        # Compute distance matrix
        distance_matrix = np.zeros((n_rois, n_rois))
        for i in range(n_rois):
            for j in range(n_rois):
                if i != j:
                    distance = np.linalg.norm(roi_coords[i] - roi_coords[j])
                    distance_matrix[i, j] = distance
        
        # Apply distance-based constraints
        for i in range(n_rois):
            for j in range(n_rois):
                if i == j:
                    continue
                
                distance = distance_matrix[i, j]
                lag = lag_matrix[i, j]
                
                # Local connections constraint
                if distance <= self.distance_constraints['local_connections']['max_distance']:
                    expected_lag = self.distance_constraints['local_connections']['expected_lag']
                    if lag < expected_lag[0] or lag > expected_lag[1]:
                        distance_mask[i, j] = False
                
                # Long-range connections constraint
                elif distance >= self.distance_constraints['long_range_connections']['min_distance']:
                    expected_lag = self.distance_constraints['long_range_connections']['expected_lag']
                    if lag < expected_lag[0] or lag > expected_lag[1]:
                        distance_mask[i, j] = False
        
        return distance_mask
    
    def apply_strength_constraints(self, 
                                 connectivity_matrix: np.ndarray,
                                 roi_types: Dict[int, str],
                                 network_assignments: Optional[Dict[int, str]] = None) -> np.ndarray:
        """
        Apply connectivity strength constraints.
        """
        
        n_rois = connectivity_matrix.shape[0]
        strength_mask = np.ones_like(connectivity_matrix, dtype=bool)
        
        for i in range(n_rois):
            for j in range(n_rois):
                if i == j:
                    continue
                
                strength = connectivity_matrix[i, j]
                
                # Within-network vs between-network constraints
                if network_assignments:
                    network_i = network_assignments.get(i, 'unassigned')
                    network_j = network_assignments.get(j, 'unassigned')
                    
                    if network_i == network_j and network_i != 'unassigned':
                        # Within-network connection
                        min_strength = self.strength_constraints['within_network']['min_strength']
                        if strength < min_strength:
                            strength_mask[i, j] = False
                    else:
                        # Between-network connection
                        max_strength = self.strength_constraints['between_network']['max_strength']
                        if strength > max_strength:
                            # Very strong between-network connections are suspicious
                            pass  # Keep for now, but could be filtered
                
                # Hub connection constraints
                # (This would require identifying hubs first)
                
        return strength_mask


class PhysiologicalSMTE(NetworkAwareSMTE):
    """
    SMTE implementation with physiological constraints.
    """
    
    def __init__(self, 
                 use_physiological_constraints: bool = True,
                 TR: float = 2.0,
                 roi_coords: Optional[np.ndarray] = None,
                 **kwargs):
        
        super().__init__(roi_coords=roi_coords, **kwargs)
        
        self.use_physiological_constraints = use_physiological_constraints
        self.TR = TR
        
        # Initialize physiological constraints
        self.physio_constraints = PhysiologicalConstraints(TR=TR)
        
        # Store constraint results
        self.constraint_info = {}
        
    def apply_physiological_filtering(self, 
                                    connectivity_matrix: np.ndarray,
                                    lag_matrix: np.ndarray,
                                    roi_labels: List[str],
                                    network_assignments: Optional[Dict[int, str]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply physiological constraints to filter connectivity results.
        """
        
        print("Applying physiological constraints...")
        
        # Classify ROI types
        roi_types = self.physio_constraints.classify_roi_types(roi_labels)
        
        # Initialize combined mask
        n_rois = connectivity_matrix.shape[0]
        combined_mask = np.ones((n_rois, n_rois), dtype=bool)
        
        constraint_info = {
            'roi_types': roi_types,
            'constraints_applied': [],
            'filtering_statistics': {}
        }
        
        # 1. Apply lag constraints
        lag_mask = self.physio_constraints.apply_lag_constraints(
            lag_matrix, roi_types, getattr(self, 'connection_types', None)
        )
        combined_mask = combined_mask & lag_mask
        
        n_filtered_lag = np.sum(~lag_mask)
        constraint_info['constraints_applied'].append('lag_constraints')
        constraint_info['filtering_statistics']['lag_filtered'] = n_filtered_lag
        
        print(f"  Lag constraints: filtered {n_filtered_lag} connections")
        
        # 2. Apply distance constraints (if coordinates available)
        if self.roi_coords is not None:
            distance_mask = self.physio_constraints.apply_distance_constraints(
                connectivity_matrix, lag_matrix, self.roi_coords
            )
            combined_mask = combined_mask & distance_mask
            
            n_filtered_distance = np.sum(~distance_mask)
            constraint_info['constraints_applied'].append('distance_constraints')
            constraint_info['filtering_statistics']['distance_filtered'] = n_filtered_distance
            
            print(f"  Distance constraints: filtered {n_filtered_distance} connections")
        
        # 3. Apply strength constraints
        strength_mask = self.physio_constraints.apply_strength_constraints(
            connectivity_matrix, roi_types, network_assignments
        )
        combined_mask = combined_mask & strength_mask
        
        n_filtered_strength = np.sum(~strength_mask)
        constraint_info['constraints_applied'].append('strength_constraints')
        constraint_info['filtering_statistics']['strength_filtered'] = n_filtered_strength
        
        print(f"  Strength constraints: filtered {n_filtered_strength} connections")
        
        # Overall statistics
        total_connections = n_rois * (n_rois - 1)  # Exclude diagonal
        remaining_connections = np.sum(combined_mask) - n_rois  # Exclude diagonal
        filtered_connections = total_connections - remaining_connections
        
        constraint_info['filtering_statistics'].update({
            'total_connections': total_connections,
            'remaining_connections': remaining_connections,
            'filtered_connections': filtered_connections,
            'filtering_proportion': filtered_connections / total_connections
        })
        
        print(f"  Total filtered: {filtered_connections}/{total_connections} "
              f"({filtered_connections/total_connections:.1%})")
        
        return combined_mask, constraint_info
    
    def compute_physiologically_constrained_connectivity(self, 
                                                       data: np.ndarray,
                                                       roi_labels: List[str],
                                                       ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute connectivity with physiological constraints.
        """
        
        print("Computing physiologically-constrained SMTE connectivity...")
        
        # First compute network-aware connectivity
        results = self.compute_network_aware_connectivity(data, roi_labels, ground_truth)
        
        if self.use_physiological_constraints:
            # Apply physiological filtering
            physio_mask, constraint_info = self.apply_physiological_filtering(
                results['connectivity_matrix'],
                results['lag_matrix'],
                roi_labels,
                results['network_structure']['network_assignments']
            )
            
            # Combine with statistical significance mask
            combined_significance = results['significance_mask'] & physio_mask
            
            # Update results
            results.update({
                'physiological_mask': physio_mask,
                'combined_significance_mask': combined_significance,
                'n_physiologically_plausible': np.sum(physio_mask) - len(roi_labels),  # Exclude diagonal
                'n_final_significant': np.sum(combined_significance),
                'constraint_info': constraint_info
            })
            
            print(f"Final significant connections: {results['n_final_significant']} "
                  f"(was {results['n_significant']} before physiological filtering)")
        
        else:
            # No physiological constraints applied
            results.update({
                'physiological_mask': np.ones_like(results['connectivity_matrix'], dtype=bool),
                'combined_significance_mask': results['significance_mask'],
                'n_physiologically_plausible': np.sum(results['significance_mask']),
                'n_final_significant': results['n_significant'],
                'constraint_info': {'constraints_applied': ['none']}
            })
        
        return results
    
    def create_constraint_visualization(self, 
                                      results: Dict[str, Any],
                                      roi_labels: List[str],
                                      save_prefix: str = 'physiological_smte'):
        """Create visualizations showing physiological constraint effects."""
        
        connectivity_matrix = results['connectivity_matrix']
        significance_mask = results['significance_mask']
        physio_mask = results['physiological_mask']
        combined_mask = results['combined_significance_mask']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Raw connectivity
        im1 = axes[0, 0].imshow(connectivity_matrix, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('Raw SMTE Connectivity')
        axes[0, 0].set_xlabel('Source ROI')
        axes[0, 0].set_ylabel('Target ROI')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        # 2. Statistical significance only
        stat_significant = connectivity_matrix * significance_mask
        im2 = axes[0, 1].imshow(stat_significant, cmap='viridis', aspect='auto')
        axes[0, 1].set_title(f'Statistically Significant ({results["n_significant"]})')
        axes[0, 1].set_xlabel('Source ROI')
        axes[0, 1].set_ylabel('Target ROI')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # 3. Physiologically plausible only
        physio_plausible = connectivity_matrix * physio_mask
        im3 = axes[0, 2].imshow(physio_plausible, cmap='viridis', aspect='auto')
        axes[0, 2].set_title(f'Physiologically Plausible ({results["n_physiologically_plausible"]})')
        axes[0, 2].set_xlabel('Source ROI')
        axes[0, 2].set_ylabel('Target ROI')
        plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # 4. Combined (stat + physio)
        final_significant = connectivity_matrix * combined_mask
        im4 = axes[1, 0].imshow(final_significant, cmap='viridis', aspect='auto')
        axes[1, 0].set_title(f'Final Significant ({results["n_final_significant"]})')
        axes[1, 0].set_xlabel('Source ROI')
        axes[1, 0].set_ylabel('Target ROI')
        plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # 5. Filtering mask (what was removed)
        filtering_mask = significance_mask & ~combined_mask
        im5 = axes[1, 1].imshow(filtering_mask.astype(int), cmap='Reds', aspect='auto')
        axes[1, 1].set_title(f'Filtered by Physiology ({np.sum(filtering_mask)})')
        axes[1, 1].set_xlabel('Source ROI')
        axes[1, 1].set_ylabel('Target ROI')
        plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # 6. ROI type classification
        if 'constraint_info' in results and 'roi_types' in results['constraint_info']:
            roi_types = results['constraint_info']['roi_types']
            type_to_num = {rtype: i for i, rtype in enumerate(set(roi_types.values()))}
            type_matrix = np.array([[type_to_num[roi_types[i]] for i in range(len(roi_labels))] 
                                  for _ in range(len(roi_labels))])
            
            im6 = axes[1, 2].imshow(type_matrix, cmap='tab10', aspect='auto')
            axes[1, 2].set_title('ROI Type Classification')
            axes[1, 2].set_xlabel('ROI Index')
            axes[1, 2].set_ylabel('ROI Index')
            
            # Add colorbar with type names
            cbar = plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)
            cbar.set_ticks(list(type_to_num.values()))
            cbar.set_ticklabels(list(type_to_num.keys()))
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_physiological_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


def test_physiological_smte():
    """Test the physiological SMTE implementation."""
    
    print("Testing Physiological SMTE Implementation")
    print("=" * 60)
    
    # Generate realistic test data
    np.random.seed(42)
    n_regions = 12
    n_timepoints = 120
    TR = 2.0
    
    # Create ROI labels with physiological meaning
    roi_labels = [
        'V1_L', 'V1_R',           # Visual cortex
        'M1_L', 'M1_R',           # Motor cortex
        'S1_L', 'S1_R',           # Sensory cortex
        'DLPFC_L', 'DLPFC_R',     # Executive control
        'PCC', 'mPFC',            # Default mode
        'Insula_L', 'Temporal_L'   # Salience/other
    ]
    
    # Create synthetic coordinates (in mm)
    roi_coords = np.array([
        [-20, -90, 10], [20, -90, 10],     # V1
        [-40, -20, 50], [40, -20, 50],     # M1
        [-40, -30, 45], [40, -30, 45],     # S1
        [-45, 25, 35], [45, 25, 35],       # DLPFC
        [0, -50, 25], [0, 50, 15],         # PCC, mPFC
        [-35, 15, 5], [-50, -10, -10]      # Insula, Temporal
    ])
    
    # Define known networks
    known_networks = {
        'visual': [0, 1],
        'sensorimotor': [2, 3, 4, 5],
        'executive': [6, 7],
        'default': [8, 9]
    }
    
    # Generate realistic fMRI data
    data = []
    for i in range(n_regions):
        base_signal = np.sin(2 * np.pi * 0.02 * np.arange(n_timepoints))
        noise = 0.4 * np.random.randn(n_timepoints)
        signal = base_signal + noise
        data.append(signal)
    
    data = np.array(data)
    
    # Add physiologically plausible connections
    # V1 -> higher visual areas (short lag)
    # Motor -> sensory feedback (short lag)
    # Executive control (longer lag)
    
    # Visual processing: V1_L -> V1_R (short lag)
    data[1, 1:] += 0.4 * data[0, :-1]
    
    # Sensorimotor: M1_L -> S1_L (physiological lag)
    data[4, 2:] += 0.5 * data[2, :-2]
    
    # Executive: DLPFC_L -> DLPFC_R (cognitive lag)
    data[7, 3:] += 0.3 * data[6, :-3]
    
    # Default mode: PCC -> mPFC (intermediate lag)
    data[9, 2:] += 0.4 * data[8, :-2]
    
    # Add implausible connection (should be filtered)
    # Very long lag visual connection
    data[1, 8:] += 0.2 * data[0, :-8]  # 8 TR lag is too long for visual
    
    # Standardize data
    scaler = StandardScaler()
    data = scaler.fit_transform(data.T).T
    
    # Test without physiological constraints
    print("\n1. Testing without physiological constraints")
    print("-" * 50)
    
    physio_smte_off = PhysiologicalSMTE(
        adaptive_mode='heuristic',
        use_network_correction=True,
        use_physiological_constraints=False,  # Disable constraints
        known_networks=known_networks,
        roi_coords=roi_coords,
        TR=TR,
        n_permutations=100,
        random_state=42
    )
    
    results_off = physio_smte_off.compute_physiologically_constrained_connectivity(
        data, roi_labels
    )
    
    print(f"Without constraints: {results_off['n_significant']} significant connections")
    
    # Test with physiological constraints
    print("\n2. Testing with physiological constraints")
    print("-" * 50)
    
    physio_smte_on = PhysiologicalSMTE(
        adaptive_mode='heuristic',
        use_network_correction=True,
        use_physiological_constraints=True,   # Enable constraints
        known_networks=known_networks,
        roi_coords=roi_coords,
        TR=TR,
        n_permutations=100,
        random_state=42
    )
    
    results_on = physio_smte_on.compute_physiologically_constrained_connectivity(
        data, roi_labels
    )
    
    print(f"With constraints: {results_on['n_final_significant']} final significant connections")
    
    # Show constraint effects
    if 'constraint_info' in results_on:
        print(f"\nConstraint filtering effects:")
        stats = results_on['constraint_info']['filtering_statistics']
        for constraint, n_filtered in stats.items():
            if constraint.endswith('_filtered'):
                print(f"  {constraint}: {n_filtered} connections filtered")
    
    # Create visualization
    physio_smte_on.create_constraint_visualization(results_on, roi_labels)
    
    return results_on


if __name__ == "__main__":
    results = test_physiological_smte()