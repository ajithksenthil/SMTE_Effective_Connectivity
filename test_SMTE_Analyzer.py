import numpy as np
import nibabel as nib
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import KBinsDiscretizer
import warnings
from typing import Tuple, List, Optional
import os

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

class SMTEAnalyzer:
    """
    Symbolic Matrix Transfer Entropy (SMTE) analyzer for fMRI data.
    
    This class implements SMTE analysis for fMRI time series data, including
    synthetic data generation for validation and visualization tools.
    """
    
    def __init__(self, num_symbols: int = 5):
        """
        Initialize the SMTE analyzer.
        
        Args:
            num_symbols (int): Number of symbols for discretization
        """
        self.num_symbols = num_symbols
        
    def load_fmri_data(self, 
                       nifti_file_path: str, 
                       mask_file_path: Optional[str] = None) -> np.ndarray:
        """
        Load fMRI data from a NIfTI file and optionally apply a mask.
        
        Args:
            nifti_file_path (str): Path to the fMRI NIfTI file
            mask_file_path (str, optional): Path to the mask NIfTI file
            
        Returns:
            np.ndarray: Time series data with shape (num_voxels, num_timepoints)
        """
        # Load fMRI data
        fmri_img = nib.load(nifti_file_path)
        fmri_data = fmri_img.get_fdata()
        
        # Apply mask if provided
        if mask_file_path and os.path.exists(mask_file_path):
            mask_img = nib.load(mask_file_path)
            mask_data = mask_img.get_fdata().astype(bool)
            time_series_data = fmri_data[mask_data]
        else:
            # Reshape to 2D array (voxels × time points)
            time_series_data = fmri_data.reshape(-1, fmri_data.shape[-1])
            
        # Standardize the time series
        time_series_data = self._standardize_time_series(time_series_data)
        
        return time_series_data
    
    def _standardize_time_series(self, data: np.ndarray) -> np.ndarray:
        """
        Standardize time series to zero mean and unit variance.
        
        Args:
            data (np.ndarray): Input time series data
            
        Returns:
            np.ndarray: Standardized time series data
        """
        # Calculate mean and std along time axis
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        
        # Handle zero standard deviation
        std[std < 1e-10] = 1.0
        
        # Standardize
        return (data - mean) / std
    
    def symbolize_time_series(self, time_series: np.ndarray) -> np.ndarray:
        """
        Convert continuous time series into symbolic sequences using equal-frequency binning.
        
        Args:
            time_series (np.ndarray): Input time series data
            
        Returns:
            np.ndarray: Symbolic sequences
        """
        est = KBinsDiscretizer(
            n_bins=self.num_symbols, 
            encode='ordinal', 
            strategy='uniform'
        )
        symbols = est.fit_transform(
            time_series.reshape(-1, 1)
        ).astype(int).flatten()
        return symbols
    
    def compute_gram_matrix(self, symbols: np.ndarray) -> np.ndarray:
        """
        Compute the Gram matrix for a symbolic sequence.
        
        Args:
            symbols (np.ndarray): Symbolic sequence
            
        Returns:
            np.ndarray: Gram matrix
        """
        N = len(symbols)
        gram_matrix = np.equal.outer(symbols, symbols).astype(float)
        return gram_matrix / N  # Normalize
    
    def compute_joint_gram_matrix(self, 
                                symbols_x: np.ndarray, 
                                symbols_y: np.ndarray) -> np.ndarray:
        """
        Compute the joint Gram matrix for two symbolic sequences.
        
        Args:
            symbols_x (np.ndarray): First symbolic sequence
            symbols_y (np.ndarray): Second symbolic sequence
            
        Returns:
            np.ndarray: Joint Gram matrix
        """
        N = len(symbols_x)
        joint_matrix = (np.equal.outer(symbols_x, symbols_x) * 
                       np.equal.outer(symbols_y, symbols_y))
        return joint_matrix.astype(float) / N
    
    def matrix_entropy(self, gram_matrix: np.ndarray) -> float:
        """
        Compute the second-order matrix entropy.
        
        Args:
            gram_matrix (np.ndarray): Input Gram matrix
            
        Returns:
            float: Matrix entropy value
        """
        trace = np.trace(np.dot(gram_matrix, gram_matrix))
        trace = max(trace, 1e-10)  # Avoid log(0)
        return -np.log(trace)
    
    def compute_smte(self, 
                    symbols_x: np.ndarray, 
                    symbols_y: np.ndarray) -> float:
        """
        Compute SMTE from time series Y to X.
        
        Args:
            symbols_x (np.ndarray): Target symbolic sequence
            symbols_y (np.ndarray): Source symbolic sequence
            
        Returns:
            float: SMTE value from Y to X
        """
        # Compute Gram matrices
        gram_x = self.compute_gram_matrix(symbols_x)
        gram_y = self.compute_gram_matrix(symbols_y)
        gram_joint = self.compute_joint_gram_matrix(symbols_x, symbols_y)
        
        # Compute entropies
        S_x = self.matrix_entropy(gram_x)
        S_y = self.matrix_entropy(gram_y)
        S_joint = self.matrix_entropy(gram_joint)
        
        # Compute conditional entropy
        S_x_given_y = S_joint - S_y
        
        # Compute SMTE
        smte_y_to_x = S_x_given_y - S_x
        return smte_y_to_x
    
    def compute_smte_matrix(self, time_series_data: np.ndarray) -> np.ndarray:
        """
        Compute pairwise SMTE matrix for all time series.
        
        Args:
            time_series_data (np.ndarray): Input time series data
            
        Returns:
            np.ndarray: SMTE matrix
        """
        num_series = time_series_data.shape[0]
        smte_matrix = np.zeros((num_series, num_series))
        
        # Convert all time series to symbols first
        symbolic_series = np.array([
            self.symbolize_time_series(ts) for ts in time_series_data
        ])
        
        # Compute SMTE for all pairs
        for i in range(num_series):
            for j in range(num_series):
                if i != j:
                    smte_matrix[i, j] = self.compute_smte(
                        symbolic_series[i], 
                        symbolic_series[j]
                    )
                    
        return smte_matrix
    

    
    def build_directed_graph(self, 
                           smte_matrix: np.ndarray, 
                           threshold_percentile: float = 95) -> nx.DiGraph:
        """
        Build directed graph from SMTE matrix.
        
        Args:
            smte_matrix (np.ndarray): SMTE matrix
            threshold_percentile (float): Percentile threshold for edges
            
        Returns:
            nx.DiGraph: Directed graph
        """
        # Threshold SMTE values
        threshold = np.percentile(smte_matrix, threshold_percentile)
        adjacency_matrix = (smte_matrix >= threshold).astype(int)
        
        # Create directed graph
        G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
        
        # Add SMTE values as edge weights
        for i, j in G.edges():
            G[i][j]['weight'] = smte_matrix[i, j]
            
        return G
    
    def plot_smte_matrix(self, 
                        smte_matrix: np.ndarray, 
                        title: str = "SMTE Matrix"):
        """
        Plot the SMTE matrix.
        
        Args:
            smte_matrix (np.ndarray): SMTE matrix to plot
            title (str): Plot title
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(smte_matrix, cmap='hot', interpolation='nearest')
        plt.colorbar(label='SMTE Value')
        plt.title(title)
        plt.xlabel('Source Time Series')
        plt.ylabel('Target Time Series')
        plt.show()
        
    def plot_directed_graph(self, G: nx.DiGraph):
        """
        Plot the directed graph.
        
        Args:
            G (nx.DiGraph): NetworkX directed graph
        """
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
        
        # Draw edges with weights determining color
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=edges,
            edge_color=weights,
            edge_cmap=plt.cm.viridis,
            width=2,
            arrowsize=20
        )
        
        plt.title('SMTE Directed Graph Network')
        plt.axis('off')
        plt.show()

    def generate_synthetic_data(self, 
                                num_timepoints: int, 
                                num_series: int,
                                dependency_config: Optional[List[dict]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic time series data with known dependencies.
        
        Args:
            num_timepoints (int): Number of time points
            num_series (int): Number of time series
            dependency_config (List[dict], optional): List of dependency configurations.
                Each dict should contain:
                - 'source': source time series index
                - 'target': target time series index
                - 'strength': dependency strength
                - 'lag': time lag (default=1)
                If None, creates default sequential dependencies
                
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - Synthetic time series data
                - Ground truth dependency matrix
        """
        np.random.seed(42)  # For reproducibility
        
        # Initialize data with random noise
        data = np.random.randn(num_series, num_timepoints)
        
        # Initialize ground truth matrix
        ground_truth = np.zeros((num_series, num_series))
        
        if dependency_config is None:
            # Default configuration: series i depends on series i-1
            dependency_config = [
                {
                    'source': i-1,
                    'target': i,
                    'strength': 0.5,
                    'lag': 1
                }
                for i in range(1, num_series)
            ]
        
        # Apply dependencies
        for dep in dependency_config:
            source = dep['source']
            target = dep['target']
            strength = dep['strength']
            lag = dep.get('lag', 1)
            
            if source >= 0 and target < num_series:
                # Add lagged dependency
                data[target, lag:] += strength * data[source, :-lag]
                
                # Record in ground truth matrix
                ground_truth[target, source] = strength
        
        # Standardize the time series
        data = self._standardize_time_series(data)
        
        return data, ground_truth

    def evaluate_smte_performance(self, 
                                smte_matrix: np.ndarray,
                                ground_truth: np.ndarray,
                                threshold_percentile: float = 95) -> dict:
        """
        Evaluate how well SMTE captures known dependencies in synthetic data.
        
        Args:
            smte_matrix (np.ndarray): Computed SMTE matrix
            ground_truth (np.ndarray): Ground truth dependency matrix
            threshold_percentile (float): Percentile for thresholding SMTE values
            
        Returns:
            dict: Dictionary containing various evaluation metrics
        """
        num_series = smte_matrix.shape[0]
        
        # Create binary ground truth matrix
        binary_ground_truth = (ground_truth > 0).astype(int)
        
        # Threshold SMTE matrix to create binary prediction
        threshold = np.percentile(smte_matrix, threshold_percentile)
        predicted = (smte_matrix >= threshold).astype(int)
        
        # Calculate basic metrics
        true_positives = np.sum((predicted == 1) & (binary_ground_truth == 1))
        false_positives = np.sum((predicted == 1) & (binary_ground_truth == 0))
        true_negatives = np.sum((predicted == 0) & (binary_ground_truth == 0))
        false_negatives = np.sum((predicted == 0) & (binary_ground_truth == 1))
        
        # Calculate evaluation metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / (num_series * num_series)
        
        # Calculate direction accuracy
        direction_correct = 0
        direction_total = 0
        
        for i in range(num_series):
            for j in range(num_series):
                if ground_truth[i,j] > 0 or ground_truth[j,i] > 0:
                    direction_total += 1
                    if ground_truth[i,j] > ground_truth[j,i]:
                        # Should have stronger SMTE in this direction
                        if smte_matrix[i,j] > smte_matrix[j,i]:
                            direction_correct += 1
                    elif ground_truth[j,i] > ground_truth[i,j]:
                        if smte_matrix[j,i] > smte_matrix[i,j]:
                            direction_correct += 1
        
        direction_accuracy = direction_correct / direction_total if direction_total > 0 else 0
        
        # Calculate strength correlation
        # Only consider non-zero entries in ground truth
        mask = (ground_truth != 0)
        strength_correlation = np.corrcoef(
            smte_matrix[mask],
            ground_truth[mask]
        )[0,1] if np.sum(mask) > 1 else 0
        
        # Calculate relative strength metric
        true_dep_values = smte_matrix[binary_ground_truth == 1]
        false_dep_values = smte_matrix[binary_ground_truth == 0]
        relative_strength = np.mean(true_dep_values) / np.mean(false_dep_values) if len(false_dep_values) > 0 else np.inf
        
        # Calculate dependency strength accuracy
        strength_error = np.mean(
            np.abs(smte_matrix[mask] - ground_truth[mask])
        ) if np.sum(mask) > 0 else np.inf
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'direction_accuracy': direction_accuracy,
            'strength_correlation': strength_correlation,
            'relative_strength': relative_strength,
            'strength_error': strength_error,
            'confusion_matrix': {
                'true_positives': true_positives,
                'false_positives': false_positives,
                'true_negatives': true_negatives,
                'false_negatives': false_negatives
            }
        }


    def plot_evaluation_results(self, evaluation_metrics: dict):
        """
        Plot evaluation metrics in a comprehensive visualization.
        
        Args:
            evaluation_metrics (dict): Dictionary of evaluation metrics
        """
        plt.figure(figsize=(15, 10))
        
        # Basic metrics plot
        plt.subplot(2, 2, 1)
        basic_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'direction_accuracy']
        values = [evaluation_metrics[metric] for metric in basic_metrics]
        plt.bar(basic_metrics, values)
        plt.title('Basic Performance Metrics')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Confusion matrix plot
        plt.subplot(2, 2, 2)
        cm = np.array([[evaluation_metrics['confusion_matrix']['true_negatives'], 
                    evaluation_metrics['confusion_matrix']['false_positives']],
                    [evaluation_metrics['confusion_matrix']['false_negatives'], 
                    evaluation_metrics['confusion_matrix']['true_positives']]])
        plt.imshow(cm, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xticks([0, 1], ['Negative', 'Positive'])
        plt.yticks([0, 1], ['Negative', 'Positive'])
        
        # Additional metrics
        plt.subplot(2, 2, 3)
        add_metrics = ['strength_correlation', 'relative_strength', 'kl_divergence']
        values = [evaluation_metrics[metric] for metric in add_metrics]
        plt.bar(add_metrics, values)
        plt.title('Advanced Metrics')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()

def run_validation_example():
    """
    Run a validation example using synthetic data with comprehensive evaluation.
    """
    # Initialize analyzer
    analyzer = SMTEAnalyzer(num_symbols=5)
    
    # Create custom dependency configuration
    dependency_config = [
        {'source': 0, 'target': 1, 'strength': 0.5, 'lag': 1},
        {'source': 1, 'target': 2, 'strength': 0.7, 'lag': 1},
        {'source': 0, 'target': 3, 'strength': 0.3, 'lag': 2},
        {'source': 2, 'target': 4, 'strength': 0.6, 'lag': 1}
    ]
    
    # Generate synthetic data with known dependencies
    print("Generating synthetic data...")
    synthetic_data, ground_truth = analyzer.generate_synthetic_data(
        num_timepoints=200,
        num_series=5,
        dependency_config=dependency_config
    )
    
    print("\nGround Truth Dependency Matrix:")
    print(ground_truth)
    
    # Compute SMTE matrix
    print("\nComputing SMTE matrix...")
    smte_matrix = analyzer.compute_smte_matrix(synthetic_data)
    
    # Evaluate SMTE performance
    print("\nEvaluating SMTE performance...")
    evaluation_metrics = analyzer.evaluate_smte_performance(
        smte_matrix,
        ground_truth
    )
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print("-" * 50)
    for metric, value in evaluation_metrics.items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.3f}")
    
    # Plot results
    analyzer.plot_evaluation_results(evaluation_metrics)
    
    # Compare ground truth and SMTE matrices
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(ground_truth, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Dependency Strength')
    plt.title('Ground Truth Dependencies')
    plt.xlabel('Source')
    plt.ylabel('Target')
    
    plt.subplot(1, 2, 2)
    plt.imshow(smte_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='SMTE Value')
    plt.title('Computed SMTE Matrix')
    plt.xlabel('Source')
    plt.ylabel('Target')
    
    plt.tight_layout()
    plt.show()
    
    return smte_matrix, ground_truth, evaluation_metrics

if __name__ == "__main__":
    # Run validation example with evaluation
    smte_matrix, ground_truth, evaluation_metrics = run_validation_example()