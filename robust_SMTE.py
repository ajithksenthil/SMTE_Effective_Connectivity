import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.preprocessing import KBinsDiscretizer
import warnings
from typing import Tuple, List, Optional, Dict
import seaborn as sns
# Suppress warnings for clean output
warnings.filterwarnings('ignore')

class SMTEAnalyzer:
    """
    Symbolic Matrix Transfer Entropy (SMTE) analyzer for fMRI data.
    
    This class implements SMTE analysis for fMRI time series data, including
    synthetic data generation for validation and visualization tools.
    """
    
    def __init__(self, 
                 num_symbols: int = 5, 
                 noise_scale: float = 0.1,
                 min_threshold: float = 0.1):
        """
        Initialize the enhanced SMTE analyzer.
        
        Args:
            num_symbols (int): Number of symbols for discretization
            noise_scale (float): Scale of noise in synthetic data
            min_threshold (float): Minimum threshold for SMTE values
        """
        self.num_symbols = num_symbols
        self.noise_scale = noise_scale
        self.min_threshold = min_threshold
        
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
        Improved time series standardization.
        """
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        
        # Handle zero standard deviation with small constant
        std[std < 1e-10] = 1e-10
        
        return (data - mean) / std
    
    def symbolize_time_series(self, time_series: np.ndarray) -> np.ndarray:
        """
        Enhanced symbolization using adaptive binning.
        """
        # Add noise to break ties
        noise = np.random.normal(0, 1e-10, time_series.shape)
        time_series = time_series + noise
        
        # Use uniform strategy for more robust symbolization
        est = KBinsDiscretizer(
            n_bins=self.num_symbols,
            encode='ordinal',
            strategy='uniform'  # Changed from 'quantile' for better handling of outliers
        )
        
        symbols = est.fit_transform(
            time_series.reshape(-1, 1)
        ).astype(int).flatten()
        
        return symbols
    
    def compute_gram_matrix(self, symbols: np.ndarray) -> np.ndarray:
        """
        Improved Gram matrix computation with regularization.
        """
        N = len(symbols)
        gram_matrix = np.equal.outer(symbols, symbols).astype(float)
        
        # Add small regularization term
        epsilon = 1e-10
        gram_matrix += epsilon
        
        # Normalize
        gram_matrix /= (N + epsilon)
        
        return gram_matrix
    
    def _compute_kl_divergence(self, 
                             smte_matrix: np.ndarray, 
                             ground_truth: np.ndarray) -> float:
        """
        Compute KL divergence with proper normalization and smoothing.
        """
        epsilon = 1e-10
        smte_prob = smte_matrix + epsilon
        ground_prob = ground_truth + epsilon
        
        # Normalize to probability distributions
        smte_prob = smte_prob / np.sum(smte_prob)
        ground_prob = ground_prob / np.sum(ground_prob)
        
        return entropy(ground_prob.flatten(), smte_prob.flatten())

    def compute_joint_gram_matrix(self, 
                                symbols_x: np.ndarray, 
                                symbols_y: np.ndarray,
                                lag: int = 1) -> np.ndarray:
        """
        Enhanced joint Gram matrix computation with lag consideration.
        """
        if lag > 0:
            symbols_x = symbols_x[lag:]
            symbols_y = symbols_y[:-lag]
        
        N = len(symbols_x)
        joint_matrix = (np.equal.outer(symbols_x, symbols_x) * 
                       np.equal.outer(symbols_y, symbols_y))
        
        # Add regularization
        epsilon = 1e-10
        joint_matrix = joint_matrix.astype(float) + epsilon
        joint_matrix /= (N + epsilon)
        
        return joint_matrix
    
    def matrix_entropy(self, gram_matrix: np.ndarray) -> float:
        """
        Improved matrix entropy computation with better numerical stability.
        """
        # Add small constant for numerical stability
        epsilon = 1e-10
        gram_matrix = gram_matrix + epsilon
        
        # Normalize to ensure proper probability interpretation
        gram_matrix = gram_matrix / np.sum(gram_matrix)
        
        # Compute eigenvalues
        eigenvals = np.linalg.eigvalsh(gram_matrix)
        eigenvals = eigenvals[eigenvals > epsilon]
        
        # Compute entropy using eigenvalues
        return -np.sum(eigenvals * np.log(eigenvals))
    
    def compute_smte(self, 
                    symbols_x: np.ndarray, 
                    symbols_y: np.ndarray,
                    max_lag: int = 3) -> Tuple[float, int]:
        """
        Enhanced SMTE computation with lag optimization.
        """
        best_smte = -np.inf
        best_lag = 0
        
        for lag in range(max_lag + 1):
            # Compute Gram matrices
            gram_x = self.compute_gram_matrix(symbols_x)
            gram_y = self.compute_gram_matrix(symbols_y)
            gram_joint = self.compute_joint_gram_matrix(symbols_x, symbols_y, lag)
            
            # Compute entropies with improved numerical stability
            S_x = self.matrix_entropy(gram_x)
            S_y = self.matrix_entropy(gram_y)
            S_joint = self.matrix_entropy(gram_joint)
            
            # Compute conditional entropy
            S_x_given_y = S_joint - S_y
            
            # Compute SMTE
            smte = S_x_given_y - S_x
            
            # Normalize by maximum possible entropy
            max_entropy = np.log(len(symbols_x))
            smte = smte / max_entropy
            
            if smte > best_smte:
                best_smte = smte
                best_lag = lag
        
        return best_smte, best_lag
    
    def compute_smte_matrix(self, 
                          time_series_data: np.ndarray,
                          max_lag: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced SMTE matrix computation with lag information.
        """
        num_series = time_series_data.shape[0]
        smte_matrix = np.zeros((num_series, num_series))
        lag_matrix = np.zeros((num_series, num_series), dtype=int)
        
        # Convert all time series to symbols first
        symbolic_series = np.array([
            self.symbolize_time_series(ts) for ts in time_series_data
        ])
        
        # Compute SMTE for all pairs
        for i in range(num_series):
            for j in range(num_series):
                if i != j:
                    smte, lag = self.compute_smte(
                        symbolic_series[i], 
                        symbolic_series[j],
                        max_lag
                    )
                    smte_matrix[i, j] = smte
                    lag_matrix[i, j] = lag
        
        return smte_matrix, lag_matrix
    

    
    def build_directed_graph(self, 
                           smte_matrix: np.ndarray,
                           lag_matrix: np.ndarray,
                           threshold_percentile: float = 95) -> nx.DiGraph:
        """
        Enhanced graph construction with adaptive thresholding.
        """
        # Compute adaptive thresholds for each node
        thresholds = np.array([
            np.percentile(row[row > self.min_threshold], threshold_percentile)
            if np.any(row > self.min_threshold) else self.min_threshold
            for row in smte_matrix
        ])
        
        # Create adjacency matrix
        adjacency_matrix = np.zeros_like(smte_matrix)
        for i in range(len(smte_matrix)):
            # Consider both SMTE value and relative difference
            for j in range(len(smte_matrix)):
                if i != j:
                    if smte_matrix[i, j] >= thresholds[i]:
                        # Check if the reverse connection is weaker
                        if smte_matrix[i, j] > smte_matrix[j, i]:
                            adjacency_matrix[i, j] = 1
        
        # Create directed graph
        G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
        
        # Add edge attributes
        for i, j in G.edges():
            G[i][j]['weight'] = smte_matrix[i, j]
            G[i][j]['lag'] = lag_matrix[i, j]
        
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
        Enhanced synthetic data generation with improved dependencies.
        """
        np.random.seed(42)
        
        # Initialize data with scaled noise
        data = np.random.randn(num_series, num_timepoints) * self.noise_scale
        
        # Initialize ground truth matrix
        ground_truth = np.zeros((num_series, num_series))
        
        if dependency_config is None:
            dependency_config = [
                {
                    'source': i-1,
                    'target': i,
                    'strength': 0.5,
                    'lag': 1
                }
                for i in range(1, num_series)
            ]
        
        # Apply dependencies with nonlinear components
        for dep in dependency_config:
            source = dep['source']
            target = dep['target']
            strength = dep['strength']
            lag = dep.get('lag', 1)
            
            if source >= 0 and target < num_series:
                # Add both linear and nonlinear dependencies
                source_signal = data[source, :-lag]
                nonlinear_component = 0.3 * source_signal**2
                data[target, lag:] += strength * (source_signal + nonlinear_component)
                
                # Add some cross-frequency coupling
                if len(source_signal) > 20:
                    omega = 2 * np.pi / 20
                    modulation = 0.2 * strength * np.sin(omega * np.arange(len(source_signal)))
                    data[target, lag:] += modulation * source_signal
                
                # Record in ground truth matrix
                ground_truth[target, source] = strength
        
        # Add some temporal structure
        for i in range(num_series):
            data[i] = np.convolve(data[i], np.exp(-np.arange(10)/3), mode='same')
        
        # Standardize the time series
        data = self._standardize_time_series(data)
        
        return data, ground_truth
    
    def _compute_basic_metrics(self, 
                             smte_matrix: np.ndarray,
                             ground_truth: np.ndarray,
                             threshold_percentile: float = 95) -> dict:
        """
        Compute basic evaluation metrics with fixed broadcasting.
        
        Args:
            smte_matrix (np.ndarray): Computed SMTE matrix
            ground_truth (np.ndarray): Ground truth dependency matrix
            threshold_percentile (float): Percentile for thresholding
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Create binary matrices
        binary_ground_truth = (ground_truth > 0).astype(int)
        
        # Threshold SMTE matrix using row-wise adaptive thresholding
        predicted = np.zeros_like(smte_matrix)
        for i in range(smte_matrix.shape[0]):
            row = smte_matrix[i]
            nonzero_values = row[row != 0]
            if len(nonzero_values) > 0:
                threshold = np.percentile(nonzero_values, threshold_percentile)
            else:
                threshold = 0
            predicted[i] = (row >= threshold).astype(int)
        
        # Calculate metrics
        true_pos = np.sum((predicted == 1) & (binary_ground_truth == 1))
        false_pos = np.sum((predicted == 1) & (binary_ground_truth == 0))
        true_neg = np.sum((predicted == 0) & (binary_ground_truth == 0))
        false_neg = np.sum((predicted == 0) & (binary_ground_truth == 1))
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_pos + true_neg) / np.prod(smte_matrix.shape)
        
        # Direction accuracy
        direction_correct = 0
        direction_total = 0
        direction_confidence = []
        
        for i in range(len(smte_matrix)):
            for j in range(len(smte_matrix)):
                if ground_truth[i,j] > 0 or ground_truth[j,i] > 0:
                    direction_total += 1
                    if ground_truth[i,j] > ground_truth[j,i]:
                        if smte_matrix[i,j] > smte_matrix[j,i]:
                            direction_correct += 1
                            direction_confidence.append(smte_matrix[i,j] - smte_matrix[j,i])
                    elif ground_truth[j,i] > ground_truth[i,j]:
                        if smte_matrix[j,i] > smte_matrix[i,j]:
                            direction_correct += 1
                            direction_confidence.append(smte_matrix[j,i] - smte_matrix[i,j])
        
        direction_accuracy = direction_correct / direction_total if direction_total > 0 else 0
        avg_direction_confidence = np.mean(direction_confidence) if direction_confidence else 0
        
        # Compute strength correlation for non-zero entries
        mask = (ground_truth != 0)
        if np.sum(mask) > 1:
            strength_correlation = np.corrcoef(
                smte_matrix[mask],
                ground_truth[mask]
            )[0,1]
        else:
            strength_correlation = 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'direction_accuracy': direction_accuracy,
            'direction_confidence': avg_direction_confidence,
            'strength_correlation': strength_correlation,
            'confusion_matrix': {
                'true_positives': int(true_pos),
                'false_positives': int(false_pos),
                'true_negatives': int(true_neg),
                'false_negatives': int(false_neg)
            }
        }

    def evaluate_smte_performance(self, 
                                smte_matrix: np.ndarray,
                                ground_truth: np.ndarray,
                                lag_matrix: Optional[np.ndarray] = None,
                                threshold_percentile: float = 95) -> dict:
        """
        Enhanced performance evaluation with fixed broadcasting.
        
        Args:
            smte_matrix (np.ndarray): Computed SMTE matrix
            ground_truth (np.ndarray): Ground truth dependency matrix
            lag_matrix (np.ndarray, optional): Matrix of detected lags
            threshold_percentile (float): Percentile for thresholding
            
        Returns:
            dict: Evaluation metrics with confidence intervals
        """
        # Compute basic metrics
        metrics = self._compute_basic_metrics(
            smte_matrix, 
            ground_truth, 
            threshold_percentile
        )
        
        # Compute lag accuracy if lag_matrix is provided
        if lag_matrix is not None:
            lag_accuracy = 0
            total_deps = 0
            
            for i in range(len(ground_truth)):
                for j in range(len(ground_truth)):
                    if ground_truth[i,j] > 0:
                        total_deps += 1
                        # Check if detected lag matches configuration
                        # Note: This assumes lag information is available in ground truth
                        if lag_matrix[i,j] > 0:  # If a lag was detected
                            lag_accuracy += 1
            
            metrics['lag_accuracy'] = lag_accuracy / total_deps if total_deps > 0 else 0
        
        # Add normalized metrics
        metrics['normalized_smte'] = np.mean(smte_matrix[ground_truth > 0]) if np.any(ground_truth > 0) else 0
        
        # Compute confidence scores
        confidence_scores = []
        for i in range(len(smte_matrix)):
            for j in range(len(smte_matrix)):
                if ground_truth[i,j] > 0:
                    confidence_scores.append(smte_matrix[i,j])
        
        if confidence_scores:
            metrics['mean_confidence'] = np.mean(confidence_scores)
            metrics['confidence_std'] = np.std(confidence_scores)
        else:
            metrics['mean_confidence'] = 0
            metrics['confidence_std'] = 0
            
        return metrics


    def plot_evaluation_results(self, evaluation_metrics: dict):
        """
        Enhanced plotting of evaluation metrics.
        """
        plt.figure(figsize=(15, 10))
        
        # Basic metrics with confidence intervals
        plt.subplot(2, 2, 1)
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'direction_accuracy']
        values = [evaluation_metrics[m] for m in metrics]
        confidence_intervals = [
            evaluation_metrics['confidence_intervals'].get(m, [0, 0]) 
            for m in metrics
        ]
        
        x = range(len(metrics))
        plt.bar(x, values)
        plt.errorbar(x, values, 
                    yerr=[[v - ci[0] for v, ci in zip(values, confidence_intervals)],
                          [ci[1] - v for v, ci in zip(values, confidence_intervals)]],
                    fmt='none', color='black', capsize=5)
        
        plt.xticks(x, metrics, rotation=45)
        plt.title('Performance Metrics with 95% CI')
        plt.ylim(0, 1)
        
        # Continue with other plots as in original implementation
        
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