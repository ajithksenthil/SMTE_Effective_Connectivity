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
    
    def generate_synthetic_data(self, 
                              num_timepoints: int, 
                              num_series: int, 
                              dependency_strength: float = 0.5) -> np.ndarray:
        """
        Generate synthetic time series data with known dependencies.
        
        Args:
            num_timepoints (int): Number of time points
            num_series (int): Number of time series
            dependency_strength (float): Strength of dependencies between series
            
        Returns:
            np.ndarray: Synthetic time series data
        """
        np.random.seed(42)  # For reproducibility
        data = np.random.randn(num_series, num_timepoints)
        
        # Introduce dependencies (series i depends on series i-1)
        for i in range(1, num_series):
            data[i] += dependency_strength * data[i - 1]
            
        return self._standardize_time_series(data)
    
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

# Example usage and validation
def run_validation_example():
    """
    Run a validation example using synthetic data.
    """
    # Initialize analyzer
    analyzer = SMTEAnalyzer(num_symbols=5)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    synthetic_data = analyzer.generate_synthetic_data(
        num_timepoints=200,
        num_series=10,
        dependency_strength=0.5
    )
    
    # Compute SMTE matrix
    print("Computing SMTE matrix...")
    smte_matrix = analyzer.compute_smte_matrix(synthetic_data)
    
    # Plot SMTE matrix
    print("Plotting SMTE matrix...")
    analyzer.plot_smte_matrix(
        smte_matrix,
        title="SMTE Matrix for Synthetic Data"
    )
    
    # Build and plot directed graph
    print("Building directed graph...")
    G = analyzer.build_directed_graph(smte_matrix)
    
    print("Plotting directed graph...")
    analyzer.plot_directed_graph(G)
    
    return smte_matrix, G

if __name__ == "__main__":
    # Run validation example
    smte_matrix, G = run_validation_example()