import numpy as np
import nibabel as nib
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import KBinsDiscretizer
import warnings
from typing import Tuple, List, Optional, Dict, Union
import os
import pandas as pd
import glob

class FSLSMTEAnalyzer:
    """
    FSL-integrated Symbolic Matrix Transfer Entropy (SMTE) analyzer for fMRI data.
    
    This class handles FSL preprocessed data and implements SMTE analysis,
    including synthetic data validation and visualization tools.
    """
    
    def __init__(self, num_symbols: int = 5):
        """
        Initialize the FSL SMTE analyzer.
        
        Args:
            num_symbols (int): Number of symbols for discretization
        """
        self.num_symbols = num_symbols
        self.feat_dir = None
        self.raw_data = None
        self.preprocessed_data = None
        self.mask = None
        self.tr = None  # Repetition time
        self.confounds = None
        
    def setup_feat_directory(self, feat_dir: str):
        """
        Set up the FEAT directory structure and verify necessary files.
        
        Args:
            feat_dir (str): Path to FEAT directory
        
        Returns:
            bool: True if setup successful
        """
        if not os.path.exists(feat_dir):
            raise ValueError(f"FEAT directory {feat_dir} does not exist")
            
        self.feat_dir = feat_dir
        
        # Check for essential files
        required_files = [
            'filtered_func_data.nii.gz',  # Preprocessed functional data
            'mean_func.nii.gz',           # Mean functional image
            'mask.nii.gz'                 # Brain mask
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(feat_dir, file)):
                missing_files.append(file)
                
        if missing_files:
            raise ValueError(f"Missing required files in FEAT directory: {missing_files}")
            
        return True
        
    def load_feat_data(self, 
                      feat_dir: str, 
                      load_confounds: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Load preprocessed data from FSL FEAT directory.
        
        Args:
            feat_dir (str): Path to FEAT directory
            load_confounds (bool): Whether to load motion parameters and other confounds
            
        Returns:
            Tuple[np.ndarray, Dict]: Preprocessed data and metadata
        """
        # Setup FEAT directory
        self.setup_feat_directory(feat_dir)
        
        # Load preprocessed functional data
        func_img = nib.load(os.path.join(feat_dir, 'filtered_func_data.nii.gz'))
        self.raw_data = func_img.get_fdata()
        
        # Load brain mask
        mask_img = nib.load(os.path.join(feat_dir, 'mask.nii.gz'))
        self.mask = mask_img.get_fdata().astype(bool)
        
        # Get TR from header
        self.tr = float(func_img.header['pixdim'][4])
        
        # Apply mask and reshape to 2D array (voxels × time points)
        masked_data = self.raw_data[self.mask]
        
        if load_confounds:
            self.load_confounds()
            
        # Store metadata
        metadata = {
            'TR': self.tr,
            'dim': func_img.shape[:3],
            'mask': self.mask,
            'confounds': self.confounds
        }
        
        # Standardize the time series
        self.preprocessed_data = self._standardize_time_series(masked_data)
        
        return self.preprocessed_data, metadata
    
    def load_confounds(self):
        """
        Load motion parameters and other confounds from FEAT directory.
        """
        # Load motion parameters
        mc_file = os.path.join(self.feat_dir, 'mc', 'prefiltered_func_data_mcf.par')
        if os.path.exists(mc_file):
            motion_params = pd.read_csv(mc_file, sep='\\s+', header=None,
                                      names=['trans_x', 'trans_y', 'trans_z',
                                            'rot_x', 'rot_y', 'rot_z'])
            self.confounds = {'motion': motion_params}
        
        # Load additional confounds if available
        confounds_file = os.path.join(self.feat_dir, 'confounds.txt')
        if os.path.exists(confounds_file):
            additional_confounds = pd.read_csv(confounds_file, sep='\\s+')
            self.confounds['additional'] = additional_confounds
    
    def extract_roi_time_series(self, 
                              roi_mask: Union[str, np.ndarray], 
                              method: str = 'mean') -> np.ndarray:
        """
        Extract time series from ROIs defined by a mask.
        
        Args:
            roi_mask (Union[str, np.ndarray]): ROI mask file path or array
            method (str): Method for combining voxels ('mean' or 'median')
            
        Returns:
            np.ndarray: ROI time series data
        """
        if isinstance(roi_mask, str):
            roi_img = nib.load(roi_mask)
            roi_mask = roi_img.get_fdata()
            
        if roi_mask.shape[:3] != self.raw_data.shape[:3]:
            raise ValueError("ROI mask dimensions do not match functional data")
            
        # Ensure mask is boolean
        roi_mask = roi_mask.astype(bool)
        
        # Extract time series for each ROI
        unique_rois = np.unique(roi_mask[roi_mask != 0])
        roi_time_series = []
        
        for roi in unique_rois:
            roi_voxels = self.raw_data[roi_mask == roi]
            if method == 'mean':
                roi_ts = np.mean(roi_voxels, axis=0)
            elif method == 'median':
                roi_ts = np.median(roi_voxels, axis=0)
            else:
                raise ValueError("Method must be 'mean' or 'median'")
            roi_time_series.append(roi_ts)
            
        return np.array(roi_time_series)
    
    def process_feat_group(self, 
                         feat_dirs: List[str], 
                         roi_mask: Optional[str] = None) -> Dict:
        """
        Process multiple FEAT directories for group analysis.
        
        Args:
            feat_dirs (List[str]): List of FEAT directories
            roi_mask (str, optional): Path to ROI mask
            
        Returns:
            Dict: Dictionary containing processed data and metadata
        """
        group_data = []
        group_metadata = []
        
        for feat_dir in feat_dirs:
            # Load individual subject data
            data, metadata = self.load_feat_data(feat_dir)
            
            # Extract ROI time series if mask provided
            if roi_mask:
                data = self.extract_roi_time_series(roi_mask)
                
            group_data.append(data)
            group_metadata.append(metadata)
            
        return {
            'data': group_data,
            'metadata': group_metadata
        }

    def compute_smte_matrix(self, time_series_data: np.ndarray) -> np.ndarray:
        """
        Compute pairwise SMTE matrix for all time series.
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

    def _standardize_time_series(self, data: np.ndarray) -> np.ndarray:
        """Standardize time series (same as before)"""
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        std[std < 1e-10] = 1.0
        return (data - mean) / std
    
    def symbolize_time_series(self, time_series: np.ndarray) -> np.ndarray:
        """Symbolize time series (same as before)"""
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
        """
        N = len(symbols)
        gram_matrix = np.equal.outer(symbols, symbols).astype(float)
        return gram_matrix / N

    def compute_joint_gram_matrix(self, symbols_x: np.ndarray, symbols_y: np.ndarray) -> np.ndarray:
        """
        Compute the joint Gram matrix for two symbolic sequences.
        """
        N = len(symbols_x)
        joint_matrix = (np.equal.outer(symbols_x, symbols_x) * 
                    np.equal.outer(symbols_y, symbols_y))
        return joint_matrix.astype(float) / N

    def matrix_entropy(self, gram_matrix: np.ndarray) -> float:
        """
        Compute the second-order matrix entropy.
        """
        trace = np.trace(np.dot(gram_matrix, gram_matrix))
        trace = max(trace, 1e-10)  # Avoid log(0)
        return -np.log(trace)

    def compute_smte(self, symbols_x: np.ndarray, symbols_y: np.ndarray) -> float:
        """
        Compute SMTE from time series Y to X.
        """
        # Gram matrices
        gram_x = self.compute_gram_matrix(symbols_x)
        gram_y = self.compute_gram_matrix(symbols_y)
        gram_joint = self.compute_joint_gram_matrix(symbols_x, symbols_y)
        
        # Entropies
        S_x = self.matrix_entropy(gram_x)
        S_y = self.matrix_entropy(gram_y)
        S_joint = self.matrix_entropy(gram_joint)
        
        # Compute conditional entropy
        S_x_given_y = S_joint - S_y
        
        # Compute SMTE
        smte_y_to_x = S_x_given_y - S_x
        return smte_y_to_x
    
    def save_results(self, 
                    output_dir: str, 
                    smte_matrix: np.ndarray, 
                    graph: nx.DiGraph,
                    metadata: Dict):
        """
        Save SMTE analysis results in FSL-compatible format.
        
        Args:
            output_dir (str): Directory to save results
            smte_matrix (np.ndarray): SMTE matrix
            graph (nx.DiGraph): Directed graph
            metadata (Dict): Analysis metadata
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save SMTE matrix
        np.save(os.path.join(output_dir, 'smte_matrix.npy'), smte_matrix)
        
        # Save graph in GraphML format
        nx.write_graphml(graph, os.path.join(output_dir, 'smte_graph.graphml'))
        
        # Save metadata
        with open(os.path.join(output_dir, 'analysis_info.txt'), 'w') as f:
            f.write(f"FSL SMTE Analysis Results\n")
            f.write(f"------------------------\n")
            f.write(f"Number of symbols: {self.num_symbols}\n")
            f.write(f"TR: {metadata['TR']}\n")
            f.write(f"Dimensions: {metadata['dim']}\n")
            
    def run_full_analysis(self, 
                         feat_dir: str, 
                         output_dir: str,
                         roi_mask: Optional[str] = None,
                         threshold_percentile: float = 95):
        """
        Run complete SMTE analysis on FSL preprocessed data.
        
        Args:
            feat_dir (str): FEAT directory
            output_dir (str): Output directory
            roi_mask (str, optional): Path to ROI mask
            threshold_percentile (float): Percentile threshold for graph edges
        """
        print("Loading FSL preprocessed data...")
        data, metadata = self.load_feat_data(feat_dir)
        
        if roi_mask:
            print("Extracting ROI time series...")
            data = self.extract_roi_time_series(roi_mask)
        
        print("Computing SMTE matrix...")
        smte_matrix = self.compute_smte_matrix(data)
        
        print("Building directed graph...")
        G = self.build_directed_graph(smte_matrix, threshold_percentile)
        
        print("Saving results...")
        self.save_results(output_dir, smte_matrix, G, metadata)
        
        print("Generating visualizations...")
        self.plot_smte_matrix(smte_matrix)
        self.plot_directed_graph(G)
        
        return smte_matrix, G, metadata

def generate_synthetic_data(self, num_timepoints: int, num_series: int, 
                          dependency_strength: float = 0.5) -> np.ndarray:
    """
    Generate synthetic time series data with known dependencies.
    """
    np.random.seed(42)  # For reproducibility
    data = np.random.randn(num_series, num_timepoints)
    
    # Introduce dependencies (series i depends on series i-1)
    for i in range(1, num_series):
        data[i] += dependency_strength * data[i - 1]
        
    return self._standardize_time_series(data)

def build_directed_graph(self, smte_matrix: np.ndarray, 
                        threshold_percentile: float = 95) -> nx.DiGraph:
    """
    Build directed graph from SMTE matrix.
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

def plot_smte_matrix(self, smte_matrix: np.ndarray, 
                     title: str = "SMTE Matrix"):
    """
    Plot the SMTE matrix.
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

# Example usage
def run_fsl_example():
    """
    Example of using FSLSMTEAnalyzer with FSL preprocessed data.
    """
    # Initialize analyzer
    analyzer = FSLSMTEAnalyzer(num_symbols=5)
    
    # Set paths
    feat_dir = '/path/to/feat/directory'  # Replace with actual FEAT directory
    output_dir = '/path/to/output'        # Replace with desired output directory
    roi_mask = '/path/to/roi/mask.nii.gz' # Optional ROI mask
    
    try:
        # Run full analysis
        smte_matrix, G, metadata = analyzer.run_full_analysis(
            feat_dir=feat_dir,
            output_dir=output_dir,
            roi_mask=roi_mask
        )
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return None
    
    return smte_matrix, G, metadata

if __name__ == "__main__":
    # Run example analysis
    results = run_fsl_example()