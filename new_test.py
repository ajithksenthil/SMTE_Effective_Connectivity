import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict
from robust_SMTE import SMTEAnalyzer

def test_enhanced_smte():
    """
    Comprehensive testing of enhanced SMTE analyzer with various dependency patterns.
    """
    print("=== Starting Enhanced SMTE Validation Tests ===\n")
    
    # Initialize analyzer with improved settings
    analyzer = SMTEAnalyzer(
        num_symbols=5,
        noise_scale=0.1,
        min_threshold=0.1
    )
    
    # Define test configurations
    test_configs = [
        # Test 1: Simple sequential dependencies
        {
            'name': 'Sequential Dependencies',
            'config': [
                {'source': 0, 'target': 1, 'strength': 0.6, 'lag': 1},
                {'source': 1, 'target': 2, 'strength': 0.7, 'lag': 1},
                {'source': 2, 'target': 3, 'strength': 0.8, 'lag': 1}
            ],
            'num_series': 4,
            'num_timepoints': 300
        },
        
        # Test 2: Multi-lag dependencies
        {
            'name': 'Multi-lag Dependencies',
            'config': [
                {'source': 0, 'target': 1, 'strength': 0.7, 'lag': 1},
                {'source': 0, 'target': 2, 'strength': 0.5, 'lag': 2},
                {'source': 1, 'target': 3, 'strength': 0.6, 'lag': 3}
            ],
            'num_series': 4,
            'num_timepoints': 300
        },
        
        # Test 3: Complex network with varying strengths
        {
            'name': 'Complex Network',
            'config': [
                {'source': 0, 'target': 1, 'strength': 0.8, 'lag': 1},
                {'source': 0, 'target': 2, 'strength': 0.6, 'lag': 1},
                {'source': 1, 'target': 3, 'strength': 0.7, 'lag': 1},
                {'source': 2, 'target': 3, 'strength': 0.5, 'lag': 1},
                {'source': 1, 'target': 4, 'strength': 0.9, 'lag': 2}
            ],
            'num_series': 5,
            'num_timepoints': 300
        },
        
        # Test 4: Bidirectional dependencies with different strengths
        {
            'name': 'Bidirectional Dependencies',
            'config': [
                {'source': 0, 'target': 1, 'strength': 0.8, 'lag': 1},
                {'source': 1, 'target': 0, 'strength': 0.4, 'lag': 1},
                {'source': 1, 'target': 2, 'strength': 0.7, 'lag': 1},
                {'source': 2, 'target': 1, 'strength': 0.3, 'lag': 1}
            ],
            'num_series': 3,
            'num_timepoints': 300
        }
    ]
    
    all_results = []
    
    for test in test_configs:
        print(f"\nTesting {test['name']}...")
        
        # Generate synthetic data
        synthetic_data, ground_truth = analyzer.generate_synthetic_data(
            num_timepoints=test['num_timepoints'],
            num_series=test['num_series'],
            dependency_config=test['config']
        )
        
        # Compute SMTE matrix
        smte_matrix, lag_matrix = analyzer.compute_smte_matrix(
            synthetic_data,
            max_lag=3
        )
        
        # Evaluate performance
        metrics = analyzer.evaluate_smte_performance(
            smte_matrix,
            ground_truth,
            lag_matrix
        )
        
        # Store results
        results = {
            'name': test['name'],
            'smte_matrix': smte_matrix,
            'ground_truth': ground_truth,
            'lag_matrix': lag_matrix,
            'metrics': metrics
        }
        all_results.append(results)
        
        # Print detailed results
        print_detailed_results(test['name'], metrics, lag_matrix)
        
        # Plot matrices
        plot_matrices_comparison(ground_truth, smte_matrix, lag_matrix, test['name'])
    
    # Plot comparative results
    plot_comparative_analysis(all_results)
    
    return all_results

def print_detailed_results(test_name: str, metrics: Dict, lag_matrix: np.ndarray):
    """
    Print detailed test results with enhanced metrics.
    """
    print(f"\nDetailed Results for {test_name}:")
    print("-" * 60)
    
    # Print main metrics
    print("\nPerformance Metrics:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 
                  'direction_accuracy', 'direction_confidence']:
        if metric in metrics:
            print(f"{metric.replace('_', ' ').title()}: {metrics[metric]:.3f}")
    
    # Print confusion matrix
    cm = metrics['confusion_matrix']
    print("\nConfusion Matrix:")
    print(f"True Positives: {cm['true_positives']}")
    print(f"False Positives: {cm['false_positives']}")
    print(f"True Negatives: {cm['true_negatives']}")
    print(f"False Negatives: {cm['false_negatives']}")
    
    # Print lag statistics
    print("\nLag Statistics:")
    unique_lags = np.unique(lag_matrix[lag_matrix > 0])
    for lag in unique_lags:
        count = np.sum(lag_matrix == lag)
        print(f"Lag {lag}: {count} connections")

def plot_matrices_comparison(ground_truth: np.ndarray, 
                           smte_matrix: np.ndarray,
                           lag_matrix: np.ndarray,
                           title: str):
    """
    Enhanced visualization of ground truth, SMTE, and lag matrices.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Common settings
    vmax = max(np.max(ground_truth), np.max(smte_matrix))
    vmin = min(np.min(ground_truth), np.min(smte_matrix))
    
    # Plot ground truth
    sns.heatmap(ground_truth, ax=axes[0], cmap='YlOrRd',
                vmin=vmin, vmax=vmax,
                annot=True, fmt='.2f',
                cbar_kws={'label': 'Dependency Strength'})
    axes[0].set_title(f'Ground Truth\n{title}')
    
    # Plot SMTE matrix
    sns.heatmap(smte_matrix, ax=axes[1], cmap='YlOrRd',
                vmin=vmin, vmax=vmax,
                annot=True, fmt='.2f',
                cbar_kws={'label': 'SMTE Value'})
    axes[1].set_title(f'SMTE Matrix\n{title}')
    
    # Plot lag matrix
    sns.heatmap(lag_matrix, ax=axes[2], cmap='viridis',
                annot=True, fmt='d',
                cbar_kws={'label': 'Lag'})
    axes[2].set_title(f'Lag Matrix\n{title}')
    
    plt.tight_layout()
    plt.show()

def plot_comparative_analysis(all_results: List[Dict]):
    """
    Enhanced comparative analysis visualization.
    """
    # Prepare data for plotting
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 
                      'direction_accuracy', 'direction_confidence']
    
    # Create performance comparison plot
    plt.figure(figsize=(15, 10))
    
    # Performance metrics comparison
    plt.subplot(2, 1, 1)
    data = []
    for result in all_results:
        for metric in metrics_to_plot:
            if metric in result['metrics']:
                data.append({
                    'Test': result['name'],
                    'Metric': metric,
                    'Value': result['metrics'][metric]
                })
    
    df = pd.DataFrame(data)
    sns.barplot(data=df, x='Test', y='Value', hue='Metric')
    plt.title('Performance Metrics Comparison')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Connection accuracy analysis
    plt.subplot(2, 1, 2)
    accuracy_data = []
    for result in all_results:
        cm = result['metrics']['confusion_matrix']
        total = sum(cm.values())
        accuracy = (cm['true_positives'] + cm['true_negatives']) / total
        accuracy_data.append({
            'Test': result['name'],
            'Accuracy': accuracy,
            'True Positive Rate': cm['true_positives'] / (cm['true_positives'] + cm['false_negatives']),
            'False Positive Rate': cm['false_positives'] / (cm['false_positives'] + cm['true_negatives'])
        })
    
    df_accuracy = pd.DataFrame(accuracy_data)
    df_accuracy.plot(x='Test', kind='bar', rot=45)
    plt.title('Connection Detection Analysis')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run enhanced validation tests
    print("Running enhanced SMTE validation tests...")
    results = test_enhanced_smte()
    
    print("\nValidation testing complete!")