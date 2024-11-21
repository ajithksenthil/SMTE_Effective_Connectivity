import numpy as np
from test_SMTE_Analyzer import SMTEAnalyzer
import matplotlib.pyplot as plt
from scipy.stats import entropy

def test_smte_with_synthetic_data():
    """
    Test SMTE analyzer with synthetic data and different dependency patterns.
    """
    print("=== Starting SMTE Validation Tests ===")
    
    # Initialize analyzer
    analyzer = SMTEAnalyzer(num_symbols=5)
    
    # Test different dependency configurations
    test_configs = [
        # Test 1: Simple sequential dependencies
        {
            'name': 'Sequential Dependencies',
            'config': [
                {'source': 0, 'target': 1, 'strength': 0.5, 'lag': 1},
                {'source': 1, 'target': 2, 'strength': 0.7, 'lag': 1},
                {'source': 2, 'target': 3, 'strength': 0.6, 'lag': 1}
            ],
            'num_series': 4,
            'num_timepoints': 200
        },
        
        # Test 2: Multi-lag dependencies
        {
            'name': 'Multi-lag Dependencies',
            'config': [
                {'source': 0, 'target': 1, 'strength': 0.5, 'lag': 1},
                {'source': 0, 'target': 2, 'strength': 0.3, 'lag': 2},
                {'source': 1, 'target': 3, 'strength': 0.4, 'lag': 3}
            ],
            'num_series': 4,
            'num_timepoints': 200
        },
        
        # Test 3: Complex network
        {
            'name': 'Complex Network',
            'config': [
                {'source': 0, 'target': 1, 'strength': 0.6, 'lag': 1},
                {'source': 0, 'target': 2, 'strength': 0.4, 'lag': 1},
                {'source': 1, 'target': 3, 'strength': 0.5, 'lag': 1},
                {'source': 2, 'target': 3, 'strength': 0.3, 'lag': 1},
                {'source': 1, 'target': 4, 'strength': 0.7, 'lag': 2}
            ],
            'num_series': 5,
            'num_timepoints': 200
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
        smte_matrix = analyzer.compute_smte_matrix(synthetic_data)
        
        # Compute KL divergence
        # Normalize matrices to sum to 1 for probability distribution
        smte_prob = smte_matrix / np.sum(smte_matrix)
        ground_prob = ground_truth / np.sum(ground_truth)
        kl_div = entropy(smte_prob.flatten() + 1e-10, ground_prob.flatten() + 1e-10)
        
        # Evaluate performance
        evaluation_metrics = analyzer.evaluate_smte_performance(smte_matrix, ground_truth)
        evaluation_metrics['kl_divergence'] = kl_div
        
        # Store results
        results = {
            'name': test['name'],
            'smte_matrix': smte_matrix,
            'ground_truth': ground_truth,
            'metrics': evaluation_metrics
        }
        all_results.append(results)
        
        # Print results
        print(f"\nResults for {test['name']}:")
        print("-" * 50)
        for metric, value in evaluation_metrics.items():
            if metric != 'confusion_matrix':
                print(f"{metric}: {value:.3f}")
        
        # Plot matrices
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(ground_truth, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Dependency Strength')
        plt.title(f'Ground Truth: {test["name"]}')
        plt.xlabel('Source')
        plt.ylabel('Target')
        
        plt.subplot(1, 2, 2)
        plt.imshow(smte_matrix, cmap='hot', interpolation='nearest')
        plt.colorbar(label='SMTE Value')
        plt.title(f'SMTE Matrix: {test["name"]}')
        plt.xlabel('Source')
        plt.ylabel('Target')
        
        plt.tight_layout()
        plt.show()
        
    return all_results

def plot_comparative_results(all_results):
    """
    Plot comparative results across different test configurations.
    """
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 
                      'direction_accuracy', 'strength_correlation']
    
    plt.figure(figsize=(15, 8))
    
    x = np.arange(len(all_results))
    width = 0.1
    multiplier = 0
    
    for metric in metrics_to_plot:
        values = [result['metrics'][metric] for result in all_results]
        offset = width * multiplier
        plt.bar(x + offset, values, width, label=metric)
        multiplier += 1
    
    plt.xlabel('Test Configuration')
    plt.ylabel('Score')
    plt.title('Comparative Performance Across Test Configurations')
    plt.xticks(x + width * (len(metrics_to_plot)/2 - 0.5), 
               [result['name'] for result in all_results], 
               rotation=45)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run tests
    print("Running SMTE validation tests...")
    results = test_smte_with_synthetic_data()
    
    # Plot comparative results
    print("\nPlotting comparative results...")
    plot_comparative_results(results)
    
    print("\nValidation testing complete!")