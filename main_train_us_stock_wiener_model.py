"""
Main Training Script for XGB-HMM-Wiener Model
Demonstrates the enhanced model with Wiener filtering integration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from XGB_HMM_Wiener.XGB_HMM_Wiener import XGB_HMM_Wiener, train_xgb_hmm_wiener
from dataset_code.enhanced_data_processing import create_enhanced_feature_matrix, save_enhanced_dataset
from dataset_code.process_us_stock_raw_data import form_feature_name


def load_sample_data():
    """
    Load sample data for demonstration
    In a real scenario, this would load actual stock market data
    """
    print("Loading sample data...")
    
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Generate synthetic financial time series data
    X = np.random.randn(n_samples, n_features)
    
    # Add some structure to make it more realistic
    for i in range(n_features):
        # Add trend component
        trend = np.linspace(0, 2, n_samples) + np.random.randn(n_samples) * 0.1
        X[:, i] += trend
        
        # Add some autocorrelation
        for j in range(1, min(5, n_samples)):
            X[j:, i] += 0.3 * X[:-j, i]
    
    # Generate labels (market trends)
    # Simple rule: if average of first 10 features is positive, label as 1 (up)
    # if negative, label as -1 (down), otherwise 0 (sideways)
    feature_avg = np.mean(X[:, :10], axis=1)
    labels = np.zeros(n_samples)
    labels[feature_avg > 0.5] = 1
    labels[feature_avg < -0.5] = -1
    
    # Create sequence lengths (simulate different stock sequences)
    lengths = [100, 150, 200, 250, 200]  # 5 sequences
    
    print(f"Sample data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Labels distribution: {np.bincount(labels.astype(int) + 1)}")
    print(f"Number of sequences: {len(lengths)}")
    
    return X, labels, lengths


def train_and_evaluate_model(X, labels, lengths, test_size=0.2):
    """
    Train and evaluate the XGB-HMM-Wiener model
    
    Parameters:
    -----------
    X : array, shape (n_samples, n_features)
        Feature matrix
    labels : array, shape (n_samples,)
        Labels
    lengths : list
        Sequence lengths
    test_size : float, default=0.2
        Proportion of data for testing
        
    Returns:
    --------
    results : dict
        Training and evaluation results
    """
    print("\n" + "="*60)
    print("Training XGB-HMM-Wiener Model")
    print("="*60)
    
    # Split data into train and test sets
    # For time series, we need to be careful about the split
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    labels_train = labels[:split_idx]
    labels_test = labels[split_idx:]
    
    # Adjust lengths for training set
    train_lengths = []
    current_idx = 0
    for length in lengths:
        if current_idx + length <= split_idx:
            train_lengths.append(length)
            current_idx += length
        else:
            remaining = split_idx - current_idx
            if remaining > 0:
                train_lengths.append(remaining)
            break
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training sequences: {len(train_lengths)}")
    
    # Initialize and train model
    model = XGB_HMM_Wiener(
        n_states=3,
        filter_order=3,
        noise_variance=0.01,
        signal_gain=1.0,
        filter_type='adaptive',
        verbose=True
    )
    
    # Train the model
    model.fit(X_train, train_lengths, max_iterations=20, min_delta=1e-4)
    
    # Evaluate on training set
    train_states, train_state_probs = model.predict_states(X_train)
    train_trends, train_trend_probs = model.predict_market_trend(X_train)
    
    # Evaluate on test set
    test_states, test_state_probs = model.predict_states(X_test)
    test_trends, test_trend_probs = model.predict_market_trend(X_test)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(labels_train, train_trends)
    test_accuracy = accuracy_score(labels_test, test_trends)
    
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Print detailed results
    print("\nTest Set Classification Report:")
    print(classification_report(labels_test, test_trends, 
                              target_names=['Down', 'Sideways', 'Up']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels_test, test_trends))
    
    # Store results
    results = {
        'model': model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_predictions': train_trends,
        'test_predictions': test_trends,
        'train_state_predictions': train_states,
        'test_state_predictions': test_states,
        'model_info': model.get_model_info()
    }
    
    return results


def plot_training_results(results):
    """
    Plot training results and model performance
    
    Parameters:
    -----------
    results : dict
        Results from training and evaluation
    """
    model = results['model']
    
    # Plot log-likelihood history
    if hasattr(model, 'log_likelihood_history') and model.log_likelihood_history:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(model.log_likelihood_history)
        plt.title('Log-Likelihood During Training')
        plt.xlabel('Iteration')
        plt.ylabel('Log-Likelihood')
        plt.grid(True)
        
        # Plot state predictions
        plt.subplot(2, 2, 2)
        test_states = results['test_state_predictions']
        plt.hist(test_states, bins=3, alpha=0.7, edgecolor='black')
        plt.title('State Distribution (Test Set)')
        plt.xlabel('Hidden State')
        plt.ylabel('Count')
        plt.grid(True)
        
        # Plot trend predictions
        plt.subplot(2, 2, 3)
        test_trends = results['test_predictions']
        trend_counts = np.bincount(test_trends.astype(int) + 1)
        trend_labels = ['Down', 'Sideways', 'Up']
        plt.bar(trend_labels, trend_counts, alpha=0.7, edgecolor='black')
        plt.title('Trend Distribution (Test Set)')
        plt.ylabel('Count')
        plt.grid(True)
        
        # Plot accuracy comparison
        plt.subplot(2, 2, 4)
        accuracies = [results['train_accuracy'], results['test_accuracy']]
        labels = ['Training', 'Test']
        plt.bar(labels, accuracies, alpha=0.7, edgecolor='black')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('wiener_model_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Results plot saved as 'wiener_model_results.png'")


def compare_with_baseline(X, labels, lengths):
    """
    Compare XGB-HMM-Wiener with baseline XGB-HMM model
    
    Parameters:
    -----------
    X : array, shape (n_samples, n_features)
        Feature matrix
    labels : array, shape (n_samples,)
        Labels
    lengths : list
        Sequence lengths
    """
    print("\n" + "="*60)
    print("Comparing with Baseline XGB-HMM Model")
    print("="*60)
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    labels_train = labels[:split_idx]
    labels_test = labels[split_idx:]
    
    train_lengths = []
    current_idx = 0
    for length in lengths:
        if current_idx + length <= split_idx:
            train_lengths.append(length)
            current_idx += length
        else:
            remaining = split_idx - current_idx
            if remaining > 0:
                train_lengths.append(remaining)
            break
    
    # Train XGB-HMM-Wiener model
    print("Training XGB-HMM-Wiener model...")
    wiener_model = XGB_HMM_Wiener(
        n_states=3, filter_order=3, noise_variance=0.01,
        signal_gain=1.0, filter_type='adaptive', verbose=False
    )
    wiener_model.fit(X_train, train_lengths, max_iterations=10)
    
    # Train baseline XGB-HMM model (without Wiener filtering)
    print("Training baseline XGB-HMM model...")
    baseline_model = XGB_HMM_Wiener(
        n_states=3, filter_order=3, noise_variance=0.01,
        signal_gain=1.0, filter_type='basic', verbose=False
    )
    baseline_model.fit(X_train, train_lengths, max_iterations=10)
    
    # Evaluate both models
    wiener_trends, _ = wiener_model.predict_market_trend(X_test)
    baseline_trends, _ = baseline_model.predict_market_trend(X_test)
    
    wiener_accuracy = accuracy_score(labels_test, wiener_trends)
    baseline_accuracy = accuracy_score(labels_test, baseline_trends)
    
    print(f"\nBaseline XGB-HMM Accuracy: {baseline_accuracy:.4f}")
    print(f"XGB-HMM-Wiener Accuracy: {wiener_accuracy:.4f}")
    print(f"Improvement: {wiener_accuracy - baseline_accuracy:.4f}")
    print(f"Relative Improvement: {((wiener_accuracy - baseline_accuracy) / baseline_accuracy * 100):.2f}%")
    
    return {
        'wiener_accuracy': wiener_accuracy,
        'baseline_accuracy': baseline_accuracy,
        'improvement': wiener_accuracy - baseline_accuracy,
        'relative_improvement': (wiener_accuracy - baseline_accuracy) / baseline_accuracy * 100
    }


def main():
    """
    Main function to run the complete training and evaluation pipeline
    """
    print("XGB-HMM-Wiener Model Training Pipeline")
    print("="*60)
    
    # Load data
    X, labels, lengths = load_sample_data()
    
    # Train and evaluate model
    results = train_and_evaluate_model(X, labels, lengths)
    
    # Plot results
    plot_training_results(results)
    
    # Compare with baseline
    comparison = compare_with_baseline(X, labels, lengths)
    
    # Save model
    model_save_path = 'trained_wiener_model.csv'
    results['model'].save_model(model_save_path)
    
    # Print final summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Final Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Improvement over Baseline: {comparison['improvement']:.4f}")
    print(f"Relative Improvement: {comparison['relative_improvement']:.2f}%")
    print(f"Model saved to: {model_save_path}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    results = main()
