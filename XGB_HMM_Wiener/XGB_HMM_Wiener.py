"""
Enhanced XGB-HMM Model with Wiener Filtering Integration
Combines XGBoost, Hidden Markov Models, and Wiener filtering for improved stock prediction
"""

import numpy as np
import pandas as pd
import pickle
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from XGB_HMM.US_stock_GMM_HMM import GMM_HMM
from XGB_HMM.re_estimate_us_stock import re_estimate
from XGB_HMM.predict_us_stock import self_pred
from XGB_HMM.xgb import self_xgb
from wiener_filtering.financial_wiener_filter import apply_wiener_filtering_to_features


class XGB_HMM_Wiener:
    """
    Enhanced XGB-HMM model with integrated Wiener filtering for noise reduction
    """
    
    def __init__(self, n_states=3, filter_order=3, noise_variance=0.01, 
                 signal_gain=1.0, filter_type='adaptive', verbose=True):
        """
        Initialize XGB-HMM-Wiener model
        
        Parameters:
        -----------
        n_states : int, default=3
            Number of hidden states in HMM
        filter_order : int, default=3
            Order of Wiener filter
        noise_variance : float, default=0.01
            Estimated noise variance
        signal_gain : float, default=1.0
            Signal gain parameter
        filter_type : str, default='adaptive'
            Type of Wiener filter ('basic' or 'adaptive')
        verbose : bool, default=True
            Whether to print progress information
        """
        self.n_states = n_states
        self.filter_order = filter_order
        self.noise_variance = noise_variance
        self.signal_gain = signal_gain
        self.filter_type = filter_type
        self.verbose = verbose
        
        # Model components
        self.wiener_filter = None
        self.hmm_model = None
        self.xgb_model = None
        self.transition_matrix = None
        self.prior_probabilities = None
        
        # Training history
        self.log_likelihood_history = []
        self.best_log_likelihood = -np.inf
        self.best_iteration = 0
        
    def _apply_wiener_filtering(self, X):
        """
        Apply Wiener filtering to input features
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input feature matrix
            
        Returns:
        --------
        X_filtered : array, shape (n_samples, n_features)
            Filtered feature matrix
        """
        if self.verbose:
            print("Applying Wiener filtering for noise reduction...")
            
        X_filtered, self.wiener_filter = apply_wiener_filtering_to_features(
            X, 
            filter_type=self.filter_type,
            filter_order=self.filter_order,
            noise_variance=self.noise_variance,
            signal_gain=self.signal_gain
        )
        
        if self.verbose:
            print(f"Wiener filtering completed. Filter info: {self.wiener_filter.get_filter_info()}")
            
        return X_filtered
    
    def _initialize_hmm(self, X_filtered, lengths):
        """
        Initialize HMM with filtered data
        
        Parameters:
        -----------
        X_filtered : array-like, shape (n_samples, n_features)
            Filtered feature matrix
        lengths : list
            Lengths of sequences
            
        Returns:
        --------
        S : array
            Initial state sequence
        A : array
            Initial transition matrix
        gamma : array
            Initial state probabilities
        """
        if self.verbose:
            print("Initializing HMM with filtered data...")
            
        S, A, gamma = GMM_HMM(X_filtered, lengths, self.n_states, verbose=self.verbose)
        
        # Calculate prior probabilities
        prior_pi = np.array([sum(S == i) / len(S) for i in range(self.n_states)])
        
        if self.verbose:
            print(f"HMM initialized with {self.n_states} states")
            print(f"Prior probabilities: {prior_pi}")
            
        return S, A, gamma, prior_pi
    
    def _train_xgb_model(self, X_filtered, gamma):
        """
        Train XGBoost model on filtered data
        
        Parameters:
        -----------
        X_filtered : array-like, shape (n_samples, n_features)
            Filtered feature matrix
        gamma : array, shape (n_samples, n_states)
            State probabilities
            
        Returns:
        --------
        xgb_model : object
            Trained XGBoost model
        """
        if self.verbose:
            print("Training XGBoost model on filtered data...")
            
        xgb_model = self_xgb(X_filtered, gamma, self.n_states)
        
        if self.verbose:
            print("XGBoost model training completed")
            
        return xgb_model
    
    def _update_emission_matrix(self, X_filtered, prior_pi):
        """
        Update emission matrix using XGBoost predictions
        
        Parameters:
        -----------
        X_filtered : array-like, shape (n_samples, n_features)
            Filtered feature matrix
        prior_pi : array, shape (n_states,)
            Prior state probabilities
            
        Returns:
        --------
        B_matrix : array, shape (n_samples, n_states)
            Updated emission matrix
        """
        if self.xgb_model is not None:
            # Use XGBoost predictions
            import xgboost as xgb
            pred = self.xgb_model.predict(xgb.DMatrix(X_filtered))
            B_matrix = pred / prior_pi
        else:
            # Fallback to uniform probabilities
            B_matrix = np.ones((X_filtered.shape[0], self.n_states)) / self.n_states
            
        return B_matrix
    
    def fit(self, X, lengths, max_iterations=50, min_delta=1e-4, stop_patience=3):
        """
        Fit the XGB-HMM-Wiener model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input feature matrix
        lengths : list
            Lengths of sequences
        max_iterations : int, default=50
            Maximum number of EM iterations
        min_delta : float, default=1e-4
            Minimum improvement threshold for convergence
        stop_patience : int, default=3
            Number of iterations without improvement before stopping
            
        Returns:
        --------
        self : object
            Returns self
        """
        if self.verbose:
            print("=" * 60)
            print("Training XGB-HMM-Wiener Model")
            print("=" * 60)
            
        # Step 1: Apply Wiener filtering
        X_filtered = self._apply_wiener_filtering(X)
        
        # Step 2: Initialize HMM
        S, A, gamma, prior_pi = self._initialize_hmm(X_filtered, lengths)
        
        # Step 3: Initialize XGBoost model
        self.xgb_model = self._train_xgb_model(X_filtered, gamma)
        
        # Step 4: EM algorithm
        stop_flag = 0
        iteration = 1
        log_likelihood = -np.inf
        
        self.log_likelihood_history = []
        self.best_log_likelihood = -np.inf
        self.best_iteration = 0
        
        # Store best results
        best_results = {
            'transition_matrix': A.copy(),
            'xgb_model': self.xgb_model,
            'prior_probabilities': prior_pi.copy(),
            'log_likelihood': -np.inf,
            'gamma': gamma.copy()
        }
        
        while stop_flag <= stop_patience and iteration <= max_iterations:
            if self.verbose:
                print(f"\nIteration {iteration}:")
                
            # E-step: Update emission matrix
            B_matrix = self._update_emission_matrix(X_filtered, prior_pi)
            
            # M-step: Re-estimate HMM parameters
            A, gamma = re_estimate(A, B_matrix, prior_pi, lengths)
            
            # Update XGBoost model
            self.xgb_model = self._train_xgb_model(X_filtered, gamma)
            
            # Calculate log-likelihood
            new_S, _, new_log_likelihood = self_pred(B_matrix, lengths, A, prior_pi)
            
            self.log_likelihood_history.append(new_log_likelihood)
            
            # Update best results if improvement
            if new_log_likelihood > self.best_log_likelihood:
                self.best_log_likelihood = new_log_likelihood
                self.best_iteration = iteration
                best_results.update({
                    'transition_matrix': A.copy(),
                    'xgb_model': self.xgb_model,
                    'prior_probabilities': prior_pi.copy(),
                    'log_likelihood': new_log_likelihood,
                    'gamma': gamma.copy()
                })
                
            # Check convergence
            if new_log_likelihood - log_likelihood <= min_delta:
                stop_flag += 1
                if self.verbose:
                    print(f"No improvement for {stop_flag} iterations")
            else:
                stop_flag = 0
                
            log_likelihood = new_log_likelihood
            
            if self.verbose:
                print(f"Log-likelihood: {new_log_likelihood:.6f}")
                print(f"Best so far: {self.best_log_likelihood:.6f} (iteration {self.best_iteration})")
                
            iteration += 1
            
        # Use best results
        self.transition_matrix = best_results['transition_matrix']
        self.xgb_model = best_results['xgb_model']
        self.prior_probabilities = best_results['prior_probabilities']
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("Training Completed")
            print(f"Best log-likelihood: {self.best_log_likelihood:.6f}")
            print(f"Best iteration: {self.best_iteration}")
            print(f"Total iterations: {iteration - 1}")
            print("=" * 60)
            
        return self
    
    def predict_states(self, X):
        """
        Predict hidden states for new data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input feature matrix
            
        Returns:
        --------
        states : array, shape (n_samples,)
            Predicted state sequence
        state_probabilities : array, shape (n_samples, n_states)
            State probabilities
        """
        if not hasattr(self, 'xgb_model') or self.xgb_model is None:
            raise ValueError("Model must be fitted before prediction")
            
        # Apply Wiener filtering
        X_filtered = self.wiener_filter.transform(X)
        
        # Get XGBoost predictions
        import xgboost as xgb
        pred = self.xgb_model.predict(xgb.DMatrix(X_filtered))
        
        # Calculate emission matrix
        B_matrix = pred / self.prior_probabilities
        
        # Predict states using Viterbi algorithm
        states = np.argmax(B_matrix, axis=1)
        state_probabilities = B_matrix / np.sum(B_matrix, axis=1, keepdims=True)
        
        return states, state_probabilities
    
    def predict_market_trend(self, X):
        """
        Predict market trend (up/down/sideways) for new data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input feature matrix
            
        Returns:
        --------
        trends : array, shape (n_samples,)
            Predicted market trends (1: up, -1: down, 0: sideways)
        trend_probabilities : array, shape (n_samples, 3)
            Trend probabilities
        """
        states, state_probabilities = self.predict_states(X)
        
        # Map states to trends (assuming states 0, 1, 2 correspond to down, sideways, up)
        state_to_trend = {0: -1, 1: 0, 2: 1}  # down, sideways, up
        trends = np.array([state_to_trend[state] for state in states])
        
        # Reorder probabilities to match trend order (down, sideways, up)
        trend_probabilities = np.column_stack([
            state_probabilities[:, 0],  # down
            state_probabilities[:, 1],  # sideways
            state_probabilities[:, 2]   # up
        ])
        
        return trends, trend_probabilities
    
    def get_model_info(self):
        """
        Get information about the trained model
        
        Returns:
        --------
        dict : Model information
        """
        return {
            'n_states': self.n_states,
            'filter_order': self.filter_order,
            'noise_variance': self.noise_variance,
            'signal_gain': self.signal_gain,
            'filter_type': self.filter_type,
            'best_log_likelihood': self.best_log_likelihood,
            'best_iteration': self.best_iteration,
            'total_iterations': len(self.log_likelihood_history),
            'is_fitted': hasattr(self, 'xgb_model') and self.xgb_model is not None,
            'wiener_filter_info': self.wiener_filter.get_filter_info() if self.wiener_filter else None
        }
    
    def save_model(self, filepath):
        """
        Save the trained model to file
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        model_data = {
            'transition_matrix': self.transition_matrix,
            'xgb_model': self.xgb_model,
            'prior_probabilities': self.prior_probabilities,
            'wiener_filter': self.wiener_filter,
            'model_info': self.get_model_info(),
            'log_likelihood_history': self.log_likelihood_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model from file
        
        Parameters:
        -----------
        filepath : str
            Path to load the model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.transition_matrix = model_data['transition_matrix']
        self.xgb_model = model_data['xgb_model']
        self.prior_probabilities = model_data['prior_probabilities']
        self.wiener_filter = model_data['wiener_filter']
        self.log_likelihood_history = model_data['log_likelihood_history']
        
        # Update model info
        model_info = model_data['model_info']
        self.best_log_likelihood = model_info['best_log_likelihood']
        self.best_iteration = model_info['best_iteration']
        
        print(f"Model loaded from {filepath}")


def train_xgb_hmm_wiener(X, lengths, **kwargs):
    """
    Convenience function to train XGB-HMM-Wiener model
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input feature matrix
    lengths : list
        Lengths of sequences
    **kwargs : dict
        Additional parameters for the model
        
    Returns:
    --------
    model : XGB_HMM_Wiener
        Trained model
    """
    model = XGB_HMM_Wiener(**kwargs)
    model.fit(X, lengths)
    return model


if __name__ == "__main__":
    print("XGB-HMM-Wiener Model")
    print("Enhanced XGB-HMM with integrated Wiener filtering")
    print("Use train_xgb_hmm_wiener() for easy training")
