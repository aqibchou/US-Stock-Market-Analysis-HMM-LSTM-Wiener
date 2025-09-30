"""
Financial Wiener Filter for Stock Market Data Denoising
Adapted from the original Wiener filter implementation for financial time series
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import wiener
from numpy.linalg import inv
import matplotlib.pyplot as plt


class FinancialWienerFilter:
    """
    Wiener Filter implementation specifically designed for financial time series data.
    Handles multiple features and provides noise reduction for stock market predictions.
    """
    
    def __init__(self, filter_order=3, noise_variance=0.01, signal_gain=1.0):
        """
        Initialize the Financial Wiener Filter
        
        Parameters:
        -----------
        filter_order : int, default=3
            Order of the Wiener filter (number of taps)
        noise_variance : float, default=0.01
            Estimated noise variance in the signal
        signal_gain : float, default=1.0
            Signal gain parameter (c in y(n) = c*x(n) + v(n))
        """
        self.filter_order = filter_order
        self.noise_variance = noise_variance
        self.signal_gain = signal_gain
        self.filter_coefficients = None
        self.is_fitted = False
        
    def _kth_diag_indices(self, matrix, k, value):
        """Helper function to set kth diagonal elements"""
        points = []
        rows, cols = np.diag_indices_from(matrix)
        if k < 0:
            rows = rows[-k:]
            cols = cols[:k]
        elif k > 0:
            rows = rows[:-k]
            cols = cols[k:]
        else:
            rows = rows
            cols = cols

        for i in range(len(rows)):
            points.append((rows[i], cols[i]))
        for i in range(len(points)):
            matrix[points[i]] = value
        return matrix
    
    def _compute_autocorrelation(self, signal_data):
        """
        Compute autocorrelation function for the signal
        
        Parameters:
        -----------
        signal_data : array-like
            Input signal data
            
        Returns:
        --------
        Ryy : array
            Autocorrelation function
        """
        n = len(signal_data)
        Ryy = np.zeros(n)
        
        for i in range(n):
            temp = 0
            for i2 in range(i, n):
                temp += signal_data[i2] * signal_data[i2 - i]
            Ryy[i] = (1.0 / n) * temp
            
        return Ryy
    
    def _design_filter(self, signal_data):
        """
        Design Wiener filter coefficients based on signal characteristics
        
        Parameters:
        -----------
        signal_data : array-like
            Input signal for filter design
            
        Returns:
        --------
        filter_coeffs : array
            Wiener filter coefficients
        """
        n = len(signal_data)
        
        # Compute autocorrelation
        Ryy = self._compute_autocorrelation(signal_data)
        
        # Prepare matrices for Wiener-Hopf equation
        a = np.zeros((self.filter_order, self.filter_order))
        b = Ryy[:self.filter_order].copy()
        
        # Adjust for noise variance
        b[0] = b[0] - self.noise_variance
        
        # Fill autocorrelation matrix
        for k in range(self.filter_order):
            a = self._kth_diag_indices(a, k, Ryy[k])
        for k in range(-self.filter_order + 1, 0):
            a = self._kth_diag_indices(a, k, Ryy[-k])
        
        # Solve Wiener-Hopf equation: a * h = b
        try:
            a_inv = np.linalg.inv(a)
            h = np.matmul(a_inv, b) / self.signal_gain
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if matrix is singular
            h = np.matmul(np.linalg.pinv(a), b) / self.signal_gain
            
        return h
    
    def fit(self, X, y=None):
        """
        Fit the Wiener filter to the training data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, optional
            Target values (not used in Wiener filter)
            
        Returns:
        --------
        self : object
            Returns self
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Design filter for each feature
        self.filter_coefficients = []
        for i in range(X.shape[1]):
            feature_data = X[:, i]
            # Remove mean for better filtering
            feature_data = feature_data - np.mean(feature_data)
            filter_coeffs = self._design_filter(feature_data)
            self.filter_coefficients.append(filter_coeffs)
            
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """
        Apply Wiener filter to the input data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data to be filtered
            
        Returns:
        --------
        X_filtered : array, shape (n_samples, n_features)
            Filtered data
        """
        if not self.is_fitted:
            raise ValueError("Filter must be fitted before transform")
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        X_filtered = np.zeros_like(X)
        
        for i in range(X.shape[1]):
            feature_data = X[:, i].copy()
            # Remove mean
            feature_mean = np.mean(feature_data)
            feature_data = feature_data - feature_mean
            
            # Apply filter
            if len(self.filter_coefficients[i]) > 0:
                filtered_feature = signal.convolve(
                    feature_data, 
                    self.filter_coefficients[i], 
                    mode='same'
                )
                # Add mean back
                X_filtered[:, i] = filtered_feature + feature_mean
            else:
                X_filtered[:, i] = feature_data + feature_mean
                
        return X_filtered
    
    def fit_transform(self, X, y=None):
        """
        Fit the filter and transform the data in one step
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, optional
            Target values (not used in Wiener filter)
            
        Returns:
        --------
        X_filtered : array, shape (n_samples, n_features)
            Filtered data
        """
        return self.fit(X, y).transform(X)
    
    def get_filter_info(self):
        """
        Get information about the fitted filter
        
        Returns:
        --------
        dict : Dictionary containing filter information
        """
        if not self.is_fitted:
            return {"error": "Filter not fitted"}
            
        return {
            "filter_order": self.filter_order,
            "noise_variance": self.noise_variance,
            "signal_gain": self.signal_gain,
            "num_features": len(self.filter_coefficients),
            "is_fitted": self.is_fitted
        }


class AdaptiveWienerFilter(FinancialWienerFilter):
    """
    Adaptive Wiener Filter that adjusts parameters based on signal characteristics
    """
    
    def __init__(self, filter_order=3, noise_variance=0.01, signal_gain=1.0, 
                 adaptation_rate=0.1):
        """
        Initialize Adaptive Wiener Filter
        
        Parameters:
        -----------
        adaptation_rate : float, default=0.1
            Rate of adaptation for filter parameters
        """
        super().__init__(filter_order, noise_variance, signal_gain)
        self.adaptation_rate = adaptation_rate
        self.adaptive_noise_variance = noise_variance
        
    def _estimate_noise_variance(self, signal_data):
        """
        Estimate noise variance from signal characteristics
        
        Parameters:
        -----------
        signal_data : array-like
            Input signal data
            
        Returns:
        --------
        noise_var : float
            Estimated noise variance
        """
        # Use high-frequency components to estimate noise
        # Apply high-pass filter to extract noise
        b, a = signal.butter(4, 0.3, btype='high')
        high_freq = signal.filtfilt(b, a, signal_data)
        
        # Estimate noise variance from high-frequency components
        noise_var = np.var(high_freq)
        return max(noise_var, 1e-6)  # Ensure minimum variance
    
    def fit(self, X, y=None):
        """
        Fit the adaptive Wiener filter
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        self.filter_coefficients = []
        self.adaptive_noise_variance = []
        
        for i in range(X.shape[1]):
            feature_data = X[:, i]
            feature_data = feature_data - np.mean(feature_data)
            
            # Estimate noise variance for this feature
            noise_var = self._estimate_noise_variance(feature_data)
            self.adaptive_noise_variance.append(noise_var)
            
            # Design filter with estimated noise variance
            original_noise_var = self.noise_variance
            self.noise_variance = noise_var
            filter_coeffs = self._design_filter(feature_data)
            self.noise_variance = original_noise_var
            
            self.filter_coefficients.append(filter_coeffs)
            
        self.is_fitted = True
        return self


def apply_wiener_filtering_to_features(X, filter_type='adaptive', **kwargs):
    """
    Convenience function to apply Wiener filtering to feature matrix
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input feature matrix
    filter_type : str, default='adaptive'
        Type of filter to use ('basic' or 'adaptive')
    **kwargs : dict
        Additional parameters for the filter
        
    Returns:
    --------
    X_filtered : array, shape (n_samples, n_features)
        Filtered feature matrix
    filter_obj : object
        Fitted filter object
    """
    if filter_type == 'adaptive':
        filter_obj = AdaptiveWienerFilter(**kwargs)
    else:
        filter_obj = FinancialWienerFilter(**kwargs)
        
    X_filtered = filter_obj.fit_transform(X)
    
    return X_filtered, filter_obj


if __name__ == "__main__":
    # Example usage
    print("Financial Wiener Filter Module")
    print("This module provides Wiener filtering for financial time series data")
    print("Use apply_wiener_filtering_to_features() for easy integration")
