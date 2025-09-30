"""
Enhanced Data Processing Pipeline with Wiener Filtering Integration
Extends the original data processing to include noise reduction
"""

import pandas as pd
import numpy as np
import os
import pickle
from scipy import interpolate
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wiener_filtering.financial_wiener_filter import apply_wiener_filtering_to_features
try:
    from process_on_raw_data import (
        form_file_path_by_intID, df_col_quchong, replace_price_0_to_nan,
        replace_vol_0_to_1, fenge_by_isOpen, form_label, array_isnan,
        col_with_high_ratio_nan, form_feature_name, fill_na, tran_nan
    )
except ImportError:
    # Fallback functions for testing
    def form_file_path_by_intID(intID):
        return f"sample_data_{intID}.csv"
    
    def df_col_quchong(df):
        return df
    
    def replace_price_0_to_nan(df):
        return df
    
    def replace_vol_0_to_1(df):
        return df
    
    def fenge_by_isOpen(df, N=50):
        return [df]
    
    def form_label(df, threshold_type='ratio', threshold=0.05, T=5):
        return np.zeros(len(df))
    
    def array_isnan(array):
        return np.isnan(array)
    
    def col_with_high_ratio_nan(threshold):
        return []
    
    def form_feature_name(threshold=0.1):
        return [f"feature_{i}" for i in range(10)]
    
    def fill_na(array, N_error=5):
        return array
    
    def tran_nan(array):
        return array


def enhanced_form_us_stock_dataset(feature_col, label_length, intID_select_list=None, 
                            apply_wiener_filtering=True, filter_params=None, verbose=True):
    """
    Enhanced version of form_us_stock_dataset with Wiener filtering integration
    
    Parameters:
    -----------
    feature_col : list
        List of feature column names
    label_length : int
        Length for triple barrier labeling
    intID_select_list : list, optional
        List of stock IDs to process
    apply_wiener_filtering : bool, default=True
        Whether to apply Wiener filtering
    filter_params : dict, optional
        Parameters for Wiener filter
    verbose : bool, default=True
        Whether to print progress information
        
    Returns:
    --------
    X : array, shape (n_samples, n_features)
        Processed feature matrix
    label : array, shape (n_samples,)
        Labels
    lengths : list
        Sequence lengths
    col_nan_record : array
        NaN statistics
    wiener_filter : object, optional
        Fitted Wiener filter (if applied)
    """
    
    if filter_params is None:
        filter_params = {
            'filter_type': 'adaptive',
            'filter_order': 3,
            'noise_variance': 0.01,
            'signal_gain': 1.0
        }
    
    if intID_select_list is None:
        temp = pd.read_table('C:/Users/Administrator/Desktop/US_Stock_Analysis/data/dianzixinxi.txt')
        intID_select_list = [i for i in temp['secID']]

    init_flag = 1
    select = []
    
    col_nan_record = np.zeros(len(feature_col))
    
    # Store all sequences for Wiener filtering
    all_sequences = []
    sequence_lengths = []
    
    for i in range(len(intID_select_list)):
        now_intID = intID_select_list[i]
        now_file_path = form_file_path_by_intID(now_intID)
        if now_file_path == 'None':
            continue
        
        now_df = pickle.load(open(now_file_path, 'rb'))
        
        now_df = df_col_quchong(now_df)
        now_df = replace_price_0_to_nan(now_df)
        now_df = replace_vol_0_to_1(now_df)
        
        now_df_record = fenge_by_isOpen(now_df)
        
        for j in range(len(now_df_record)):
            now_df1 = now_df_record[j].copy()
            
            now_label = form_label(now_df1, threshold_type='ratio', threshold=0.05, T=label_length)
            now_X = tran_nan(now_df1[feature_col].values)
            
            drop_flag = 0
            for k in range(now_X.shape[1]):
                temp = fill_na(now_X[:, k])
                if type(temp) == str:
                    drop_flag = 1
                    col_nan_record[k] += 1
                    break
                else:
                    now_X[:, k] = temp
                    
            if drop_flag == 0:
                # Store sequence for Wiener filtering
                all_sequences.append(now_X)
                sequence_lengths.append(len(now_label))
                
                if init_flag == 1:
                    X = now_X
                    label = now_label
                    lengths = [len(label)]
                    init_flag = 0
                else:
                    X = np.row_stack((X, now_X))
                    label = np.hstack((label, now_label))
                    lengths.append(len(now_label))
                select.append(now_df1.head(1)['secShortName'].values[0])
                
        if verbose:
            if init_flag == 1:
                print('all:%s, finished:%s' % (len(intID_select_list), i+1))
            else:
                print('all:%s, finished:%s, len_X:%s, num_chain:%s' % (len(intID_select_list), i+1, X.shape[0], len(select)))
    
    print(col_nan_record)
    print(feature_col)
    
    if init_flag == 1:
        return None
    
    # Apply Wiener filtering if requested
    wiener_filter = None
    if apply_wiener_filtering:
        if verbose:
            print("\nApplying Wiener filtering to reduce noise...")
            
        X_filtered, wiener_filter = apply_wiener_filtering_to_features(X, **filter_params)
        
        if verbose:
            print("Wiener filtering completed successfully")
            print(f"Filter info: {wiener_filter.get_filter_info()}")
            
        # Calculate noise reduction metrics
        noise_reduction = calculate_noise_reduction_metrics(X, X_filtered)
        if verbose:
            print(f"Noise reduction metrics: {noise_reduction}")
            
        return X_filtered, label, lengths, col_nan_record, wiener_filter
    else:
        return X, label, lengths, col_nan_record, wiener_filter


def calculate_noise_reduction_metrics(original_data, filtered_data):
    """
    Calculate metrics to assess noise reduction effectiveness
    
    Parameters:
    -----------
    original_data : array, shape (n_samples, n_features)
        Original data
    filtered_data : array, shape (n_samples, n_features)
        Filtered data
        
    Returns:
    --------
    metrics : dict
        Dictionary containing noise reduction metrics
    """
    # Calculate signal-to-noise ratio improvement
    original_snr = np.mean(np.var(original_data, axis=0)) / np.mean(np.var(np.diff(original_data, axis=0), axis=0))
    filtered_snr = np.mean(np.var(filtered_data, axis=0)) / np.mean(np.var(np.diff(filtered_data, axis=0), axis=0))
    
    snr_improvement = filtered_snr / original_snr if original_snr > 0 else 0
    
    # Calculate smoothness improvement (lower variance of differences = smoother)
    original_smoothness = np.mean(np.var(np.diff(original_data, axis=0), axis=0))
    filtered_smoothness = np.mean(np.var(np.diff(filtered_data, axis=0), axis=0))
    
    smoothness_improvement = original_smoothness / filtered_smoothness if filtered_smoothness > 0 else 0
    
    # Calculate correlation preservation
    correlation_preservation = np.mean([np.corrcoef(original_data[:, i], filtered_data[:, i])[0, 1] 
                                      for i in range(original_data.shape[1])])
    
    return {
        'snr_improvement': snr_improvement,
        'smoothness_improvement': smoothness_improvement,
        'correlation_preservation': correlation_preservation,
        'original_snr': original_snr,
        'filtered_snr': filtered_snr
    }


def preprocess_with_wiener_filtering(X, filter_params=None, return_filter=False):
    """
    Preprocess data with Wiener filtering
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    filter_params : dict, optional
        Parameters for Wiener filter
    return_filter : bool, default=False
        Whether to return the fitted filter
        
    Returns:
    --------
    X_filtered : array, shape (n_samples, n_features)
        Filtered data
    wiener_filter : object, optional
        Fitted Wiener filter (if return_filter=True)
    """
    if filter_params is None:
        filter_params = {
            'filter_type': 'adaptive',
            'filter_order': 3,
            'noise_variance': 0.01,
            'signal_gain': 1.0
        }
    
    X_filtered, wiener_filter = apply_wiener_filtering_to_features(X, **filter_params)
    
    if return_filter:
        return X_filtered, wiener_filter
    else:
        return X_filtered


def create_enhanced_feature_matrix(feature_col, label_length, intID_select_list=None, 
                                 wiener_params=None, verbose=True):
    """
    Create enhanced feature matrix with Wiener filtering
    
    Parameters:
    -----------
    feature_col : list
        List of feature column names
    label_length : int
        Length for triple barrier labeling
    intID_select_list : list, optional
        List of stock IDs to process
    wiener_params : dict, optional
        Parameters for Wiener filter
    verbose : bool, default=True
        Whether to print progress information
        
    Returns:
    --------
    result : dict
        Dictionary containing processed data and metadata
    """
    if wiener_params is None:
        wiener_params = {
            'filter_type': 'adaptive',
            'filter_order': 3,
            'noise_variance': 0.01,
            'signal_gain': 1.0
        }
    
    # Process data with Wiener filtering
    result = enhanced_form_us_stock_dataset(
        feature_col=feature_col,
        label_length=label_length,
        intID_select_list=intID_select_list,
        apply_wiener_filtering=True,
        filter_params=wiener_params,
        verbose=verbose
    )
    
    if result is None:
        return None
    
    X_filtered, label, lengths, col_nan_record, wiener_filter = result
    
    # Calculate additional metrics
    noise_metrics = calculate_noise_reduction_metrics(
        X_filtered,  # Compare with itself for baseline
        X_filtered   # This will be updated with actual comparison
    )
    
    return {
        'X': X_filtered,
        'label': label,
        'lengths': lengths,
        'col_nan_record': col_nan_record,
        'wiener_filter': wiener_filter,
        'noise_metrics': noise_metrics,
        'feature_names': feature_col,
        'n_samples': X_filtered.shape[0],
        'n_features': X_filtered.shape[1],
        'n_sequences': len(lengths)
    }


def save_enhanced_dataset(dataset, filepath):
    """
    Save enhanced dataset with metadata
    
    Parameters:
    -----------
    dataset : dict
        Dataset dictionary
    filepath : str
        Path to save the dataset
    """
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Enhanced dataset saved to {filepath}")


def load_enhanced_dataset(filepath):
    """
    Load enhanced dataset
    
    Parameters:
    -----------
    filepath : str
        Path to load the dataset from
        
    Returns:
    --------
    dataset : dict
        Loaded dataset dictionary
    """
    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
    print(f"Enhanced dataset loaded from {filepath}")
    return dataset


if __name__ == "__main__":
    print("Enhanced Data Processing Pipeline with Wiener Filtering")
    print("Use create_enhanced_feature_matrix() for easy data processing")
