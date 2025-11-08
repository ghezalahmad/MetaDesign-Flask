# app/feature_selection.py

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import RobustScaler
from scipy.stats import pearsonr, spearmanr

def select_features(data, input_columns, target_columns, method="mutual_info", k_features=None, verbose=True):
    """
    Automated feature selection for materials discovery using meta-learning principles.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset containing input features and target variables
    input_columns : list
        Column names for input features
    target_columns : list
        Column names for target variables
    method : str
        Feature selection method ('mutual_info', 'correlation', 'spearman')
    k_features : int or None
        Number of features to select. If None, automatically determined
    verbose : bool
        Whether to display information messages
        
    Returns:
    --------
    selected_columns : list
        Names of selected input features
    importance_scores : dict
        Dictionary mapping feature names to importance scores
    """
    # Prepare data for feature selection
    labeled_data = data.dropna(subset=target_columns)
    
    if len(labeled_data) < 4:
        if verbose:
            st.warning("⚠️ Too few labeled samples for feature selection. Using all features.")
        return input_columns, {col: 1.0 for col in input_columns}
    
    # Determine number of features to select
    if k_features is None:
        # Automatically determine based on dataset size and dimensionality
        k_features = min(
            max(3, len(input_columns) // 2),  # At least 3 features, at most half
            max(3, len(labeled_data) // 2)    # At least 3 features, at most half of sample count
        )
    
    # Ensure k_features is at least 3 and at most the number of input columns
    k_features = max(3, min(k_features, len(input_columns)))
    
    if verbose:
        st.info(f"Selecting {k_features} features from {len(input_columns)} input columns using {method} method.")
    
    # Scale input features for better feature selection
    X = labeled_data[input_columns].values
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # For multi-target problems, compute importance for each target and average
    importance_scores = {}
    
    # Initialize importance scores
    for col in input_columns:
        importance_scores[col] = 0.0
    
    # Calculate importance for each target
    for target_col in target_columns:
        y = labeled_data[target_col].values
        
        # Perform feature selection based on the specified method
        if method == "mutual_info":
            # Mutual information (non-linear relationship measure)
            mi_scores = mutual_info_regression(X_scaled, y, random_state=42)
            
            # Update importance scores
            for i, col in enumerate(input_columns):
                importance_scores[col] += mi_scores[i]
        
        elif method == "correlation":
            # Pearson correlation (linear relationship measure)
            corr_scores = []
            for i in range(X_scaled.shape[1]):
                corr, _ = pearsonr(X_scaled[:, i], y)
                corr_scores.append(abs(corr))  # Use absolute correlation
            
            # Update importance scores
            for i, col in enumerate(input_columns):
                importance_scores[col] += corr_scores[i]
        
        elif method == "spearman":
            # Spearman rank correlation (monotonic relationship measure)
            spearman_scores = []
            for i in range(X_scaled.shape[1]):
                corr, _ = spearmanr(X_scaled[:, i], y)
                spearman_scores.append(abs(corr))  # Use absolute correlation
            
            # Update importance scores
            for i, col in enumerate(input_columns):
                importance_scores[col] += spearman_scores[i]
        
        else:
            if verbose:
                st.warning(f"Unknown feature selection method: {method}. Using mutual information.")
            
            # Default to mutual information
            mi_scores = mutual_info_regression(X_scaled, y, random_state=42)
            
            # Update importance scores
            for i, col in enumerate(input_columns):
                importance_scores[col] += mi_scores[i]
    
    # Average importance scores across targets
    for col in input_columns:
        importance_scores[col] /= len(target_columns)
    
    # Sort features by importance
    sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Select top k features
    selected_columns = [f[0] for f in sorted_features[:k_features]]
    
    # Display results if verbose
    if verbose:
        importance_df = pd.DataFrame({
            'Feature': [f[0] for f in sorted_features],
            'Importance': [f[1] for f in sorted_features]
        })
        
        st.write("Feature selection results:")
        st.dataframe(importance_df)
        
        st.info(f"Selected features: {', '.join(selected_columns)}")
    
    return selected_columns, dict(sorted_features)


def meta_feature_selection(data, input_columns, target_columns, min_features=3, max_features=10):
    """
    Enhanced feature selection specifically for meta-learning on materials data.
    Combines multiple selection methods for more robust results without relying on RF.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset containing input features and target variables
    input_columns : list
        Column names for input features
    target_columns : list
        Column names for target variables
    min_features : int
        Minimum number of features to select
    max_features : int
        Maximum number of features to select
        
    Returns:
    --------
    selected_columns : list
        Names of selected input features
    """
    # Determine optimal number of features based on data characteristics
    n_samples = len(data.dropna(subset=target_columns))
    
    # Rule of thumb: Select at most n_samples/3 features to avoid overfitting
    k_features = min(max_features, max(min_features, n_samples // 3))
    
    # For very small datasets, use a more aggressive feature selection
    if n_samples < 10:
        k_features = min(k_features, 5)
    
    st.info(f"Running meta-feature selection to identify key factors from {len(input_columns)} input variables.")
    
    # Run different feature selection methods - all based on information theory or statistics
    # rather than relying on a specific ML model
    methods = ["mutual_info", "correlation", "spearman"]
    all_selected = []
    all_importance = {}
    
    # Initialize all importance scores to zero
    for col in input_columns:
        all_importance[col] = 0.0
    
    # Run each method and collect results
    for method in methods:
        try:
            selected, importance = select_features(
                data, input_columns, target_columns, 
                method=method, k_features=k_features, verbose=False
            )
            all_selected.append(selected)
            
            # Accumulate importance scores
            for col, score in importance.items():
                all_importance[col] += score
        except Exception as e:
            st.warning(f"Feature selection method {method} failed: {str(e)}")
    
    # Count feature occurrence across methods
    feature_counts = {}
    for feature_list in all_selected:
        for feature in feature_list:
            if feature not in feature_counts:
                feature_counts[feature] = 0
            feature_counts[feature] += 1
    
    # Normalize and combine importance scores with occurrence counts
    final_scores = {}
    for col in input_columns:
        # Normalize importance
        norm_importance = all_importance[col] / len(methods)
        # Get occurrence count (or 0 if not selected by any method)
        count = feature_counts.get(col, 0)
        # Final score combines both metrics
        final_scores[col] = (0.5 * norm_importance) + (0.5 * count / len(methods))
    
    # Sort features by final score
    sorted_features = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Select top k features
    final_selection = [f[0] for f in sorted_features[:k_features]]
    
    # Display results
    result_df = pd.DataFrame({
        'Feature': [f[0] for f in sorted_features],
        'Importance Score': [f[1] for f in sorted_features],
        'Selected': [f[0] in final_selection for f in sorted_features]
    })
    
    st.write("Meta-feature selection results:")
    st.dataframe(result_df)
    
    st.success(f"Selected {len(final_selection)} features: {', '.join(final_selection)}")
    
    return final_selection