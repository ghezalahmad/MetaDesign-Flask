"""
WEBSLAMD-Exact Utility Calculation

This module provides the exact utility calculation as implemented in WEBSLAMD's
ExperimentConductor._calculate_utility method.

Formula:
    utility = apriori_sum + prediction_sum + curiosity * uncertainty_sum

Where:
- predictions are z-score normalized using LABELED data mean/std
- uncertainties are divided by labels_std (not z-scored)
- minimization targets get multiplied by -1
- each target is multiplied by its weight
- final values are summed across targets
"""
import numpy as np
import pandas as pd


def calculate_webslamd_utility(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    labeled_data: pd.DataFrame,
    target_columns: list,
    max_or_min: list,
    weights: list,
    curiosity: float = 0.0,
    apriori_data: pd.DataFrame = None,
    apriori_columns: list = None,
    apriori_max_or_min: list = None,
    apriori_weights: list = None
) -> np.ndarray:
    """
    Calculate utility exactly as WEBSLAMD does.
    
    Args:
        predictions: Model predictions shape (n_samples, n_targets)
        uncertainties: Prediction uncertainties shape (n_samples, n_targets)
        labeled_data: DataFrame containing labeled rows (for calculating mean/std)
        target_columns: List of target column names
        max_or_min: List of 'max' or 'min' for each target
        weights: List of weights for each target
        curiosity: Exploration factor (0 = exploit, 1 = explore)
        apriori_data: Optional DataFrame for a-priori info
        apriori_columns: Optional list of a-priori column names
        apriori_max_or_min: Optional list of 'max'/'min' for a-priori
        apriori_weights: Optional list of weights for a-priori
    
    Returns:
        Utility scores as 1D numpy array
    """
    # Ensure correct shapes
    predictions = np.atleast_2d(predictions)
    uncertainties = np.atleast_2d(uncertainties)
    n_samples = predictions.shape[0]
    n_targets = len(target_columns)
    
    # Get labeled data statistics for normalization
    labels_mean = labeled_data[target_columns].mean(skipna=True)
    labels_std = labeled_data[target_columns].std(skipna=True).replace(0, 1)
    
    # Normalize predictions and uncertainties
    preds_norm = np.zeros((n_samples, n_targets), dtype=float)
    unc_norm = np.zeros((n_samples, n_targets), dtype=float)
    
    for i, col in enumerate(target_columns):
        mean_val = labels_mean.iloc[i] if hasattr(labels_mean, 'iloc') else labels_mean[i]
        std_val = labels_std.iloc[i] if hasattr(labels_std, 'iloc') else labels_std[i]
        
        # Normalize prediction: (value - mean) / std
        preds_norm[:, i] = (predictions[:, i] - mean_val) / std_val
        
        # Invert for minimization targets
        if max_or_min[i].lower() == "min":
            preds_norm[:, i] *= -1
        
        # Apply weight
        preds_norm[:, i] *= weights[i]
        
        # Normalize uncertainty: value / std (NOT z-scored)
        unc_norm[:, i] = uncertainties[:, i] / std_val
        
        # Apply weight
        unc_norm[:, i] *= weights[i]
    
    # Calculate prediction and uncertainty sums
    pred_sum = preds_norm.sum(axis=1)
    unc_sum = unc_norm.sum(axis=1)
    
    # Calculate apriori contribution
    apriori_sum = 0
    if apriori_data is not None and apriori_columns and len(apriori_columns) > 0:
        apriori_std = apriori_data[apriori_columns].std().replace(0, 1)
        apriori_mean = apriori_data[apriori_columns].mean()
        
        # Normalize apriori
        normed_apriori = (apriori_data[apriori_columns] - apriori_mean) / apriori_std
        
        # Invert for minimization
        if apriori_max_or_min:
            for col, direction in zip(apriori_columns, apriori_max_or_min):
                if direction.lower() == 'min':
                    normed_apriori[col] *= -1
        
        # Apply weights
        if apriori_weights:
            for col, weight in zip(apriori_columns, apriori_weights):
                normed_apriori[col] *= weight
        
        apriori_sum = normed_apriori.sum(axis=1).values
    
    # WEBSLAMD utility formula
    utility = apriori_sum + pred_sum + curiosity * unc_sum
    
    return utility
