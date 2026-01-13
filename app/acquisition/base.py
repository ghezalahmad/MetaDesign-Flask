"""
Acquisition Function Interface and Implementations

This module provides a pluggable interface for different acquisition functions
used in active learning. The WEBSLAMD utility is the default, but UCB, EI,
and other methods can be easily swapped in.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import pandas as pd


class AcquisitionFunction(ABC):
    """
    Abstract base class for acquisition functions.
    
    Acquisition functions score candidate samples based on model predictions
    and uncertainties, balancing exploitation (high predicted values) and
    exploration (high uncertainty).
    """
    
    @abstractmethod
    def compute(self,
                predictions: np.ndarray,
                uncertainties: np.ndarray,
                labeled_data: pd.DataFrame,
                target_columns: List[str],
                max_or_min: List[str],
                weights: np.ndarray,
                curiosity: float = 0.5,
                **kwargs) -> np.ndarray:
        """
        Compute acquisition scores for candidate samples.
        
        Args:
            predictions: Model predictions, shape (n_candidates, n_targets)
            uncertainties: Prediction uncertainties, shape (n_candidates, n_targets)
            labeled_data: DataFrame of already-labeled samples
            target_columns: Names of target columns
            max_or_min: List of 'max' or 'min' for each target
            weights: Importance weights for each target
            curiosity: Exploration-exploitation trade-off (0=exploit, 1=explore)
            **kwargs: Additional method-specific parameters
        
        Returns:
            scores: Acquisition scores, shape (n_candidates,)
                   Higher score = more valuable to sample
        """
        pass
    
    @property
    def name(self) -> str:
        """Return the acquisition function name."""
        return self.__class__.__name__


class WEBSLAMD(AcquisitionFunction):
    """
    WEBSLAMD-style utility function.
    
    This is the current default acquisition function that matches the
    WEBSLAMD framework exactly:
    
    utility = sum(normalized_predictions) + curiosity * sum(normalized_uncertainties)
    
    Where:
    - predictions are z-score normalized using labeled data statistics
    - minimization targets are inverted (multiplied by -1)
    - each target is weighted by its importance
    """
    
    def compute(self,
                predictions: np.ndarray,
                uncertainties: np.ndarray,
                labeled_data: pd.DataFrame,
                target_columns: List[str],
                max_or_min: List[str],
                weights: np.ndarray,
                curiosity: float = 0.5,
                **kwargs) -> np.ndarray:
        """Compute WEBSLAMD utility scores."""
        
        n_samples = predictions.shape[0]
        n_targets = len(target_columns)
        
        # Get labeled data statistics
        labels_mean = labeled_data[target_columns].mean(skipna=True)
        labels_std = labeled_data[target_columns].std(skipna=True).replace(0, 1)
        
        preds_norm = np.zeros((n_samples, n_targets), dtype=float)
        unc_norm = np.zeros((n_samples, n_targets), dtype=float)
        
        for i, col in enumerate(target_columns):
            mean_val = labels_mean.iloc[i]
            std_val = labels_std.iloc[i]
            
            # Z-score normalize prediction
            preds_norm[:, i] = (predictions[:, i] - mean_val) / std_val
            
            # Invert for minimization targets
            if max_or_min[i].lower() == "min":
                preds_norm[:, i] *= -1
            
            # Apply weight
            preds_norm[:, i] *= weights[i]
            
            # Normalize uncertainty (no z-scoring per WEBSLAMD spec)
            unc_norm[:, i] = uncertainties[:, i] / std_val
            unc_norm[:, i] *= weights[i]
        
        # WEBSLAMD formula
        utility = preds_norm.sum(axis=1) + curiosity * unc_norm.sum(axis=1)
        
        return utility


class UCB(AcquisitionFunction):
    """
    Upper Confidence Bound (UCB) acquisition function.
    
    UCB balances exploitation and exploration using:
    UCB = μ + β * σ
    
    where β controls the exploration-exploitation trade-off.
    For multi-objective, we sum the weighted UCB values.
    """
    
    def __init__(self, beta: float = 2.0):
        """
        Args:
            beta: Exploration weight (higher = more exploration)
        """
        self.beta = beta
    
    def compute(self,
                predictions: np.ndarray,
                uncertainties: np.ndarray,
                labeled_data: pd.DataFrame,
                target_columns: List[str],
                max_or_min: List[str],
                weights: np.ndarray,
                curiosity: float = 0.5,
                **kwargs) -> np.ndarray:
        """Compute UCB acquisition scores."""
        
        # Use beta from init, but allow curiosity to modulate
        effective_beta = self.beta * (1 + curiosity)
        
        n_samples = predictions.shape[0]
        n_targets = len(target_columns)
        
        ucb_values = np.zeros((n_samples, n_targets), dtype=float)
        
        for i, col in enumerate(target_columns):
            pred = predictions[:, i]
            unc = uncertainties[:, i]
            
            # UCB formula
            if max_or_min[i].lower() == "min":
                # For minimization: lower is better, so negate
                ucb = -(pred - effective_beta * unc)
            else:
                ucb = pred + effective_beta * unc
            
            ucb_values[:, i] = ucb * weights[i]
        
        return ucb_values.sum(axis=1)


class ExpectedImprovement(AcquisitionFunction):
    """
    Expected Improvement (EI) acquisition function.
    
    EI computes the expected improvement over the current best observation:
    EI = (μ - f_best) * Φ(Z) + σ * φ(Z)
    
    where Z = (μ - f_best) / σ
    """
    
    def compute(self,
                predictions: np.ndarray,
                uncertainties: np.ndarray,
                labeled_data: pd.DataFrame,
                target_columns: List[str],
                max_or_min: List[str],
                weights: np.ndarray,
                curiosity: float = 0.5,
                **kwargs) -> np.ndarray:
        """Compute Expected Improvement scores."""
        from scipy.stats import norm
        
        n_samples = predictions.shape[0]
        n_targets = len(target_columns)
        
        xi = 0.01 * (1 + curiosity)  # Exploration parameter
        
        ei_values = np.zeros((n_samples, n_targets), dtype=float)
        
        for i, col in enumerate(target_columns):
            pred = predictions[:, i]
            unc = uncertainties[:, i] + 1e-10  # Avoid division by zero
            
            # Get best observed value
            if max_or_min[i].lower() == "min":
                f_best = labeled_data[col].min()
                improvement = f_best - pred - xi
            else:
                f_best = labeled_data[col].max()
                improvement = pred - f_best - xi
            
            Z = improvement / unc
            ei = improvement * norm.cdf(Z) + unc * norm.pdf(Z)
            ei = np.maximum(ei, 0)  # EI should be non-negative
            
            ei_values[:, i] = ei * weights[i]
        
        return ei_values.sum(axis=1)


class ThompsonSampling(AcquisitionFunction):
    """
    Thompson Sampling acquisition function.
    
    Samples from the posterior distribution and selects based on sampled values.
    Naturally balances exploration and exploitation.
    """
    
    def compute(self,
                predictions: np.ndarray,
                uncertainties: np.ndarray,
                labeled_data: pd.DataFrame,
                target_columns: List[str],
                max_or_min: List[str],
                weights: np.ndarray,
                curiosity: float = 0.5,
                **kwargs) -> np.ndarray:
        """Compute Thompson Sampling scores."""
        
        n_samples = predictions.shape[0]
        n_targets = len(target_columns)
        
        # Draw samples from posterior (Gaussian approximation)
        ts_values = np.zeros((n_samples, n_targets), dtype=float)
        
        for i, col in enumerate(target_columns):
            # Sample from N(μ, σ²)
            sampled = np.random.normal(predictions[:, i], uncertainties[:, i])
            
            if max_or_min[i].lower() == "min":
                sampled = -sampled
            
            ts_values[:, i] = sampled * weights[i]
        
        return ts_values.sum(axis=1)


# Registry of available acquisition functions
ACQUISITION_FUNCTIONS = {
    'webslamd': WEBSLAMD,
    'ucb': UCB,
    'ei': ExpectedImprovement,
    'thompson': ThompsonSampling,
}


def get_acquisition_function(name: str, **kwargs) -> AcquisitionFunction:
    """
    Factory function to get an acquisition function by name.
    
    Args:
        name: Name of the acquisition function
        **kwargs: Arguments passed to the acquisition function constructor
    
    Returns:
        AcquisitionFunction instance
    """
    if name.lower() not in ACQUISITION_FUNCTIONS:
        raise ValueError(f"Unknown acquisition function: {name}. "
                        f"Available: {list(ACQUISITION_FUNCTIONS.keys())}")
    
    return ACQUISITION_FUNCTIONS[name.lower()](**kwargs)
