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


def _exploration_weight(curiosity: float) -> float:
    """Return a non-negative exploration weight from the UI curiosity value."""
    try:
        return max(0.0, float(curiosity))
    except (TypeError, ValueError):
        return 0.0


def _target_statistics(labeled_data: pd.DataFrame, target_columns: List[str]) -> tuple[pd.Series, pd.Series]:
    labels_mean = labeled_data[target_columns].mean(skipna=True)
    labels_std = labeled_data[target_columns].std(skipna=True).replace(0, 1)
    return labels_mean, labels_std


def _oriented_normalized_values(
    values: np.ndarray,
    labeled_data: pd.DataFrame,
    target_columns: List[str],
    max_or_min: List[str],
    weights: np.ndarray,
) -> np.ndarray:
    """Normalize target values and orient every objective as maximize."""
    labels_mean, labels_std = _target_statistics(labeled_data, target_columns)
    arr = np.atleast_2d(values).astype(float)
    output = np.zeros((arr.shape[0], len(target_columns)), dtype=float)

    for i, col in enumerate(target_columns):
        output[:, i] = (arr[:, i] - labels_mean.iloc[i]) / labels_std.iloc[i]
        if max_or_min[i].lower() == "min":
            output[:, i] *= -1
        output[:, i] *= weights[i]

    return output


def _normalized_uncertainties(
    uncertainties: np.ndarray,
    labeled_data: pd.DataFrame,
    target_columns: List[str],
    weights: np.ndarray,
) -> np.ndarray:
    """Normalize uncertainties into the same target space as predictions."""
    _, labels_std = _target_statistics(labeled_data, target_columns)
    arr = np.atleast_2d(uncertainties).astype(float)
    output = np.zeros((arr.shape[0], len(target_columns)), dtype=float)

    for i, col in enumerate(target_columns):
        output[:, i] = np.maximum(arr[:, i], 0.0) / labels_std.iloc[i]
        output[:, i] *= abs(weights[i])

    return output


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
        
        preds_norm = _oriented_normalized_values(
            predictions, labeled_data, target_columns, max_or_min, weights
        )
        unc_norm = _normalized_uncertainties(
            uncertainties, labeled_data, target_columns, weights
        )
        
        # WEBSLAMD formula
        utility = preds_norm.sum(axis=1) + _exploration_weight(curiosity) * unc_norm.sum(axis=1)
        
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
        
        effective_beta = self.beta * _exploration_weight(curiosity)
        preds_norm = _oriented_normalized_values(
            predictions, labeled_data, target_columns, max_or_min, weights
        )
        unc_norm = _normalized_uncertainties(
            uncertainties, labeled_data, target_columns, weights
        )
        ucb_values = preds_norm + effective_beta * unc_norm
        
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
        
        xi = 0.01 * (1 + _exploration_weight(curiosity))  # Exploration parameter
        
        ei_values = np.zeros((n_samples, n_targets), dtype=float)
        preds_norm = _oriented_normalized_values(
            predictions, labeled_data, target_columns, max_or_min, weights
        )
        unc_norm = _normalized_uncertainties(
            uncertainties, labeled_data, target_columns, weights
        )
        labeled_norm = _oriented_normalized_values(
            labeled_data[target_columns].values,
            labeled_data,
            target_columns,
            max_or_min,
            weights,
        )
        
        for i, col in enumerate(target_columns):
            pred = preds_norm[:, i]
            unc = unc_norm[:, i] + 1e-10  # Avoid division by zero
            f_best = np.nanmax(labeled_norm[:, i])
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
        
        preds_norm = _oriented_normalized_values(
            predictions, labeled_data, target_columns, max_or_min, weights
        )
        unc_norm = _normalized_uncertainties(
            uncertainties, labeled_data, target_columns, weights
        )
        ts_values = np.random.normal(
            preds_norm,
            unc_norm * (1.0 + _exploration_weight(curiosity))
        )
        
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
