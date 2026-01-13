"""
Abstract Base Class for Surrogate Models

This module provides the foundational interface that all surrogate models
in MetaDesign must implement. It ensures consistency across models and
enables features like model registry, swappable implementations, and
unified evaluation.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd


class SurrogateModel(ABC):
    """
    Abstract base class for all surrogate models.
    
    All model implementations (PINN, MAML, GP, RF, etc.) should inherit
    from this class and implement the required methods.
    
    Attributes:
        is_trained (bool): Whether the model has been trained
        scaler_x: Feature scaler (fitted during training)
        scaler_y: Target scaler (fitted during training)
        train_min: Minimum target values from training (for soft clipping)
        train_max: Maximum target values from training (for soft clipping)
    """
    
    def __init__(self):
        self._is_trained = False
        self.scaler_x = None
        self.scaler_y = None
        self.train_min = None
        self.train_max = None
        self.input_columns = None
        self.target_columns = None
    
    @property
    def is_trained(self) -> bool:
        """Check if model is ready for inference."""
        return self._is_trained
    
    @is_trained.setter
    def is_trained(self, value: bool):
        self._is_trained = value
    
    @abstractmethod
    def train(self, 
              data: pd.DataFrame,
              input_columns: List[str],
              target_columns: List[str],
              **kwargs) -> 'SurrogateModel':
        """
        Train the model on labeled data.
        
        Args:
            data: Full dataset (labeled + unlabeled)
            input_columns: List of feature column names
            target_columns: List of target column names
            **kwargs: Model-specific hyperparameters
        
        Returns:
            self: The trained model instance
        """
        pass
    
    @abstractmethod
    def predict_with_uncertainty(self,
                                  X: pd.DataFrame,
                                  input_columns: Optional[List[str]] = None,
                                  num_samples: int = 50
                                  ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Generate predictions with uncertainty estimates.
        
        Args:
            X: Input features (DataFrame or numpy array)
            input_columns: Optional column names for ordering
            num_samples: Number of MC samples for uncertainty
        
        Returns:
            tuple: (mean_predictions, uncertainties, posterior_samples)
                - mean_predictions: Shape (n_samples, n_targets)
                - uncertainties: Shape (n_samples, n_targets)
                - posterior_samples: Optional, shape (num_samples, n_samples, n_targets)
        """
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate point predictions (no uncertainty).
        
        Default implementation uses predict_with_uncertainty.
        Override if more efficient method is available.
        
        Args:
            X: Input features
        
        Returns:
            predictions: Shape (n_samples, n_targets)
        """
        mean, _, _ = self.predict_with_uncertainty(X, num_samples=1)
        return mean
    
    def get_input_columns(self) -> Optional[List[str]]:
        """Return the input column names used during training."""
        if self.scaler_x is not None and hasattr(self.scaler_x, 'feature_names_in_'):
            return list(self.scaler_x.feature_names_in_)
        return self.input_columns
    
    def get_target_columns(self) -> Optional[List[str]]:
        """Return the target column names."""
        return self.target_columns
    
    def apply_soft_clipping(self, predictions: np.ndarray, 
                            extension_factor: float = 0.5) -> np.ndarray:
        """
        Apply soft clipping to predictions.
        
        Limits predictions to training range extended by extension_factor.
        This prevents unrealistic extrapolations while allowing exploration.
        
        Args:
            predictions: Raw predictions
            extension_factor: How much to extend beyond training range (0.5 = 50%)
        
        Returns:
            Clipped predictions
        """
        if self.train_min is not None and self.train_max is not None:
            train_range = self.train_max - self.train_min
            soft_min = self.train_min - extension_factor * train_range
            soft_max = self.train_max + extension_factor * train_range
            return np.clip(predictions, soft_min, soft_max)
        return predictions
    
    def store_training_stats(self, labeled_data: pd.DataFrame, 
                              target_columns: List[str]):
        """
        Store statistics from training data for later use.
        
        Args:
            labeled_data: DataFrame with labeled samples
            target_columns: Target column names
        """
        self.train_min = labeled_data[target_columns].min().values
        self.train_max = labeled_data[target_columns].max().values
        self.target_columns = target_columns
    
    def __repr__(self):
        status = "trained" if self.is_trained else "untrained"
        return f"{self.__class__.__name__}({status})"


class MultiTargetWrapper(ABC):
    """
    Abstract base for models that train separate sub-models per target.
    
    Used by MAML, Reptile, and potentially other meta-learning models
    that benefit from independent per-target training.
    """
    
    def __init__(self, target_columns: List[str]):
        self.target_columns = target_columns
        self.models = {}  # Dict[str, SurrogateModel]
        self._is_trained = False
        self.scaler_x = None
    
    @property
    def is_trained(self) -> bool:
        return self._is_trained
    
    @is_trained.setter
    def is_trained(self, value: bool):
        self._is_trained = value
    
    @abstractmethod
    def train(self, data: pd.DataFrame, input_columns: List[str], 
              target_columns: List[str], **kwargs) -> 'MultiTargetWrapper':
        """Train all per-target models."""
        pass
    
    def predict_with_uncertainty(self, X: pd.DataFrame,
                                  input_columns: Optional[List[str]] = None,
                                  num_samples: int = 50
                                  ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Aggregate predictions from all per-target models.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        
        n_samples = len(X)
        n_targets = len(self.target_columns)
        
        all_means = np.zeros((n_samples, n_targets))
        all_stds = np.zeros((n_samples, n_targets))
        
        for i, col in enumerate(self.target_columns):
            model = self.models[col]
            mean, std, _ = model.predict_with_uncertainty(X, input_columns, num_samples)
            all_means[:, i] = mean.flatten()
            all_stds[:, i] = std.flatten()
        
        return all_means, all_stds, None
