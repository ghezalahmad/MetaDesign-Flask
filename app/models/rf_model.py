import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Internal imports
from app.models.base import SurrogateModel
from app.models.bayesian_optimizer import multi_objective_bayesian_optimization 

class RFModel:
    """A wrapper for scikit-learn's RandomForestRegressor, supporting multi-output 
    and prediction with uncertainty (via tree variance).
    """
    def __init__(self, n_estimators=100, random_state=42):
        # RandomForestRegressor naturally supports multi-output targets
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_trained = False # Added for consistency with other models

    def train(self, data, input_columns, target_columns):
        """Trains the model on the provided data."""
        train_data = data.dropna(subset=target_columns)
        
        if train_data.empty or len(train_data) < 2:
            # Handle no training data case or insufficient data
            X_dummy = pd.DataFrame(np.zeros((2, len(input_columns))), columns=input_columns)
            y_dummy = pd.DataFrame(np.zeros((2, len(target_columns))), columns=target_columns)
            self.scaler_x.fit(X_dummy)
            self.scaler_y.fit(y_dummy)
            self.is_trained = False
            return

        X = train_data[input_columns]
        y = train_data[target_columns]

        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Fit the model
        self.model.fit(X_scaled, y_scaled)
        self.is_trained = True

    def predict_with_uncertainty(self, X, input_columns=None, num_samples=None):
        """
        Generates predictions and uncertainties (Standard Deviation).
        
        Returns:
            predictions: (n_samples, n_targets) in original scale
            std_deviations: (n_samples, n_targets) in original scale
            None: RF doesn't provide posterior samples
        """
        if not self.is_trained or not hasattr(self.model, 'estimators_'):
            # If the model hasn't been fitted, return zeros
            n = len(X)
            out_targets = self.scaler_y.scale_.shape[0] if hasattr(self.scaler_y, 'scale_') else 1
            return np.zeros((n, out_targets)), np.zeros((n, out_targets)), None

        # 1. Preprocess Input
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=input_columns)
        X_scaled = self.scaler_x.transform(X_df)

        # 2. Get predictions from all trees
        # tree_predictions shape: (n_estimators, n_samples, n_targets)
        tree_predictions = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
        
        # Reshape tree predictions if only one target was used by RFR (to always be 3D)
        if tree_predictions.ndim == 2:
            tree_predictions = tree_predictions[:, :, np.newaxis]
        
        # 3. Calculate mean and variance in scaled space
        predictions_scaled = np.mean(tree_predictions, axis=0)  # (n_samples, n_targets)
        variance_scaled = np.var(tree_predictions, axis=0)     # (n_samples, n_targets)

        # 4. Inverse Transform Mean
        predictions = self.scaler_y.inverse_transform(predictions_scaled)

        # 5. Inverse Transform Variance to get Std Dev
        # Var(k*X) = k^2 * Var(X). Scaling factor k is self.scaler_y.scale_
        scale_sq = self.scaler_y.scale_ ** 2
        
        # The scale_sq array must be broadcasted across the samples
        variance_original = variance_scaled * scale_sq
        std_original = np.sqrt(np.maximum(variance_original, 1e-12)) # (n_samples, n_targets)

        return predictions, std_original, None


def train_rf_model(data, input_columns, target_columns):
    """Trains a Random Forest model."""
    model = RFModel()
    model.train(data, input_columns, target_columns)
    return model, model.scaler_x, model.scaler_y


def evaluate_rf_model(model, data, input_columns, target_columns, curiosity, weights, max_or_min):
    """Evaluates the Random Forest model using WEBSLAMD utility formula."""
    from app.utils.webslamd_utility import calculate_webslamd_utility
    from app.utils.utils import calculate_novelty

    labeled_data = data.dropna(subset=target_columns)
    candidate_df = data[data[target_columns[0]].isnull()].copy()

    if candidate_df.empty:
        return pd.DataFrame()

    if not getattr(model, 'is_trained', False):
        print("evaluate_rf_model: Model untrained. Training now...")
        model.train(data, input_columns, target_columns)

    candidate_inputs = candidate_df[input_columns]
    
    # 1. Get Predictions and Uncertainties
    predictions, uncertainties, _ = model.predict_with_uncertainty(candidate_inputs)
    
    # 2. Map Predictions and Uncertainties to columns
    for i, col in enumerate(target_columns):
        candidate_df[col] = predictions[:, i]
        candidate_df[f"Uncertainty ({col})"] = uncertainties[:, i]

    # 3. Calculate Utility using centralized function
    utility_scores = calculate_webslamd_utility(
        predictions=predictions,
        uncertainties=uncertainties,
        labeled_data=labeled_data,
        target_columns=target_columns,
        max_or_min=max_or_min,
        weights=weights,
        curiosity=curiosity
    )
    candidate_df["Utility"] = utility_scores
    candidate_df["Utility"] = pd.to_numeric(candidate_df["Utility"], errors="coerce").fillna(0.0).astype(float)

    # 4. Calculate Aggregate Uncertainty
    candidate_df["Uncertainty"] = np.mean(uncertainties, axis=1)

    # 5. Calculate Novelty
    X_candidate_np = candidate_inputs.values
    X_labeled_np = labeled_data[input_columns].values
    novelty_scores = calculate_novelty(X_candidate_np, X_labeled_np)
    candidate_df["Novelty"] = novelty_scores

    # 6. Selection Flag
    candidate_df["Selected for Testing"] = False
    if not candidate_df["Utility"].empty:
        max_utility_idx = candidate_df["Utility"].idxmax()
        candidate_df.loc[max_utility_idx, "Selected for Testing"] = True

    # 7. Final Sort
    result_df = candidate_df.sort_values(by="Utility", ascending=False).reset_index(drop=True)

    return result_df


# =============================================================================
# SURROGATE MODEL WRAPPER (Implements SurrogateModel Interface)
# =============================================================================

class RFSurrogate(SurrogateModel):
    """
    Random Forest Surrogate Model that implements the SurrogateModel interface.
    """
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        super().__init__()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model: Optional[RFModel] = None
    
    def train(self, data: pd.DataFrame, input_columns: List[str],
              target_columns: List[str], **kwargs) -> 'RFSurrogate':
        self._model = RFModel(
            n_estimators=kwargs.get('n_estimators', self.n_estimators),
            random_state=kwargs.get('random_state', self.random_state)
        )
        self._model.train(data, input_columns, target_columns)
        
        self.scaler_x = self._model.scaler_x
        self.scaler_y = self._model.scaler_y
        self.input_columns = input_columns
        self.target_columns = target_columns
        
        labeled_data = data.dropna(subset=target_columns)
        self.store_training_stats(labeled_data, target_columns)
        self.is_trained = self._model.is_trained
        return self
    
    def predict_with_uncertainty(self, X: pd.DataFrame,
                                  input_columns: Optional[List[str]] = None,
                                  num_samples: int = 50
                                  ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if not self.is_trained or self._model is None:
            raise RuntimeError("Model is not trained yet.")
        cols = input_columns or self.input_columns
        return self._model.predict_with_uncertainty(X, input_columns=cols, num_samples=num_samples)