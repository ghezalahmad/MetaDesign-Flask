import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
# Assuming this internal import path is correct for your setup
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
    """Evaluates the Random Forest model on candidate samples, aligning output with BO structure."""

    labeled_data = data.dropna(subset=target_columns)
    # Changed condition to match what the Bayesian optimizer expects (any target is null)
    candidate_df = data[data[target_columns[0]].isnull()].copy()

    if candidate_df.empty:
        return pd.DataFrame()

    # 1. Ensure model is trained (RF is trained once, unlike MAML)
    if not getattr(model, 'is_trained', False):
        print("evaluate_rf_model: Model untrained. Training now...")
        model.train(data, input_columns, target_columns)


    train_inputs = labeled_data[input_columns]
    train_targets = labeled_data[target_columns].values
    candidate_inputs = candidate_df[input_columns]
    
    # 2. Run Bayesian Optimization to get Utility Scores
    utility_scores = multi_objective_bayesian_optimization(
        train_inputs=train_inputs,
        train_targets=train_targets,
        candidate_inputs=candidate_inputs,
        weights=weights,
        max_or_min=max_or_min,
        curiosity=curiosity,
        acquisition="UCB",
        strategy="weighted_sum",
        surrogate_model=model,
        input_columns=input_columns
    )

    # 3. Get Predictions explicitly to fill DataFrame columns
    predictions, uncertainties, _ = model.predict_with_uncertainty(candidate_inputs)
    
    # 4. Map Predictions and Uncertainties to columns
    for i, col in enumerate(target_columns):
        # Predictions
        candidate_df[col] = predictions[:, i]
            
        # Uncertainties 
        candidate_df[f"Uncertainty ({col})"] = uncertainties[:, i]

    # 5. Assign Utility Scores
    if utility_scores is None:
        candidate_df["Utility"] = 0.0
    else:
        utility_scores = np.array(utility_scores, dtype=np.float64).flatten()
        # Safety truncation
        n = min(len(utility_scores), len(candidate_df))
        if len(utility_scores) != len(candidate_df):
            candidate_df = candidate_df.iloc[:n].copy()
            utility_scores = utility_scores[:n]
        candidate_df["Utility"] = utility_scores

    candidate_df["Utility"] = pd.to_numeric(candidate_df["Utility"], errors="coerce").fillna(0.0).astype(float)

    # 6. Calculate Aggregate Uncertainty (Mean of target uncertainties)
    candidate_df["Uncertainty"] = np.mean(uncertainties, axis=1)

    # 7. Calculate Novelty (Requires the utility function to be provided or calculated here)
    # Placeholder for Novelty (assuming calculation is elsewhere or using a simpler proxy)
    # The MAML code uses this:
    # X_candidate_np = candidate_inputs.values
    # X_labeled_np = labeled_data[input_columns].values
    # novelty_scores = calculate_novelty(X_candidate_np, X_labeled_np)
    # candidate_df["Novelty"] = novelty_scores
    candidate_df["Novelty"] = candidate_df["Uncertainty"] # Using uncertainty as novelty proxy

    # 8. Exploration / Exploitation
    candidate_df["Exploration"] = candidate_df["Uncertainty"] * (1.0 + curiosity)
    # Note: Exploitation usually refers to the mean utility score, or mean target prediction.
    candidate_df["Exploitation"] = candidate_df[target_columns].mean(axis=1) # Using mean prediction as proxy

    # 9. Selection Flag
    candidate_df["Selected for Testing"] = False
    if not candidate_df["Utility"].empty:
        max_utility_idx = candidate_df["Utility"].idxmax()
        candidate_df.loc[max_utility_idx, "Selected for Testing"] = True

    # 10. Final Sort
    result_df = candidate_df.sort_values(by="Utility", ascending=False).reset_index(drop=True)

    return result_df