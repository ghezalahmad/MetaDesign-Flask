
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
from app.models.bayesian_optimizer import multi_objective_bayesian_optimization

class RFModel:
    """A wrapper for scikit-learn's RandomForestRegressor."""
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def train(self, data, input_columns, target_columns):
        """Trains the model on the provided data."""
        train_data = data.dropna(subset=target_columns)
        if train_data.empty:
            # Handle no training data case by fitting on dummy data
            X_dummy = pd.DataFrame(np.zeros((1, len(input_columns))), columns=input_columns)
            y_dummy = pd.DataFrame(np.zeros((1, len(target_columns))), columns=target_columns)
            self.scaler_x.fit(X_dummy)
            self.scaler_y.fit(y_dummy)
            # The model doesn't need to be fitted as it's just a placeholder
            return

        X = train_data[input_columns]
        y = train_data[target_columns]

        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        self.model.fit(X_scaled, y_scaled)

    def predict_with_uncertainty(self, X):
        """Generates predictions and uncertainties."""
        if not hasattr(self.model, 'estimators_'):
            # If the model hasn't been fitted, return zeros
            return np.zeros((len(X), 1)), np.zeros((len(X), 1))

        X_scaled = self.scaler_x.transform(X)

        predictions_scaled = self.model.predict(X_scaled)
        if predictions_scaled.ndim == 1:
            predictions_scaled = predictions_scaled.reshape(-1, 1)
        predictions = self.scaler_y.inverse_transform(predictions_scaled)

        tree_predictions = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
        uncertainty_scaled = np.var(tree_predictions, axis=0)

        if uncertainty_scaled.ndim > 1:
            uncertainty_scaled = np.mean(uncertainty_scaled, axis=1)

        # This is a simplification; inverse transform might not be perfect for variance
        uncertainty = uncertainty_scaled * (self.scaler_y.scale_ ** 2)

        return predictions, uncertainty.reshape(-1, 1)


def train_rf_model(data, input_columns, target_columns):
    """Trains a Random Forest model."""
    model = RFModel()
    model.train(data, input_columns, target_columns)
    return model, model.scaler_x, model.scaler_y


def evaluate_rf_model(model, data, input_columns, target_columns, curiosity, weights, max_or_min):
    """Evaluates the Random Forest model on candidate samples."""

    train_data = data.dropna(subset=target_columns)
    candidate_samples = data[data[target_columns].isnull().any(axis=1)].copy()

    if candidate_samples.empty:
        return pd.DataFrame()

    utility_scores = multi_objective_bayesian_optimization(
        train_inputs=train_data[input_columns],
        train_targets=train_data[target_columns].values,
        candidate_inputs=candidate_samples[input_columns],
        weights=weights,
        max_or_min=max_or_min,
        curiosity=curiosity,
        surrogate_model=model,
        input_columns=input_columns
    )

    predictions, uncertainties = model.predict_with_uncertainty(candidate_samples[input_columns])

    candidate_samples['Utility'] = utility_scores
    candidate_samples['Uncertainty'] = uncertainties

    for i, col in enumerate(target_columns):
        candidate_samples[f'Prediction {i+1}'] = predictions[:, i]

    return candidate_samples.sort_values(by='Utility', ascending=False).head(10)
