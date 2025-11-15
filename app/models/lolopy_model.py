
import numpy as np
import pandas as pd
from lolopy.learners import RandomForestRegressor

from app.models.bayesian_optimizer import multi_objective_bayesian_optimization
from app.utils.utils import calculate_novelty

class LolopyRFModel:
    """
    A wrapper for lolopy's RandomForestRegressor that handles both single and multi-target
    regression by training a separate model for each target.
    This model is compatible as a surrogate for BayesianOptimization.
    """
    def __init__(self, num_trees=100, input_columns=None, target_columns=None):
        self.num_trees = num_trees
        self.models = []
        self.is_trained = False
        self.input_columns = input_columns
        self.target_columns = target_columns
        self.scaler_x = None

    def train(self, X, y):
        """Trains the model. If y is 2D, it trains one model per column."""
        if isinstance(X, pd.DataFrame):
            self.input_columns = X.columns.tolist()
            X_np = X.values
        else:
            X_np = X

        if isinstance(y, pd.DataFrame):
            self.target_columns = y.columns.tolist()
            y_np = y.values
        else:
            y_np = y

        self.models = []
        X_np = np.array(X_np, dtype=np.double)

        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)

        num_targets = y_np.shape[1]
        for i in range(num_targets):
            model = RandomForestRegressor(num_trees=self.num_trees)
            model.fit(X_np, np.array(y_np[:, i], dtype=np.double))
            self.models.append(model)

        self.is_trained = True
        if self.input_columns:
            self.scaler_x = type('scaler', (), {'feature_names_in_': self.input_columns})()


    def predict_with_uncertainty(self, X, input_columns=None, num_samples=None):
        """Generates predictions and uncertainties from all trained models."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")

        if isinstance(X, pd.DataFrame):
            if self.input_columns:
                X = X[self.input_columns]
            X_np = X.values
        else:
            X_np = X

        X_np = np.array(X_np, dtype=np.double)
        all_predictions = []
        all_uncertainties = []

        for model in self.models:
            predictions, uncertainties = model.predict(X_np, return_std=True)
            all_predictions.append(predictions.reshape(-1, 1))
            all_uncertainties.append(uncertainties.reshape(-1, 1))

        final_predictions = np.hstack(all_predictions)
        final_uncertainties = np.hstack(all_uncertainties)

        return final_predictions, final_uncertainties

def train_lolopy_model(data: pd.DataFrame, input_columns: list, target_columns: list, n_estimators: int = 100):
    """Trains a lolopy RandomForestRegressor model."""
    train_df = data.dropna(subset=target_columns)

    model_wrapper = LolopyRFModel(
        num_trees=n_estimators,
        input_columns=input_columns,
        target_columns=target_columns
    )

    if train_df.empty:
        # Create a dummy model with 8 rows if no training data is available, as lolopy requires at least 8 rows.
        X_train_dummy = pd.DataFrame(np.zeros((8, len(input_columns))), columns=input_columns)
        y_train_dummy = pd.DataFrame(np.zeros((8, len(target_columns))), columns=target_columns)
        model_wrapper.train(X_train_dummy, y_train_dummy)
    else:
        X_train = train_df[input_columns]
        y_train = train_df[target_columns]
        model_wrapper.train(X_train, y_train)

    return model_wrapper, None, None

def evaluate_lolopy_model(model: LolopyRFModel, data: pd.DataFrame, input_columns: list,
                          target_columns: list, curiosity: float, weights_targets: np.ndarray,
                          max_or_min_targets: list[str]):
    """
    Evaluates candidates using the trained Lolopy model as a surrogate in Bayesian Optimization.
    """
    labeled_data = data.dropna(subset=target_columns)
    candidate_df = data[data[target_columns[0]].isnull()].copy()

    if candidate_df.empty:
        return pd.DataFrame()

    train_inputs = labeled_data[input_columns]
    train_targets = labeled_data[target_columns].values
    candidate_inputs = candidate_df[input_columns]

    utility_scores = multi_objective_bayesian_optimization(
        train_inputs=train_inputs,
        train_targets=train_targets,
        candidate_inputs=candidate_inputs,
        weights=weights_targets,
        max_or_min=max_or_min_targets,
        curiosity=curiosity,
        acquisition="UCB",
        strategy="weighted_sum",
        surrogate_model=model,
        input_columns=input_columns
    )

    predictions, uncertainties = model.predict_with_uncertainty(candidate_inputs)

    for i, col in enumerate(target_columns):
        candidate_df[col] = predictions[:, i]
        candidate_df[f"Uncertainty ({col})"] = uncertainties[:, i]

    candidate_df["Utility"] = utility_scores if utility_scores is not None else 0
    candidate_df["Uncertainty"] = np.mean(uncertainties, axis=1)

    # Calculate Novelty
    X_candidate_np = candidate_inputs.values
    X_labeled_np = labeled_data[input_columns].values
    novelty_scores = calculate_novelty(X_candidate_np, X_labeled_np)
    candidate_df["Novelty"] = novelty_scores

    # Simplified Exploration/Exploitation metrics
    candidate_df["Exploration"] = candidate_df["Uncertainty"] * (1 + curiosity)
    candidate_df["Exploitation"] = candidate_df[target_columns].mean(axis=1) # Based on predicted value

    candidate_df["Selected for Testing"] = False
    if not candidate_df.empty and "Utility" in candidate_df.columns:
        # Find the row with the maximum utility and set "Selected for Testing" to True
        max_utility_idx = candidate_df["Utility"].idxmax()
        candidate_df.loc[max_utility_idx, "Selected for Testing"] = True

    result_df = candidate_df.sort_values(by="Utility", ascending=False).reset_index(drop=True)

    return result_df
