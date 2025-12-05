
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
        """
        Generates predictions and uncertainties from all trained models.
        
        Returns:
            final_predictions: (n_samples, n_targets)
            final_uncertainties: (n_samples, n_targets)
            None: lolopy doesn't provide posterior samples
        """
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

        return final_predictions, final_uncertainties, None

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
    Returns a candidate_df with predictions, uncertainties, Utility, Novelty, etc.
    This version is hardened to ALWAYS provide a numeric 'Utility' column.
    """
    # 1. Split labeled vs candidate rows
    labeled_data = data.dropna(subset=target_columns)
    candidate_df = data[data[target_columns[0]].isnull()].copy()

    if candidate_df.empty:
        print("⚠ evaluate_lolopy_model: No candidate rows found (all targets labeled).")
        return pd.DataFrame()

    # 2. Train inputs / targets / candidate inputs
    train_inputs = labeled_data[input_columns]
    train_targets = labeled_data[target_columns].values
    candidate_inputs = candidate_df[input_columns]

    # 3. Compute utility via Bayesian optimization
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

    # 4. Surrogate predictions and uncertainties
    predictions, uncertainties, _ = model.predict_with_uncertainty(candidate_inputs)

    for i, col in enumerate(target_columns):
        candidate_df[col] = predictions[:, i]
        candidate_df[f"Uncertainty ({col})"] = uncertainties[:, i]

    # 5. Utility: ensure existence and numeric type
    if utility_scores is None:
        print("⚠ evaluate_lolopy_model: utility_scores is None, setting Utility = 0.0")
        candidate_df["Utility"] = 0.0
    else:
        utility_scores = np.array(utility_scores, dtype=np.float64).flatten()
        if len(utility_scores) != len(candidate_df):
            print(
                f"⚠ evaluate_lolopy_model: "
                f"len(utility_scores)={len(utility_scores)} != len(candidate_df)={len(candidate_df)}. "
                f"Using min length."
            )
            n = min(len(utility_scores), len(candidate_df))
            candidate_df = candidate_df.iloc[:n].copy()
            utility_scores = utility_scores[:n]

        candidate_df["Utility"] = utility_scores

    # Make sure Utility is clean
    candidate_df["Utility"] = (
        pd.to_numeric(candidate_df["Utility"], errors="coerce")
        .fillna(0.0)
        .astype(float)
    )

    # 6. Uncertainty (aggregate)
    candidate_df["Uncertainty"] = np.mean(uncertainties, axis=1)

    # 7. Novelty
    X_candidate_np = candidate_inputs.values
    X_labeled_np = labeled_data[input_columns].values
    novelty_scores = calculate_novelty(X_candidate_np, X_labeled_np)
    candidate_df["Novelty"] = novelty_scores

    # 8. Simple Exploration / Exploitation metrics
    candidate_df["Exploration"] = candidate_df["Uncertainty"] * (1.0 + curiosity)
    # Exploitation: mean predicted target across selected targets
    candidate_df["Exploitation"] = candidate_df[target_columns].mean(axis=1)

    # 9. Select best candidate
    candidate_df["Selected for Testing"] = False
    if "Utility" in candidate_df.columns and not candidate_df["Utility"].empty:
        max_utility_idx = candidate_df["Utility"].idxmax()
        candidate_df.loc[max_utility_idx, "Selected for Testing"] = True
    else:
        print("⚠ evaluate_lolopy_model: Utility column is missing or empty when selecting best candidate.")

    # 10. Final sorting
    result_df = candidate_df.sort_values(by="Utility", ascending=False).reset_index(drop=True)

    print("✅ evaluate_lolopy_model: result_df shape =", result_df.shape)
    print("✅ evaluate_lolopy_model: columns =", list(result_df.columns))

    return result_df
