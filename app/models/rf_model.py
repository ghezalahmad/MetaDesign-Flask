
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
import streamlit as st

from app.models.bayesian_optimizer import multi_objective_bayesian_optimization
from app.utils.utils import calculate_novelty

class RFModel:
    def __init__(self, n_estimators=100, random_state=42, **kwargs):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            **kwargs
        )
        self.scaler_x = None
        self.scaler_y = None
        self.is_trained = False
        self.input_columns = None

    def train(self, data: pd.DataFrame, input_columns: list[str], target_columns: list[str], perform_grid_search=False):
        labeled_data = data.dropna(subset=target_columns).copy()
        if len(labeled_data) < 5:
            st.error("Not enough labeled samples for Random Forest training.")
            return self, None, None

        X = labeled_data[input_columns]
        y = labeled_data[target_columns]
        self.input_columns = input_columns

        self.scaler_x = RobustScaler()
        self.scaler_y = RobustScaler()

        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)

        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)

        y_train_final = y_scaled.ravel() if y_scaled.shape[1] == 1 else y_scaled

        if perform_grid_search and len(labeled_data) >= 10:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
            }
            grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                                       param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
            grid_search.fit(X_scaled, y_train_final)
            self.model = grid_search.best_estimator_
        else:
            self.model.fit(X_scaled, y_train_final)

        self.is_trained = True
        self.scaler_x.feature_names_in_ = input_columns
        return self, self.scaler_x, self.scaler_y

    def predict(self, X_unlabeled: pd.DataFrame):
        predictions, _ = self.predict_with_uncertainty(X_unlabeled)
        return predictions

    def predict_with_uncertainty(self, X_unlabeled: pd.DataFrame, input_columns=None, num_samples=None):
        if not self.is_trained or self.scaler_x is None or self.scaler_y is None:
            raise RuntimeError("Model is not trained yet or scalers are missing.")

        if input_columns is None:
            input_columns = self.input_columns

        X_scaled = self.scaler_x.transform(X_unlabeled[input_columns])
        tree_predictions = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])

        mean_predictions_scaled = np.mean(tree_predictions, axis=0)
        variance_scaled = np.var(tree_predictions, axis=0)

        if self.model.n_outputs_ == 1:
            predictions_original = self.scaler_y.inverse_transform(mean_predictions_scaled.reshape(-1, 1))
            variance_original = variance_scaled * (self.scaler_y.scale_**2)
            std_dev_original = np.sqrt(variance_original).reshape(-1, 1)
        else:
            predictions_original = self.scaler_y.inverse_transform(mean_predictions_scaled)
            variance_original = variance_scaled * (self.scaler_y.scale_**2)
            std_dev_original = np.sqrt(variance_original)

        return predictions_original, std_dev_original

    def _get_input_columns(self):
        return self.input_columns

def train_rf_model(data: pd.DataFrame, input_columns: list[str], target_columns: list[str], **kwargs):
    model = RFModel(**kwargs)
    model.train(data, input_columns, target_columns)
    return model, model.scaler_x, model.scaler_y

def evaluate_rf_model(rf_model: RFModel, data: pd.DataFrame, input_columns: list[str],
                      target_columns: list[str], curiosity: float, weights: np.ndarray,
                      max_or_min: list[str]):
    if not rf_model.is_trained:
        st.error("Random Forest model is not trained.")
        return pd.DataFrame(columns=input_columns + target_columns + ["Utility", "Uncertainty", "Novelty", "Exploration", "Exploitation", "Selected for Testing"])

    labeled_data = data.dropna(subset=target_columns)
    candidate_df = data[data[target_columns[0]].isnull()].copy()

    if candidate_df.empty:
        st.warning("No candidate samples for RF model evaluation.")
        return pd.DataFrame()

    train_inputs = labeled_data[input_columns]
    train_targets = labeled_data[target_columns].values
    candidate_inputs = candidate_df[input_columns]

    st.info("Using Bayesian Optimization with RF surrogate to score candidates.")

    utility_scores = multi_objective_bayesian_optimization(
        train_inputs=train_inputs,
        train_targets=train_targets,
        candidate_inputs=candidate_inputs,
        weights=weights,
        max_or_min=max_or_min,
        curiosity=curiosity,
        acquisition="UCB",
        strategy="weighted_sum",
        surrogate_model=rf_model,
        input_columns=input_columns
    )

    predictions, uncertainties = rf_model.predict_with_uncertainty(candidate_inputs)

    for i, col in enumerate(target_columns):
        candidate_df[col] = predictions[:, i]

    candidate_df["Utility"] = utility_scores if utility_scores is not None else 0
    candidate_df["Uncertainty"] = np.mean(uncertainties, axis=1)

    X_candidate_np = candidate_inputs.values
    X_labeled_np = labeled_data[input_columns].values
    novelty_scores = calculate_novelty(X_candidate_np, X_labeled_np)
    candidate_df["Novelty"] = novelty_scores

    candidate_df["Exploration"] = candidate_df["Uncertainty"] * (1 + curiosity)
    candidate_df["Exploitation"] = candidate_df[target_columns].mean(axis=1)

    candidate_df["Selected for Testing"] = False
    if not candidate_df.empty:
        max_utility_idx = candidate_df["Utility"].idxmax()
        candidate_df.loc[max_utility_idx, "Selected for Testing"] = True

    result_df = candidate_df.sort_values(by="Utility", ascending=False).reset_index(drop=True)
    return result_df
