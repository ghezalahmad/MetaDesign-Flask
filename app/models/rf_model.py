import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import streamlit as st

class RFModel:
    def __init__(self, n_estimators=100, random_state=42, **kwargs):
        """
        Random Forest Regressor model.

        Args:
            n_estimators (int): The number of trees in the forest.
            random_state (int): Controls both the randomness of the bootstrapping of the samples used
                                when building trees and the sampling of the features to consider when
                                looking for the best split at each node.
            **kwargs: Additional keyword arguments for RandomForestRegressor.
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            **kwargs
        )
        self.scaler_x = None
        self.scaler_y = None
        self.is_trained = False

    def train(self, data: pd.DataFrame, input_columns: list[str], target_columns: list[str], test_size=0.2, perform_grid_search=False):
        """
        Trains the Random Forest model.

        Args:
            data (pd.DataFrame): The training data.
            input_columns (list[str]): List of input feature column names.
            target_columns (list[str]): List of target variable column names.
            test_size (float): Proportion of the dataset to include in the test split for validation during grid search.
            perform_grid_search (bool): Whether to perform GridSearchCV to find best hyperparameters.
        """
        st.info(f"Starting Random Forest training with {len(data)} samples.")

        labeled_data = data.dropna(subset=target_columns).copy()
        if len(labeled_data) < 5: # Need enough samples for train/test split if using grid search
            st.error("Not enough labeled samples for Random Forest training (need at least 5).")
            return self, None, None

        X = labeled_data[input_columns]
        y = labeled_data[target_columns]

        self.scaler_x = RobustScaler()
        self.scaler_y = RobustScaler()

        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)

        # If only one target, y_scaled might be (n_samples,). Reshape to (n_samples, 1) if so.
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)

        # If multiple targets, train one model per target for simplicity,
        # though RandomForestRegressor handles multi-output directly.
        # For now, let's assume direct multi-output if y has multiple columns.
        # If y_scaled is (n_samples, 1) and RandomForestRegressor expects (n_samples,), ravel it.
        if y_scaled.shape[1] == 1:
            y_train_final = y_scaled.ravel()
        else:
            y_train_final = y_scaled # For multi-output

        if perform_grid_search and len(labeled_data) >= 10:
            st.info("Performing GridSearchCV for Random Forest...")
            X_train_gs, X_val_gs, y_train_gs, y_val_gs = train_test_split(X_scaled, y_train_final, test_size=test_size, random_state=42)

            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                                       param_grid=param_grid, cv=3, n_jobs=-1, verbose=0, scoring='neg_mean_squared_error')
            grid_search.fit(X_train_gs, y_train_gs)
            self.model = grid_search.best_estimator_
            st.success(f"GridSearchCV complete. Best params: {grid_search.best_params_}")
        else:
            if perform_grid_search:
                st.warning("Not enough samples for GridSearchCV, using default RF parameters.")
            self.model.fit(X_scaled, y_train_final)

        self.is_trained = True
        st.success("Random Forest training complete.")
        return self, self.scaler_x, self.scaler_y

    def predict_with_uncertainty(self, X_unlabeled: pd.DataFrame, input_columns: list[str]):
        """
        Makes predictions and estimates uncertainty.
        Uncertainty is estimated as the variance of predictions from individual trees in the forest.

        Args:
            X_unlabeled (pd.DataFrame): DataFrame with unlabeled input features.
            input_columns (list[str]): List of input feature column names.

        Returns:
            tuple: (predictions_original_scale, uncertainties, Optional[all_tree_preds_original_scale])
                   predictions_original_scale (np.ndarray): Predictions in the original target scale.
                   uncertainties (np.ndarray): Uncertainty estimates (std dev based).
                   all_tree_preds_original_scale (np.ndarray, optional): Predictions from all trees in original scale. Shape (n_estimators, n_samples, n_targets)
        """
        if not self.is_trained or self.scaler_x is None or self.scaler_y is None:
            raise RuntimeError("Model is not trained yet or scalers are missing.")

        X_unlabeled_processed = X_unlabeled[input_columns]
        X_scaled = self.scaler_x.transform(X_unlabeled_processed)

        # Get predictions from each tree
        tree_predictions = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])

        # tree_predictions shape: (n_estimators, n_samples) if single target
        # or (n_estimators, n_samples, n_targets) if multi-target

        if self.model.n_outputs_ == 1: # Single target
            # Mean prediction across trees
            mean_predictions_scaled = np.mean(tree_predictions, axis=0)
            # Variance of predictions across trees (uncertainty)
            variance_scaled = np.var(tree_predictions, axis=0)

            predictions_original_scale = self.scaler_y.inverse_transform(mean_predictions_scaled.reshape(-1, 1))
            # Scale the variance. Variance scales by square of scaler's scale factor.
            # This is an approximation. For a single target, scaler_y.scale_ is a scalar.
            variance_original_scale = variance_scaled * (self.scaler_y.scale_**2)
            std_dev_original_scale = np.sqrt(variance_original_scale)
            uncertainties = std_dev_original_scale.reshape(-1, 1) # Return std_dev

        else: # Multi-target
            # Mean prediction across trees for each target
            mean_predictions_scaled = np.mean(tree_predictions, axis=0) # Shape (n_samples, n_targets)
            # Variance of predictions across trees for each target
            variance_scaled = np.var(tree_predictions, axis=0) # Shape (n_samples, n_targets)

            predictions_original_scale = self.scaler_y.inverse_transform(mean_predictions_scaled)
            # Scale the variance for each target. scaler_y.scale_ is an array for multi-target.
            variance_original_scale = variance_scaled * (self.scaler_y.scale_**2)
            std_dev_original_scale = np.sqrt(variance_original_scale)
            # We need a single uncertainty value per sample, so average std_dev across targets
            uncertainties = np.mean(std_dev_original_scale, axis=1).reshape(-1, 1) # Return mean std_dev

        # Inverse transform all tree predictions for posterior samples
        # tree_predictions is (n_estimators, n_samples) or (n_estimators, n_samples, n_targets_rf)
        all_tree_preds_original_scale_list = []
        if self.model.n_outputs_ == 1: # tree_predictions is (n_estimators, n_samples)
            for i in range(tree_predictions.shape[0]): # Iterate over estimators
                # tree_predictions[i, :] is (n_samples,). Reshape for scaler.
                tree_pred_orig_scale = self.scaler_y.inverse_transform(tree_predictions[i, :].reshape(-1,1))
                all_tree_preds_original_scale_list.append(tree_pred_orig_scale)
        else: # tree_predictions is (n_estimators, n_samples, n_targets_rf)
             for i in range(tree_predictions.shape[0]): # Iterate over estimators
                # tree_predictions[i, :, :] is (n_samples, n_targets_rf)
                tree_pred_orig_scale = self.scaler_y.inverse_transform(tree_predictions[i, :, :])
                all_tree_preds_original_scale_list.append(tree_pred_orig_scale)

        all_tree_preds_original_scale = np.stack(all_tree_preds_original_scale_list, axis=0)
        # Final shape: (n_estimators, n_samples, n_actual_targets_after_scaler)

        return predictions_original_scale, uncertainties, all_tree_preds_original_scale

def train_rf_model(data: pd.DataFrame, input_columns: list[str], target_columns: list[str],
                   n_estimators=100, random_state=42, perform_grid_search=False, **kwargs):
    """
    Helper function to initialize and train an RFModel.
    """
    model = RFModel(n_estimators=n_estimators, random_state=random_state, **kwargs)
    model, scaler_x, scaler_y = model.train(data, input_columns, target_columns, perform_grid_search=perform_grid_search)
    return model, scaler_x, scaler_y

def evaluate_rf_model(
    rf_model: RFModel,
    data: pd.DataFrame,
    input_columns: list[str],
    target_columns: list[str],
    curiosity: float,
    weights: np.ndarray,
    max_or_min: list[str],
    acquisition="UCB" # Allow passing acquisition function, though utils.calculate_utility might also select one
):
    """
    Evaluates the trained RF model on unlabeled data and calculates utility.
    """
    if not rf_model.is_trained:
        st.error("Random Forest model is not trained. Please train the model first.")
        return None

    unlabeled_data = data[data[target_columns].isna().any(axis=1)].copy()
    labeled_data = data.dropna(subset=target_columns)

    if unlabeled_data.empty:
        st.warning("No unlabeled samples available for RF model evaluation.")
        return None

    st.info(f"Evaluating RF model with {len(labeled_data)} labeled samples and {len(unlabeled_data)} unlabeled samples.")

    # Unpack all three return values, ignore the third if not used by this function
    predictions_orig_scale, uncertainties, _ = rf_model.predict_with_uncertainty(unlabeled_data, input_columns)

    # Ensure predictions are non-negative, aligning with expected output characteristics.
    predictions_final = np.maximum(predictions_orig_scale, 0)

    # Calculate novelty (re-using existing utility from app.utils)
    # Need to import calculate_novelty and calculate_utility from app.utils
    from app.utils import calculate_novelty, calculate_utility, select_acquisition_function

    # Novelty calculation
    if rf_model.scaler_x:
        # Use the scaler from the trained RF model
        scaled_labeled_inputs = rf_model.scaler_x.transform(labeled_data[input_columns])
        scaled_unlabeled_inputs = rf_model.scaler_x.transform(unlabeled_data[input_columns])
        novelty_scores = calculate_novelty(scaled_unlabeled_inputs, scaled_labeled_inputs)
    else: # Should not happen if model is trained
        novelty_scores = np.zeros(len(unlabeled_data))


    if acquisition is None: # If not passed, select dynamically
        acquisition = select_acquisition_function(curiosity, len(labeled_data))
    st.info(f"Using acquisition function: {acquisition} for RF model.")

    # Ensure weights and max_or_min are properly formatted for calculate_utility
    if not isinstance(weights, np.ndarray):
        weights = np.array(weights)
    if max_or_min is None or not isinstance(max_or_min, list) or len(max_or_min) != len(target_columns):
        max_or_min = ['max'] * len(target_columns)

    # Define min_strength_threshold for penalization within calculate_utility if necessary,
    # or rely on calculate_utility's own internal logic if it has one.
    # For consistency, we'll use the same approach as other models: penalize utilities for predictions < 10.
    min_strength_threshold = 10

    # Note: calculate_utility from app.utils does not internally clamp predictions to min_strength_threshold
    # but uses them as is for the positive part of utility.
    # It can apply threshold factors if thresholds are passed.
    # We will apply a penalty after if predictions are below min_strength_threshold.

    utility_scores = calculate_utility(
        predictions_final, # Already clamped at 0
        uncertainties,
        novelty_scores,
        curiosity,
        weights,
        max_or_min,
        acquisition=acquisition
    )

    # Penalize utility for predictions that fall below the min_strength_threshold
    # This makes it consistent with how MAML (after recent changes) and others handle it.
    invalid_rows = np.any(predictions_final < min_strength_threshold, axis=1)
    if np.any(invalid_rows):
        utility_scores[invalid_rows] = -np.inf # Or a very small number

    result_df = unlabeled_data.copy()
    for i, col in enumerate(target_columns):
        result_df[col] = predictions_final[:, i]

    result_df["Utility"] = utility_scores.flatten()
    # Ensure uncertainties are positive and have correct shape for DataFrame
    result_df["Uncertainty"] = np.clip(uncertainties, 1e-9, None).flatten()
    result_df["Novelty"] = novelty_scores.flatten()
    result_df["Exploration"] = result_df["Uncertainty"] * result_df["Novelty"]
    result_df["Exploitation"] = 1.0 - result_df["Uncertainty"] # Simplistic exploitation

    result_df = result_df.sort_values(by="Utility", ascending=False)
    result_df["Selected for Testing"] = False
    if not result_df.empty:
        result_df.iloc[0, result_df.columns.get_loc("Selected for Testing")] = True

    result_df.reset_index(drop=True, inplace=True)

    # Reorder columns (optional, for consistency with other models' outputs)
    # This part can be adapted from existing evaluate_* functions
    columns_to_front = ["Idx_Sample"] if "Idx_Sample" in result_df.columns else []
    metrics_columns = ["Utility", "Exploration", "Exploitation", "Novelty", "Uncertainty"]
    remaining_columns = [col for col in result_df.columns if col not in columns_to_front + metrics_columns + target_columns + ["Selected for Testing"]]
    new_column_order = columns_to_front + metrics_columns + target_columns + remaining_columns + ["Selected for Testing"]
    new_column_order = [col for col in new_column_order if col in result_df.columns] # Ensure all columns exist
    result_df = result_df[new_column_order]

    return result_df
