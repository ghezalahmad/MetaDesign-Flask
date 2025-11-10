# app/models/ensemble.py

import numpy as np
import torch
import streamlit as st
import pandas as pd
from sklearn.preprocessing import RobustScaler

from app.models.bayesian_optimizer import multi_objective_bayesian_optimization
from app.utils.utils import calculate_novelty


class EnsembleSurrogate:
    """
    A wrapper for an ensemble of models to make it compatible with the
    BayesianOptimizer as a surrogate model.
    """
    def __init__(self, models, weights, input_columns, target_columns):
        self.models = models
        self.weights = weights
        self.input_columns = input_columns
        self.target_columns = target_columns
        self.is_trained = True  # The ensemble components are assumed to be trained

    def predict_with_uncertainty(self, X, input_columns=None, num_samples=None):
        """
        Predicts mean and uncertainty for the given input using the weighted ensemble.

        Args:
            X (pd.DataFrame or np.ndarray): Input data for prediction.
            input_columns (list[str], optional): List of input columns. Defaults to None.
            num_samples (int, optional): Number of MC samples for dropout. Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - ensemble_preds (np.ndarray): The mean predictions from the ensemble.
                - ensemble_stds (np.ndarray): The standard deviation (uncertainty) of the predictions.
        """
        if not self.models:
            raise ValueError("No models in the ensemble surrogate.")

        if isinstance(X, pd.DataFrame):
            X_np = X[self.input_columns].values
        else:
            X_np = X

        all_predictions = {}
        all_uncertainties = {}

        for model_name, (model, scaler_inputs, scaler_targets) in self.models.items():
            X_test_scaled = scaler_inputs.transform(X_np)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

            num_mc_samples = 20
            predictions_list = []
            model.train()  # Enable dropout for MC sampling
            with torch.no_grad():
                for _ in range(num_mc_samples):
                    preds = model(X_test_tensor).numpy()
                    predictions_list.append(preds)

            predictions_stack = np.stack(predictions_list)
            mean_predictions_scaled = np.mean(predictions_stack, axis=0)
            uncertainties_scaled = np.std(predictions_stack, axis=0)

            predictions_orig_scale = scaler_targets.inverse_transform(mean_predictions_scaled)
            
            if hasattr(scaler_targets, 'scale_') and scaler_targets.scale_ is not None:
                uncertainties_orig_scale = uncertainties_scaled * scaler_targets.scale_
            else:
                uncertainties_orig_scale = uncertainties_scaled

            all_predictions[model_name] = predictions_orig_scale
            all_uncertainties[model_name] = uncertainties_orig_scale

        # Ensure weights are normalized
        total_weight = sum(self.weights.values())
        if total_weight == 0:
            normalized_weights = {k: 1.0 / len(self.weights) for k in self.weights}
        else:
            normalized_weights = {k: v / total_weight for k, v in self.weights.items()}
        
        # Get shape from the first model's prediction
        first_pred = next(iter(all_predictions.values()))
        ensemble_preds = np.zeros_like(first_pred)
        ensemble_uncertainties_sq = np.zeros_like(first_pred)

        for model_name, weight in normalized_weights.items():
            if model_name in all_predictions:
                ensemble_preds += weight * all_predictions[model_name]
                # Variance is std^2, so we square the uncertainty
                ensemble_uncertainties_sq += (weight**2) * (all_uncertainties[model_name]**2)

        ensemble_stds = np.sqrt(ensemble_uncertainties_sq)

        return ensemble_preds, ensemble_stds


def weighted_uncertainty_ensemble(models, data, input_columns, target_columns,
                                  acquisition_function="UCB", curiosity=0, weights=None,
                                  max_or_min_objectives=None):
    """
    Scores unlabeled samples using an ensemble of models and Bayesian Optimization.
    """
    unlabeled_data = data[data[target_columns].isna().any(axis=1)]
    labeled_data = data.dropna(subset=target_columns)
    
    if unlabeled_data.empty:
        st.warning("No unlabeled samples available for prediction.")
        return None, None
    
    if weights is None:
        weights = {model_name: 1.0 / len(models) for model_name in models.keys()}
    
    if not labeled_data.empty and len(labeled_data) >= 3:
        st.info("Calculating model weights based on validation performance...")
        model_errors = {}
        for model_name, (model, scaler_inputs, scaler_targets) in models.items():
            try:
                X_val = labeled_data[input_columns].values
                y_val = labeled_data[target_columns].values
                X_val_scaled = scaler_inputs.transform(X_val)
                X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
                
                model.eval()
                with torch.no_grad():
                    scaled_preds = model(X_val_tensor).numpy()
                
                preds = scaler_targets.inverse_transform(scaled_preds)
                mse = np.mean((preds - y_val) ** 2)
                model_errors[model_name] = mse
            except Exception as e:
                st.warning(f"Error evaluating model {model_name}: {str(e)}")
                model_errors[model_name] = float('inf')
        
        valid_errors = {k: v for k, v in model_errors.items() if v < float('inf')}
        if valid_errors:
            inv_errors = {k: 1.0 / (v + 1e-10) for k, v in valid_errors.items()}
            total = sum(inv_errors.values())
            weights = {k: v / total for k, v in inv_errors.items()}
        
        weight_df = pd.DataFrame({'Model': list(weights.keys()), 'Weight': list(weights.values())})
        st.write("Model weights based on validation performance:")
        st.dataframe(weight_df)

    st.info("Using Bayesian Optimization with Ensemble Surrogate to find best candidates.")

    ensemble_surrogate = EnsembleSurrogate(
        models=models,
        weights=weights,
        input_columns=input_columns,
        target_columns=target_columns
    )

    train_inputs = labeled_data[input_columns]
    train_targets = labeled_data[target_columns].values
    candidate_inputs = unlabeled_data[input_columns]

    if max_or_min_objectives is None:
        max_or_min_objectives = ['max'] * len(target_columns)

    # Use MOBO to get utility scores
    utility_scores = multi_objective_bayesian_optimization(
        train_inputs=train_inputs,
        train_targets=train_targets,
        candidate_inputs=candidate_inputs,
        weights=np.array([1.0] * len(target_columns)), # Let MOBO handle scalarization
        max_or_min=max_or_min_objectives,
        curiosity=curiosity,
        acquisition=acquisition_function,
        strategy="weighted_sum",
        surrogate_model=ensemble_surrogate,
        input_columns=input_columns
    )

    # Get predictions and uncertainties for the result dataframe
    ensemble_preds, ensemble_stds = ensemble_surrogate.predict_with_uncertainty(candidate_inputs)

    result_df = unlabeled_data.copy()
    for i, col in enumerate(target_columns):
        result_df[col] = ensemble_preds[:, i]
    
    result_df["Uncertainty"] = np.mean(ensemble_stds, axis=1)
    result_df["Utility"] = utility_scores if utility_scores is not None else 0

    # Calculate novelty
    if not labeled_data.empty:
        labeled_inputs = labeled_data[input_columns].values
        unlabeled_inputs = unlabeled_data[input_columns].values

        # Use a representative scaler
        scaler_inputs = list(models.values())[0][1]
        labeled_inputs_scaled = scaler_inputs.transform(labeled_inputs)
        unlabeled_inputs_scaled = scaler_inputs.transform(unlabeled_inputs)

        novelty_scores = calculate_novelty(unlabeled_inputs_scaled, labeled_inputs_scaled)
        result_df["Novelty"] = novelty_scores
    else:
        result_df["Novelty"] = 1.0 # Max novelty if no labeled data

    result_df["Exploration"] = result_df["Uncertainty"] * result_df["Novelty"]
    result_df["Exploitation"] = 1.0 - result_df["Uncertainty"] # This is a simple metric

    result_df = result_df.sort_values("Utility", ascending=False)
    result_df["Selected_for_Testing"] = False
    if not result_df.empty:
        result_df.iloc[0, result_df.columns.get_loc("Selected_for_Testing")] = True
    
    result_df = result_df.reset_index(drop=True)
    
    ensemble_info = {"model_weights": weights}

    return result_df, ensemble_info
