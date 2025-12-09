import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import RobustScaler
import streamlit as st # Keeping this in case you use it elsewhere

# Assuming this path is correct for where your Multi-Objective BO logic lives
from app.models.bayesian_optimizer import multi_objective_bayesian_optimization
# Assuming this path is correct for where the novelty calculation lives
from app.utils.utils import calculate_novelty


class EnsembleSurrogate:
    """
    An Advanced Ensemble wrapper compatible with Bayesian Optimization.
    It combines predictions from multiple models (NN/GP/RF) and uses model
    disagreement for a robust exploration component.
    """
    def __init__(self, models, weights, input_columns, target_columns):
        self.models = models
        self.weights = weights
        self.input_columns = input_columns
        self.target_columns = target_columns
        # Mark the ensemble as trained since it wraps already trained models
        self.is_trained = True 

    def _get_single_model_prediction(self, model_data, X_np):
        """Internal helper to standardize prediction and uncertainty retrieval."""
        model, scaler_inputs, scaler_targets = model_data

        if hasattr(model, 'predict_with_uncertainty'):
            # Handles DKL, RF (via custom wrapper), or other models with this method
            # Models now return 3 values: (mean, std, posterior_samples)
            result = model.predict_with_uncertainty(X_np, self.input_columns)
            if len(result) == 3:
                predictions_orig_scale, uncertainties_orig_scale, _ = result
            else:
                predictions_orig_scale, uncertainties_orig_scale = result
        
        elif isinstance(model, torch.nn.Module):
            # Handles PyTorch models (like PINN/MAML) using MC Dropout
            X_test_scaled = scaler_inputs.transform(X_np)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

            num_mc_samples = 20
            predictions_list = []
            model.train()  # Enable dropout for MC sampling
            with torch.no_grad():
                for _ in range(num_mc_samples):
                    preds_scaled = model(X_test_tensor).numpy()
                    predictions_list.append(preds_scaled)

            predictions_stack = np.stack(predictions_list)
            mean_predictions_scaled = np.mean(predictions_stack, axis=0)
            uncertainties_scaled = np.std(predictions_stack, axis=0)

            predictions_orig_scale = scaler_targets.inverse_transform(mean_predictions_scaled)
            
            # Rescale uncertainty back to original space using target scaler's scale factor
            if hasattr(scaler_targets, 'scale_') and scaler_targets.scale_ is not None:
                # Assuming scaler_targets is a RobustScaler or similar with a scale_ attribute
                # Check if scale_ is 1D or 2D and adjust if necessary
                target_scales = scaler_targets.scale_
                if target_scales.ndim == 1:
                    target_scales = target_scales.reshape(1, -1)
                    
                uncertainties_orig_scale = uncertainties_scaled * target_scales
            else:
                # Fallback if target scaler's scale is not easily available
                uncertainties_orig_scale = uncertainties_scaled

        else:
            # Fallback for models that only have 'predict' (e.g., simple scikit-learn regressor)
            predictions_orig_scale = model.predict(X_np)
            # Assign zero uncertainty, which is low and discourages exploitation from this model
            uncertainties_orig_scale = np.zeros_like(predictions_orig_scale) 

        return predictions_orig_scale, uncertainties_orig_scale

    def predict_with_uncertainty(self, X: pd.DataFrame | np.ndarray, input_columns=None, num_samples=None):
        """
        Predicts mean and uncertainty using the ensemble, prioritizing model disagreement.

        The mean is a weighted average of individual predictions.
        The uncertainty (STD) is derived from the disagreement (variance) between models.
        
        Returns: 
            ensemble_preds (np.ndarray): Shape (N_candidates, N_targets) - Weighted mean predictions.
            ensemble_stds_broadcasted (np.ndarray): Shape (N_candidates, N_targets) - Uncertainty based on model disagreement.
        """
        if not self.models:
            raise ValueError("No models in the ensemble surrogate.")

        if isinstance(X, pd.DataFrame):
            X_np = X[self.input_columns].values
        else:
            X_np = X

        N_targets = len(self.target_columns)
        N_candidates = X_np.shape[0]
        N_models = len(self.models)
        
        # Array to hold all predictions: (N_candidates, N_targets, N_models)
        all_predictions_np = np.empty((N_candidates, N_targets, N_models))

        # 1. Gather Predictions from all models
        model_names = list(self.models.keys())
        for i, (model_name, model_data) in enumerate(self.models.items()):
            # predictions shape: (N_candidates, N_targets)
            predictions, _ = self._get_single_model_prediction(model_data, X_np) 
            all_predictions_np[:, :, i] = predictions

        # 2. Normalize Weights
        total_weight = sum(self.weights.values())
        if total_weight == 0:
            # Fallback to equal weighting if all weights sum to zero
            normalized_weights = {k: 1.0 / N_models for k in self.weights}
        else:
            normalized_weights = {k: v / total_weight for k, v in self.weights.items()}

        # 3. Calculate Ensemble Mean (Exploitation)
        ensemble_preds = np.zeros((N_candidates, N_targets))
        for i, model_name in enumerate(model_names):
            weight = normalized_weights.get(model_name, 0)
            ensemble_preds += weight * all_predictions_np[:, :, i]

        # 4. Calculate Ensemble Disagreement (Exploration)
        # Disagreement = Variance of predictions across all models for each candidate/target
        model_variance = np.var(all_predictions_np, axis=2) # Shape: (N_candidates, N_targets)
        
        # Calculate the Root Mean Square of the target variances for a single STD value
        # This provides a single, scalar measure of uncertainty (disagreement) per input point
        ensemble_stds = np.sqrt(np.mean(model_variance, axis=1)).reshape(-1, 1)

        # Broadcast the single STD value per candidate back to match the target shape 
        ensemble_stds_broadcasted = np.tile(ensemble_stds, (1, N_targets))

        return ensemble_preds, ensemble_stds_broadcasted


def weighted_uncertainty_ensemble(models, data, input_columns, target_columns,
                                  acquisition_function="UCB", curiosity=0, weights=None,
                                  max_or_min_objectives=None):
    """
    Scores unlabeled samples using an ensemble of models and Multi-Objective Bayesian Optimization (MOBO).
    """
    unlabeled_data = data[data[target_columns].isna().any(axis=1)]
    labeled_data = data.dropna(subset=target_columns)
    
    if unlabeled_data.empty:
        # Returning an empty result structure is better than None for stability
        return pd.DataFrame(), {}
    
    if weights is None:
        # Simple equal weighting if no weights are provided
        weights = {model_name: 1.0 for model_name in models.keys()} # Use 1.0 initially, normalization happens in class
    
    # --- Start: Weight Adjustment Logic based on validation error ---
    if not labeled_data.empty and len(labeled_data) >= 3:
        model_errors = {}
        for model_name, (model, scaler_inputs, scaler_targets) in models.items():
            try:
                X_val = labeled_data[input_columns].values
                y_val = labeled_data[target_columns].values
                
                # Use the helper function's logic to get predictions to be consistent
                temp_ensemble = EnsembleSurrogate(models, weights, input_columns, target_columns)
                preds, _ = temp_ensemble._get_single_model_prediction(
                    (model, scaler_inputs, scaler_targets), X_val
                )

                mse = np.mean((preds - y_val) ** 2)
                model_errors[model_name] = mse
            except Exception as e:
                # Log error if evaluation fails for a specific model
                print(f"Error evaluating model {model_name} for weighting: {str(e)}") 
                model_errors[model_name] = float('inf')
        
        valid_errors = {k: v for k, v in model_errors.items() if v < float('inf')}
        if valid_errors:
            # Calculate weights inversely proportional to MSE
            inv_errors = {k: 1.0 / (v + 1e-10) for k, v in valid_errors.items()}
            
            # Normalize weights to sum to 1.0
            total = sum(inv_errors.values())
            weights = {k: v / total for k, v in inv_errors.items()}
    # --- End: Weight Adjustment Logic ---


    # Initialize the Ensemble Surrogate
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

    # Initialize the Ensemble Surrogate
    ensemble_surrogate = EnsembleSurrogate(
        models=models,
        weights=weights,
        input_columns=input_columns,
        target_columns=target_columns
    )

    # Get ensemble predictions and uncertainties
    ensemble_preds, ensemble_stds_broadcasted = ensemble_surrogate.predict_with_uncertainty(candidate_inputs)
    ensemble_stds = ensemble_stds_broadcasted[:, 0]

    result_df = unlabeled_data.copy()
    for i, col in enumerate(target_columns):
        result_df[col] = ensemble_preds[:, i]
        result_df[f"Uncertainty ({col})"] = ensemble_stds_broadcasted[:, i] if ensemble_stds_broadcasted.shape[1] > 1 else ensemble_stds

    # WEBSLAMD-EXACT UTILITY CALCULATION
    labels_mean = labeled_data[target_columns].mean(skipna=True)
    labels_std = labeled_data[target_columns].std(skipna=True).replace(0, 1)
    
    n_samples = len(ensemble_preds)
    n_targets = len(target_columns)
    preds_norm = np.zeros((n_samples, n_targets), dtype=float)
    unc_norm = np.zeros((n_samples, n_targets), dtype=float)
    
    for i, col in enumerate(target_columns):
        mean_val = labels_mean.iloc[i]
        std_val = labels_std.iloc[i]
        preds_norm[:, i] = (ensemble_preds[:, i] - mean_val) / std_val
        if max_or_min_objectives[i].lower() == "min":
            preds_norm[:, i] *= -1
        preds_norm[:, i] *= 1.0  # uniform weight
        unc_norm[:, i] = (ensemble_stds_broadcasted[:, i] if ensemble_stds_broadcasted.shape[1] > 1 else ensemble_stds) / std_val
    
    utility_scores = preds_norm.sum(axis=1) + curiosity * unc_norm.sum(axis=1)
    result_df["Utility"] = utility_scores
    result_df["Uncertainty"] = ensemble_stds

    # Calculate novelty
    if not labeled_data.empty:
        labeled_inputs = labeled_data[input_columns].values
        unlabeled_inputs = unlabeled_data[input_columns].values
        novelty_scores = calculate_novelty(unlabeled_inputs, labeled_inputs)
        result_df["Novelty"] = novelty_scores
    else:
        result_df["Novelty"] = 1.0

    result_df = result_df.sort_values("Utility", ascending=False)
    result_df["Selected_for_Testing"] = False
    if not result_df.empty:
        result_df.iloc[0, result_df.columns.get_loc("Selected_for_Testing")] = True
    
    result_df = result_df.reset_index(drop=True)
    
    ensemble_info = {"model_weights": weights}

    return result_df, ensemble_info