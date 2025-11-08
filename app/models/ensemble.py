# app/ensemble.py

import numpy as np
import torch
import streamlit as st
import pandas as pd
from sklearn.preprocessing import RobustScaler

def ensemble_predictions(models, data, input_columns, target_columns, weights=None):

    if not models:
        st.error("No models provided for ensemble prediction.")
        return None, None
    
    # Get unlabeled data
    unlabeled_data = data[data[target_columns].isna().any(axis=1)]
    
    if unlabeled_data.empty:
        st.warning("No unlabeled samples available for prediction.")
        return None, None
    
    st.info(f"Creating ensemble predictions using {len(models)} models.")

    # Calculate model similarity matrix
    model_names = list(all_predictions.keys())
    num_models = len(model_names)
    similarity_matrix = np.zeros((num_models, num_models))

    for i in range(num_models):
        for j in range(i + 1, num_models):
            pred_i = all_predictions[model_names[i]].flatten()
            pred_j = all_predictions[model_names[j]].flatten()
            similarity = np.corrcoef(pred_i, pred_j)[0, 1]
            similarity_matrix[i, j] = similarity_matrix[j, i] = similarity

    # Penalize models with high similarity (>0.95)
    redundant_pairs = np.argwhere(similarity_matrix > 0.95)
    penalty_factor = 0.7  # Reduce weight by 30% for redundancy

    for i, j in redundant_pairs:
        weights[model_names[i]] *= penalty_factor
        weights[model_names[j]] *= penalty_factor

    # Normalize weights again
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    
    # Initialize storage for predictions and uncertainties from each model
    all_predictions = {}
    all_uncertainties = {}
    
    # If weights are not provided, use equal weights
    if weights is None:
        weights = {model_name: 1.0 / len(models) for model_name in models.keys()}
    else:
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
    
    # Process each model
    for model_name, (model, scaler_inputs, scaler_targets) in models.items():
        st.write(f"Processing model: {model_name}")
        
        try:
            # Transform inputs
            X_test = unlabeled_data[input_columns].values
            X_test_scaled = scaler_inputs.transform(X_test)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            
            # Get predictions and uncertainties
            model.eval()
            
            # Use Monte Carlo Dropout for uncertainty estimation
            num_samples = 20
            predictions_list = []
            
            # Enable dropout layers for Monte Carlo sampling
            model.train()
            
            # Collect multiple predictions with dropout enabled
            with torch.no_grad():
                for _ in range(num_samples):
                    preds = model(X_test_tensor).numpy()
                    predictions_list.append(preds)
            
            # Compute mean prediction and uncertainty
            predictions_stack = np.stack(predictions_list)
            mean_predictions = np.mean(predictions_stack, axis=0)
            uncertainties = np.std(predictions_stack, axis=0)
            
            # Transform back to original scale
            predictions = scaler_targets.inverse_transform(mean_predictions)
            
            # Store results
            all_predictions[model_name] = predictions
            all_uncertainties[model_name] = uncertainties
            
        except Exception as e:
            st.error(f"Error processing model {model_name}: {str(e)}")
            # Remove this model from weights
            if model_name in weights:
                del weights[model_name]
                # Re-normalize weights
                if weights:
                    total_weight = sum(weights.values())
                    weights = {k: v / total_weight for k, v in weights.items()}
    
    # If no valid models, return None
    if not all_predictions:
        st.error("No valid models for ensemble prediction.")
        return None, None
    
    # Create ensemble predictions through weighted averaging
    ensemble_predictions = np.zeros((len(unlabeled_data), len(target_columns)))
    ensemble_uncertainties = np.zeros((len(unlabeled_data), len(target_columns)))
    
    for model_name in all_predictions.keys():
        ensemble_predictions += weights[model_name] * all_predictions[model_name]
        ensemble_uncertainties += (weights[model_name] ** 2) * (all_uncertainties[model_name] ** 2)
    
    # Take square root of summed squared uncertainties (propagation of uncertainty)
    ensemble_uncertainties = np.sqrt(ensemble_uncertainties)
    
    # Create result DataFrame
    result_df = unlabeled_data.copy()
    
    # Add ensemble predictions to result
    for i, col in enumerate(target_columns):
        result_df[col] = ensemble_predictions[:, i]
    
    # Add ensemble uncertainty
    result_df["Uncertainty"] = np.mean(ensemble_uncertainties, axis=1)
    
    # Calculate model agreement metric (coefficient of variation across models)
    model_agreement = np.zeros((len(unlabeled_data), len(target_columns)))
    
    if len(all_predictions) > 1:
        # Stack all predictions for each model
        stacked_predictions = np.stack([all_predictions[model_name] for model_name in all_predictions.keys()])
        
        # Calculate coefficient of variation across models
        model_agreement = np.std(stacked_predictions, axis=0) / (np.mean(stacked_predictions, axis=0) + 1e-10)
        
    # Add model agreement to result
    result_df["Model_Agreement"] = 1.0 - np.mean(model_agreement, axis=1)
    
    # Determine model contribution for each sample
    model_contributions = {}
    for model_name in all_predictions.keys():
        # Calculate prediction error relative to ensemble
        pred_diff = np.abs(all_predictions[model_name] - ensemble_predictions)
        # Calculate contribution score (inverse of error)
        contribution = 1.0 / (pred_diff + 1e-10)
        # Normalize to [0, 1]
        contribution = contribution / np.max(contribution)
        # Average across all targets
        model_contributions[model_name] = np.mean(contribution, axis=1)
        
        # Add to DataFrame for inspection
        result_df[f"Contribution_{model_name}"] = model_contributions[model_name]
    
    # Gather ensemble model info
    ensemble_model_info = {
        "model_weights": weights,
        "model_contributions": model_contributions,
        "uncertainty_scaling": 1.0  # Default scaling factor
    }
    
    return result_df, ensemble_model_info


def weighted_uncertainty_ensemble(models, data, input_columns, target_columns, 
                                  acquisition_function="UCB", curiosity=0, weights=None):

    # Get both labeled and unlabeled data
    unlabeled_data = data[data[target_columns].isna().any(axis=1)]
    labeled_data = data.dropna(subset=target_columns)
    
    if unlabeled_data.empty:
        st.warning("No unlabeled samples available for prediction.")
        return None, None
    
    # If no initial weights, use equal weighting
    if weights is None:
        weights = {model_name: 1.0 / len(models) for model_name in models.keys()}
    
    # Calculate model performance on labeled data (if available)
    if not labeled_data.empty and len(labeled_data) >= 3:
        st.info("Calculating model weights based on validation performance...")
        
        # Calculate performance for each model
        model_errors = {}
        
        for model_name, (model, scaler_inputs, scaler_targets) in models.items():
            try:
                # Get validation inputs and targets
                X_val = labeled_data[input_columns].values
                y_val = labeled_data[target_columns].values
                
                # Scale inputs
                X_val_scaled = scaler_inputs.transform(X_val)
                X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
                
                # Get predictions
                model.eval()
                with torch.no_grad():
                    scaled_preds = model(X_val_tensor).numpy()
                
                # Transform back to original scale
                preds = scaler_targets.inverse_transform(scaled_preds)
                
                # Calculate mean squared error
                mse = np.mean((preds - y_val) ** 2)
                model_errors[model_name] = mse
                
            except Exception as e:
                st.warning(f"Error evaluating model {model_name}: {str(e)}")
                model_errors[model_name] = float('inf')
        
        # Convert errors to weights (lower error = higher weight)
        if all(err == float('inf') for err in model_errors.values()):
            # If all models failed, use equal weights
            weights = {model_name: 1.0 / len(models) for model_name in models.keys()}
        else:
            # Remove infinities
            valid_errors = {k: v for k, v in model_errors.items() if v < float('inf')}
            
            # Invert errors (lower error = higher weight)
            inv_errors = {k: 1.0 / (v + 1e-10) for k, v in valid_errors.items()}
            
            # Normalize to sum to 1.0
            total = sum(inv_errors.values())
            weights = {k: v / total for k, v in inv_errors.items()}
        
        # Show model weights
        weight_df = pd.DataFrame({
            'Model': list(weights.keys()),
            'Weight': list(weights.values())
        })
        st.write("Model weights based on validation performance:")
        st.dataframe(weight_df)
    
    # Get ensemble predictions
    result_df, ensemble_info = ensemble_predictions(
        models, data, input_columns, target_columns, weights
    )
    
    if result_df is None:
        return None, None
    
    # Apply acquisition function to balance exploration and exploitation
    if acquisition_function == "UCB":
        # Upper Confidence Bound
        result_df["Utility"] = result_df[target_columns].mean(axis=1) + curiosity * result_df["Uncertainty"]
    
    elif acquisition_function == "EI":
        # Expected Improvement
        if not labeled_data.empty:
            # Calculate the best observed value so far
            best_value = labeled_data[target_columns].mean(axis=1).max()
            
            # Calculate expected improvement
            mean = result_df[target_columns].mean(axis=1)
            sigma = result_df["Uncertainty"]
            z = (mean - best_value) / (sigma + 1e-10)
            
            # Calculate EI using the standard normal CDF and PDF
            cdf = 0.5 * (1 + np.tanh(z / np.sqrt(2)))
            pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
            
            result_df["Utility"] = (mean - best_value) * cdf + sigma * pdf * curiosity
        else:
            # Fall back to UCB if no labeled data
            result_df["Utility"] = result_df[target_columns].mean(axis=1) + curiosity * result_df["Uncertainty"]
    
    elif acquisition_function == "PI":
        # Probability of Improvement
        if not labeled_data.empty:
            # Calculate the best observed value so far
            best_value = labeled_data[target_columns].mean(axis=1).max()
            
            # Calculate probability of improvement
            mean = result_df[target_columns].mean(axis=1)
            sigma = result_df["Uncertainty"]
            z = (mean - best_value) / (sigma + 1e-10)
            
            # Calculate PI using the standard normal CDF
            result_df["Utility"] = 0.5 * (1 + np.tanh(z / np.sqrt(2)))
        else:
            # Fall back to mean prediction if no labeled data
            result_df["Utility"] = result_df[target_columns].mean(axis=1)
    
    else:
        # Default to mean prediction
        result_df["Utility"] = result_df[target_columns].mean(axis=1)
    
    # Calculate novelty
    from app.utils import calculate_novelty
    labeled_inputs = labeled_data[input_columns].values
    unlabeled_inputs = unlabeled_data[input_columns].values

    # We need to scale the inputs before calculating novelty
    # Assuming the first model's scaler is representative
    scaler_inputs = list(models.values())[0][1]
    labeled_inputs_scaled = scaler_inputs.transform(labeled_inputs)
    unlabeled_inputs_scaled = scaler_inputs.transform(unlabeled_inputs)

    novelty_scores = calculate_novelty(unlabeled_inputs_scaled, labeled_inputs_scaled)
    result_df["Novelty"] = novelty_scores

    # Add exploration and exploitation columns
    result_df["Exploration"] = result_df["Uncertainty"] * result_df["Novelty"]
    result_df["Exploitation"] = 1.0 - result_df["Uncertainty"]

    # Sort by utility and mark highest utility sample
    result_df = result_df.sort_values("Utility", ascending=False)
    result_df["Selected_for_Testing"] = False
    result_df.iloc[0, result_df.columns.get_loc("Selected_for_Testing")] = True
    
    # Reset index for better display
    result_df = result_df.reset_index(drop=True)
    
    return result_df, ensemble_info