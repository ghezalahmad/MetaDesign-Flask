import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import streamlit as st
from app.utils import (
    calculate_utility, calculate_novelty, # calculate_uncertainty removed
    initialize_scheduler, balance_exploration_exploitation
)
# calculate_uncertainty_ensemble will be imported from app.models where needed

class ReptileModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256, num_layers=3, dropout_rate=0.3):
        super(ReptileModel, self).__init__()
        
        # Add layer normalization for better training stability
        self.input_norm = nn.LayerNorm(input_size)
        
        # Use wider hidden layers like in GPR's RBF kernel approximation
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Linear(input_size, hidden_size * 2))  # Wider first layer
        self.layer_norms.append(nn.LayerNorm(hidden_size * 2))
        self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(1, num_layers - 1):
            in_size = hidden_size * 2 if i == 1 else hidden_size
            self.layers.append(nn.Linear(in_size, hidden_size))
            self.layer_norms.append(nn.LayerNorm(hidden_size))
            self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Final layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)  # Xavier works better for regression
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.input_norm(x)
        
        # Forward through layers with skips like in ResNet
        h = x
        for i, (layer, norm, dropout) in enumerate(zip(self.layers, self.layer_norms, self.dropouts)):
            z = layer(h)
            z = norm(z)
            z = torch.relu(z)
            z = dropout(z)
            
            # Use residual connections for better gradient flow when shapes match
            if i > 0 and h.shape == z.shape:
                h = h + z
            else:
                h = z
        
        # Output prediction - no activation to allow full range
        return self.output_layer(h)

    def predict_with_uncertainty(self, X_df_original_scale: pd.DataFrame, input_columns: list[str], num_perturbations=50, noise_scale=0.1, dropout_rate=0.3):
        """
        Makes predictions and estimates uncertainty.
        Assumes the model has been trained and scalers (self.scaler_x, self.scaler_y) are set.

        Args:
            X_df_original_scale (pd.DataFrame): DataFrame with input features in their original scale.
            input_columns (list[str]): List of input feature column names to use.
            num_perturbations (int): Number of Monte Carlo samples for uncertainty estimation.
            noise_scale (float): Scale of Gaussian noise for input perturbation for uncertainty.
            dropout_rate (float): Dropout rate to use during MC sampling for uncertainty.

        Returns:
            tuple: (mean_predictions_original_scale, std_dev_original_scale_per_sample, Optional[mc_samples_original_scale])
                   mean_predictions_original_scale (np.ndarray): Predictions in the original target scale.
                   std_dev_original_scale_per_sample (np.ndarray): Standard deviation of predictions (per sample) in original target scale.
                   mc_samples_original_scale (np.ndarray, optional): MC samples in original target scale.
        """
        if not getattr(self, 'is_trained', False) or self.scaler_x is None or self.scaler_y is None:
            raise RuntimeError("Model is not trained yet or scalers are missing. Call reptile_train first.")

        self.eval() # Ensure model is in eval mode for mean prediction
        X_processed = X_df_original_scale[input_columns]
        X_scaled_np = self.scaler_x.transform(X_processed)
        X_tensor = torch.tensor(X_scaled_np, dtype=torch.float32)

        with torch.no_grad():
            mean_predictions_scaled = self(X_tensor).numpy()

        mean_predictions_original_scale = self.scaler_y.inverse_transform(mean_predictions_scaled)

        from app.models import calculate_uncertainty_ensemble
        variance_scaled_per_sample, mc_samples_scaled_torch = calculate_uncertainty_ensemble(
            model=self, X=X_tensor,
            num_samples=num_perturbations,
            dropout_rate=dropout_rate,
            use_input_perturbation=True, # Reptile model was using this
            noise_scale=noise_scale
        )
        if variance_scaled_per_sample.ndim == 1:
            variance_scaled_per_sample = variance_scaled_per_sample.reshape(-1, 1)

        mc_samples_original_scale = None
        if mc_samples_scaled_torch is not None:
            mc_samples_scaled_np = mc_samples_scaled_torch.cpu().numpy()
            mc_samples_original_scale_list = []
            for i in range(mc_samples_scaled_np.shape[0]):
                sample_slice_scaled = mc_samples_scaled_np[i, :, :]
                mc_samples_original_scale_list.append(self.scaler_y.inverse_transform(sample_slice_scaled))
            mc_samples_original_scale = np.stack(mc_samples_original_scale_list, axis=0)

        # Inverse transform uncertainty (variance -> std_dev)
        # scaler_y.scale_ is an array of scales for each target feature
        # Variance scales by scale_factor^2
        scale_sq = self.scaler_y.scale_ ** 2

        # Since util_calc_uncertainty already averages for multi-target,
        # we use the mean of squared scales for consistency if scaler_y was multi-target.
        # If scaler_y was single-target, scale_sq is scalar.
        if variance_scaled_per_sample.shape[1] == 1:
            mean_scale_sq = np.mean(scale_sq) if len(scale_sq) > 0 else 1.0
            variance_original_scale = variance_scaled_per_sample * mean_scale_sq
        else: # Should not happen with current app.utils.calculate_uncertainty
            st.warning("Unexpected shape for variance_scaled_per_sample in ReptileModel. Using mean scale.")
            mean_scale_sq = np.mean(scale_sq) if len(scale_sq) > 0 else 1.0
            variance_original_scale = variance_scaled_per_sample * mean_scale_sq


        std_dev_original_scale_per_sample = np.sqrt(np.maximum(variance_original_scale, 1e-12)) # ensure non-negative before sqrt

        return mean_predictions_original_scale, std_dev_original_scale_per_sample.reshape(-1, 1), mc_samples_original_scale
    
def reptile_train(model, data, input_columns, target_columns, epochs, learning_rate, num_tasks, **kwargs):
    
    # Get labeled data
    labeled_data = data.dropna(subset=target_columns)
    MINIMUM_DATA_POINTS = 8
    if len(labeled_data) > 0 and len(labeled_data) < MINIMUM_DATA_POINTS:
        st.warning(f"🔄 Few labeled samples ({len(labeled_data)}). Tiling for robust training.")
        tile_factor = int(np.ceil(MINIMUM_DATA_POINTS / len(labeled_data)))
        labeled_data = pd.concat([labeled_data] * tile_factor, ignore_index=True)

        for col in input_columns:
            noise_scale = labeled_data[col].std() * 0.05
            labeled_data[col] += np.random.normal(0, noise_scale, size=len(labeled_data))
        for col in target_columns:
            noise_scale = max(labeled_data[col].std() * 0.02, 1e-6)
            labeled_data[col] += np.random.normal(0, noise_scale, size=len(labeled_data))

    
    scaler_x = RobustScaler().fit(data[input_columns])
    scaler_y = RobustScaler().fit(labeled_data[target_columns])
    
    inputs = scaler_x.transform(labeled_data[input_columns])
    targets = scaler_y.transform(labeled_data[target_columns])
    
    # Similar to SLAMD: Print target range to verify
    print(f"Target range: {labeled_data[target_columns].min().values} to {labeled_data[target_columns].max().values}")
    
    # Convert to tensors
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    
    # Use a more appropriate loss function for material properties regression
    # Huber loss is more robust to outliers than MSE
    loss_function = nn.HuberLoss(delta=1.0)
    
    # Like SLAMD: Use Adam with a higher initial learning rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Like SLAMD's GPR: Add a validation set for early stopping
    from sklearn.model_selection import train_test_split
    train_inputs, val_inputs, train_targets, val_targets = train_test_split(
        inputs, targets, test_size=0.2, random_state=42
    )
    
    # Train like SLAMD: Focus on fitting the training data very well first
    # This is different from standard meta-learning but works better for extrapolation
    best_loss = float('inf')
    best_model = None
    patience = 20
    patience_counter = 0
    
    # Progress bar for training
    progress_bar = st.progress(0)
    st.info(f"Starting Reptile training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # Update progress bar
        progress_bar.progress((epoch + 1) / epochs)

        # Partition data into tasks
        task_size = len(train_inputs) // num_tasks
        tasks = [(i * task_size, min((i + 1) * task_size, len(train_inputs))) for i in range(num_tasks)]
        
        for start, end in tasks:
            if start == end:
                continue
                
            task_inputs = train_inputs[start:end]
            task_targets = train_targets[start:end]
            
            # Inner loop optimization
            for _ in range(5):  # 5 steps per task
                optimizer.zero_grad()
                predictions = model(task_inputs)
                loss = loss_function(predictions, task_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validate after each epoch
        model.eval()
        with torch.no_grad():
            val_preds = model(val_inputs)
            val_loss = loss_function(val_preds, val_targets).item()
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss/len(tasks):.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_model:
        model.load_state_dict(best_model)
    
    # Like SLAMD: Final evaluation on all labeled data
    model.eval()
    with torch.no_grad():
        all_preds = model(inputs)
        final_loss = loss_function(all_preds, targets).item()
    
    print(f"Final loss on all labeled data: {final_loss:.4f}")
    
    # SLAMD-like feature: Verify prediction range at the end
    with torch.no_grad():
        all_preds = model(inputs).numpy()
        all_preds = scaler_y.inverse_transform(all_preds)
        pred_min = all_preds.min()
        pred_max = all_preds.max()
        true_min = scaler_y.inverse_transform(targets.numpy()).min()
        true_max = scaler_y.inverse_transform(targets.numpy()).max()
        
        print(f"Target range: {true_min:.2f} to {true_max:.2f}")
        print(f"Prediction range: {pred_min:.2f} to {pred_max:.2f}")
        
        # SLAMD-like extrapolation check
        extrapolation_factor = (pred_max - true_min) / (true_max - true_min)
        print(f"Extrapolation factor: {extrapolation_factor:.2f}x")
        if extrapolation_factor < 1.2:
            print("⚠️ Model has limited extrapolation capability. SLAMD would typically achieve 1.5-2.0x")

    model.is_trained = True
    model.scaler_x = scaler_x
    model.scaler_y = scaler_y
    
    return model, scaler_x, scaler_y


def evaluate_reptile(model, data, input_columns, target_columns, curiosity, weights, max_or_min, 
                    acquisition="UCB", strict_optimization=True):
   
    # Get unlabeled data
    unlabeled_data = data[data[target_columns].isna().any(axis=1)]
    labeled_data = data.dropna(subset=target_columns)
    
    if unlabeled_data.empty:
        st.warning("No unlabeled samples available for evaluation.")
        return None
    
    # Scale inputs using all available data for better distribution
    scaler_inputs = RobustScaler().fit(data[input_columns])
    
    # Scale targets using only labeled data
    scaler_targets = RobustScaler().fit(labeled_data[target_columns])
    
    # Transform unlabeled inputs
    inputs_infer = scaler_inputs.transform(unlabeled_data[input_columns])
    inputs_infer_tensor = torch.tensor(inputs_infer, dtype=torch.float32)
    
    # Get mean predictions
    model.eval()
    with torch.no_grad():
        predictions_scaled = model(inputs_infer_tensor).numpy()
    
    # Calculate uncertainty using Monte Carlo dropout from app.models
    from app.models import calculate_uncertainty_ensemble
    # model.train() is handled by calculate_uncertainty_ensemble
    # calculate_uncertainty_ensemble returns a tuple (uncertainty_np, predictions_stack)
    uncertainty_scores, _ = calculate_uncertainty_ensemble(
        model=model,
        X=inputs_infer_tensor,
        num_samples=50,  # Corresponds to num_perturbations
        dropout_rate=0.3, # Default dropout rate, can be parameterized if needed
        use_input_perturbation=True, # Reptile's old uncertainty used input perturbation
        noise_scale=0.05
    )
    if uncertainty_scores.ndim == 1: # Ensure shape (n_samples, 1)
        uncertainty_scores = uncertainty_scores.reshape(-1, 1)
    
    # Inverse transform predictions to original scale
    predictions = scaler_targets.inverse_transform(predictions_scaled)
    # 🔒 Ensure non-negative predictions, aligning with expected output characteristics.
    predictions = np.maximum(predictions, 0)

    # 🚫 Threshold enforcement
    # Define a minimum acceptable strength/property threshold.
    min_strength_threshold = 10
    # Identify rows where predictions are below the threshold for utility penalization.
    invalid_rows = np.any(predictions < min_strength_threshold, axis=1)
    
    # Calculate novelty
    X_labeled_scaled = scaler_inputs.transform(labeled_data[input_columns])
    novelty_scores = calculate_novelty(inputs_infer, X_labeled_scaled)
    
    # Ensure weights and max_or_min are properly formatted
    if not isinstance(weights, np.ndarray):
        weights = np.array(weights)
    
    if max_or_min is None or not isinstance(max_or_min, list):
        max_or_min = ['max'] * len(target_columns)
    
     # ⚖️ SLAMD-style acquisition weights adjustment
    weights = weights + np.random.uniform(0.01, 0.1, size=weights.shape)
    weights = np.clip(weights, 0.1, 1.0)
    
    # Get the direction for the first/main target property
    main_direction = max_or_min[0] if isinstance(max_or_min, list) and len(max_or_min) > 0 else "max"
   

    
    # Debug: Print optimization direction and current approach
    print(f"🎯 Target optimization direction: {main_direction}")
    print(f"🔍 Acquisition function: {acquisition}, Curiosity level: {curiosity}")
    
    # DIRECT OVERRIDE FOR MAXIMIZATION
    # 🎯 SLAMD-style acquisition selection
    from app.utils import select_acquisition_function
    if acquisition is None:
        acquisition = select_acquisition_function(curiosity, len(labeled_data))
    # Debug: Print optimization direction and current approach
    print(f"🎯 Target optimization direction: {main_direction}")
    print(f"🔍 Acquisition function: {acquisition}, Curiosity level: {curiosity}")


    # For maximization problems with strict_optimization, we'll use predictions directly
    if main_direction == "max" and strict_optimization:
        print("⚠️ Using direct maximization override to ensure highest values are selected")
        
        # For maximization, we create a simple rank-weighted utility
        # that strongly favors the highest predicted values
        raw_values = predictions[:, 0]
        ranks = np.argsort(np.argsort(-raw_values))  # Double argsort for ranking (highest=0)
        
        # Create utility based mostly on rank with a small uncertainty component
        # This heavily weights toward the highest predicted values
        rank_weight = 0.9  # 90% weight on highest prediction
        uncertainty_weight = 0.1  # Only 10% weight on uncertainty
        
        # Normalize ranks to [0,1] with highest value = 1
        norm_ranks = (len(ranks) - ranks) / len(ranks)
        
        # Normalize uncertainty for the small uncertainty component
        norm_uncertainty = uncertainty_scores.flatten() / (np.max(uncertainty_scores) + 1e-10)
        
        # Calculate utility directly - highest prediction gets highest utility
        direct_utility = rank_weight * norm_ranks + uncertainty_weight * norm_uncertainty
        
        # Set as utility scores
        utility_scores = direct_utility.reshape(-1, 1)
      

        
        print(f"Highest predicted value: {np.max(raw_values)}")
        print(f"Directly prioritizing highest predictions for maximization")
    else:
        # Original calculation for minimization or when strict_optimization is False
        utility_scores = calculate_utility(
            predictions,
            uncertainty_scores,
            novelty_scores,
            curiosity,
            weights,
            max_or_min,
            acquisition=acquisition
        )

        # 🛑 Apply penalty for invalid predictions
        # Penalize utility for predictions that fall below the min_strength_threshold.
        if invalid_rows.any():
            utility_scores[invalid_rows] = -np.inf

    
    # Create result DataFrame
    result_df = unlabeled_data.copy()
    
    # Add calculated metrics
    result_df["Utility"] = utility_scores.flatten()
    result_df["Uncertainty"] = np.clip(uncertainty_scores, 1e-6, None).flatten()
    result_df["Novelty"] = novelty_scores.flatten()
    
    # Add exploration-exploitation balance metrics
    result_df["Exploration"] = result_df["Uncertainty"] * result_df["Novelty"]
    result_df["Exploitation"] = 1.0 - result_df["Uncertainty"]
    
    # Add predicted values for target columns
    for i, col in enumerate(target_columns):
        result_df[col] = predictions[:, i]
    
    # Sort by utility and mark the best candidate
    result_df = result_df.sort_values(by="Utility", ascending=False)
    result_df["Selected for Testing"] = False
    result_df.iloc[0, result_df.columns.get_loc("Selected for Testing")] = True
    
    # Double-check for maximization problems that we're selecting a high value
    if main_direction == "max" and strict_optimization:
        # Get the index of the highest predicted value
        highest_value_idx = np.argmax(predictions[:, 0])
        highest_value = predictions[highest_value_idx, 0]
        
        # Get the selected value
        selected_idx = result_df.index[0]
        selected_value = result_df.iloc[0][target_columns[0]]
        
        print(f"Highest predicted value: {highest_value}")
        print(f"Selected value: {selected_value}")
        
        # If the selected value is significantly lower, override with the highest value
        if highest_value > selected_value * 1.1:  # If highest is more than 10% better
            print(f"⚠️ Overriding selection to ensure highest value ({highest_value})")
            # Find the row with the highest value
            highest_row = result_df[result_df[target_columns[0]] == highest_value]
            if not highest_row.empty:
                # Reset the selection
                result_df["Selected for Testing"] = False
                # Mark the highest value row
                highest_idx = highest_row.index[0]
                result_df.loc[highest_idx, "Selected for Testing"] = True
    
    # Debug: Report final selected candidate
    selected_idx = result_df["Selected for Testing"].idxmax()
    print(f"✅ Selected candidate with value: {result_df.loc[selected_idx, target_columns[0]]}")
    print(f"✅ Utility: {result_df.loc[selected_idx, 'Utility']}, Uncertainty: {result_df.loc[selected_idx, 'Uncertainty']}")
    
    # Reset index for better display
    result_df.reset_index(drop=True, inplace=True)
    
    # Organize columns for better presentation
    columns_to_front = ["Idx_Sample"] if "Idx_Sample" in result_df.columns else []
    metrics_columns = ["Utility", "Exploration", "Exploitation", "Novelty", "Uncertainty"]
    
    # Get all other columns excluding metrics and targets
    remaining_columns = [col for col in result_df.columns 
                         if col not in columns_to_front + metrics_columns + target_columns + ["Selected for Testing"]]
    
    # Set the new column order, ensuring all columns exist
    new_column_order = [col for col in (columns_to_front + metrics_columns + target_columns + 
                                       remaining_columns + ["Selected for Testing"])
                        if col in result_df.columns]
    
    result_df = result_df[new_column_order]
    
    return result_df