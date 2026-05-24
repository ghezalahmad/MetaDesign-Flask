
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import streamlit as st
from app.utils.utils import calculate_novelty


class ReptileModel(nn.Module):
    """Single-output Reptile model for one target."""
    
    def __init__(self, input_size, output_size=1, hidden_size=128, num_layers=3, dropout_rate=0.3):
        super(ReptileModel, self).__init__()
        
        self.input_norm = nn.LayerNorm(input_size)
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layer_norms.append(nn.LayerNorm(hidden_size))
        self.dropouts.append(nn.Dropout(dropout_rate))
        
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layer_norms.append(nn.LayerNorm(hidden_size))
            self.dropouts.append(nn.Dropout(dropout_rate))

        self.output_layer = nn.Linear(hidden_size, output_size)
        self.is_trained = False
        self.scaler_x = None
        self.scaler_y = None
        # Store training range for soft clipping
        self.train_min = None
        self.train_max = None

    def forward(self, x):
        x = self.input_norm(x)
        h = x
        for i, (layer, norm, dropout) in enumerate(zip(self.layers, self.layer_norms, self.dropouts)):
            z = layer(h)
            z = norm(z)
            z = torch.relu(z)
            z = dropout(z)
            if i > 0 and h.shape == z.shape:
                h = h + z
            else:
                h = z
        return self.output_layer(h)

    def predict(self, X_input: pd.DataFrame | np.ndarray):
        mean_preds, _, _ = self.predict_with_uncertainty(X_input, num_samples=1)
        return mean_preds

    def predict_with_uncertainty(self, X_input: pd.DataFrame | np.ndarray, input_columns=None, num_samples=50, dropout_rate=0.3):
        if not self.is_trained or self.scaler_x is None or self.scaler_y is None:
            raise RuntimeError("Model is not trained yet or scalers are missing.")

        # Handle None value for num_samples - increased to 50 for better uncertainty
        if num_samples is None:
            num_samples = 50

        self.train()
        for module in self.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout_rate

        if isinstance(X_input, pd.DataFrame):
            X_processed = X_input[self.scaler_x.feature_names_in_].values
        else:
            X_processed = X_input

        X_scaled_np = self.scaler_x.transform(X_processed)
        X_tensor = torch.tensor(X_scaled_np, dtype=torch.float32)

        with torch.no_grad():
            predictions_scaled = [self(X_tensor).numpy() for _ in range(num_samples)]

        predictions_scaled = np.array(predictions_scaled)
        mean_predictions_scaled = predictions_scaled.mean(axis=0)
        std_dev_scaled = predictions_scaled.std(axis=0)

        mean_predictions_original = self.scaler_y.inverse_transform(mean_predictions_scaled)
        std_dev_original = std_dev_scaled * self.scaler_y.scale_
        
        # Apply SOFT CLIPPING: allow 50% beyond training range for exploration
        if self.train_min is not None and self.train_max is not None:
            train_range = self.train_max - self.train_min
            soft_min = self.train_min - 0.5 * train_range
            soft_max = self.train_max + 0.5 * train_range
            mean_predictions_original = np.clip(mean_predictions_original, soft_min, soft_max)
        
        # Return posterior samples for compatibility
        posterior_samples_original = np.array([
            self.scaler_y.inverse_transform(predictions_scaled[i])
            for i in range(num_samples)
        ])

        return mean_predictions_original, std_dev_original, posterior_samples_original

    def _get_input_columns(self):
        return self.scaler_x.feature_names_in_ if self.scaler_x else None


class ReptileMultiTargetWrapper:
    """Wrapper that trains separate Reptile models per target (like Lolopy)."""
    
    def __init__(self, input_size, target_columns, hidden_size=128, num_layers=3, dropout_rate=0.3):
        self.target_columns = target_columns
        self.models = {}
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.is_trained = False
        self.scaler_x = None
        
        # Create one model per target
        for col in target_columns:
            self.models[col] = ReptileModel(
                input_size=input_size,
                output_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_rate=dropout_rate
            )
    
    def predict_with_uncertainty(self, X_input, input_columns=None, num_samples=50):
        """Aggregate predictions from all per-target models."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        
        n_samples = len(X_input) if hasattr(X_input, '__len__') else X_input.shape[0]
        n_targets = len(self.target_columns)
        
        all_means = np.zeros((n_samples, n_targets))
        all_stds = np.zeros((n_samples, n_targets))
        
        for i, col in enumerate(self.target_columns):
            model = self.models[col]
            mean, std, _ = model.predict_with_uncertainty(X_input, input_columns, num_samples)
            all_means[:, i] = mean.flatten()
            all_stds[:, i] = std.flatten()
        
        return all_means, all_stds, None


def _train_single_reptile(model, inputs_tensor, targets_tensor, epochs, learning_rate, batch_size, train_min, train_max):
    """Train a single Reptile model with step size decay and more inner steps."""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    
    # Store training range for soft clipping
    model.train_min = train_min
    model.train_max = train_max
    
    # Meta-learning loop with STEP SIZE DECAY
    for meta_epoch in range(epochs):
        # Decaying step size: starts at 0.1, decays linearly to 0.01
        reptile_step = 0.1 * (1 - 0.9 * meta_epoch / epochs)
        
        # Create a copy of the model weights
        weights_before = model.state_dict()
        
        # Sample a task (a batch of data)
        indices = torch.randperm(len(inputs_tensor))
        task_indices = indices[:min(batch_size, len(inputs_tensor))]
        task_inputs = inputs_tensor[task_indices]
        task_targets = targets_tensor[task_indices]

        # Inner loop optimization - INCREASED TO 10 STEPS
        for _ in range(10):
            optimizer.zero_grad()
            predictions = model(task_inputs)
            loss = loss_function(predictions, task_targets)
            loss.backward()
            optimizer.step()

        # Reptile update with decaying step size
        weights_after = model.state_dict()
        model.load_state_dict({
            name: weights_before[name] + (weights_after[name] - weights_before[name]) * reptile_step 
            for name in weights_before
        })
    
    return model


def reptile_train(model, data, input_columns, target_columns, epochs, learning_rate, num_tasks, batch_size):
    """Train Reptile with per-target models (like Lolopy)."""
    labeled_data = data.dropna(subset=target_columns)
    if len(labeled_data) < 3:
        st.error("Not enough labeled data for Reptile training.")
        return model, None, None

    # Fit shared X scaler
    scaler_x = RobustScaler().fit(data[input_columns])
    scaler_x.feature_names_in_ = input_columns
    inputs = scaler_x.transform(labeled_data[input_columns])
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    
    # Create wrapper if model is vanilla ReptileModel
    if isinstance(model, ReptileModel):
        wrapper = ReptileMultiTargetWrapper(
            input_size=len(input_columns),
            target_columns=target_columns,
            hidden_size=128,
            num_layers=3,
            dropout_rate=0.3
        )
    else:
        wrapper = model
    
    # Train each target's model independently
    for col in target_columns:
        target_data = labeled_data[[col]].values
        scaler_y = RobustScaler().fit(target_data)
        targets_scaled = scaler_y.transform(target_data)
        targets_tensor = torch.tensor(targets_scaled, dtype=torch.float32)
        
        # Get training range for soft clipping
        train_min = target_data.min()
        train_max = target_data.max()
        
        # Train the per-target model
        single_model = wrapper.models[col]
        single_model.scaler_x = scaler_x
        single_model.scaler_y = scaler_y
        
        _train_single_reptile(
            single_model, inputs_tensor, targets_tensor, 
            epochs, learning_rate, batch_size,
            train_min, train_max
        )
        single_model.is_trained = True
    
    wrapper.is_trained = True
    wrapper.scaler_x = scaler_x

    return wrapper, scaler_x, None

def evaluate_reptile(model, data, input_columns, target_columns, curiosity, weights, max_or_min):
    """Evaluate Reptile model using WEBSLAMD utility formula."""
    labeled_data = data.dropna(subset=target_columns)
    candidate_df = data[data[target_columns].isnull().any(axis=1)].copy()

    if candidate_df.empty:
        st.warning("No candidate samples to evaluate.")
        return pd.DataFrame()

    candidate_inputs = candidate_df[input_columns]
    predictions, uncertainties, _ = model.predict_with_uncertainty(candidate_inputs)

    for i, col in enumerate(target_columns):
        candidate_df[col] = predictions[:, i]
        candidate_df[f"Uncertainty ({col})"] = uncertainties[:, i]

    # WEBSLAMD-EXACT UTILITY CALCULATION
    labels_mean = labeled_data[target_columns].mean(skipna=True)
    labels_std = labeled_data[target_columns].std(skipna=True).replace(0, 1)
    
    preds_norm = np.zeros_like(predictions, dtype=float)
    unc_norm = np.zeros_like(uncertainties, dtype=float)
    
    for i, col in enumerate(target_columns):
        mean_val = labels_mean.iloc[i]
        std_val = labels_std.iloc[i]
        preds_norm[:, i] = (predictions[:, i] - mean_val) / std_val
        if max_or_min[i].lower() == "min":
            preds_norm[:, i] *= -1
        preds_norm[:, i] *= weights[i]
        unc_norm[:, i] = uncertainties[:, i] / std_val
        unc_norm[:, i] *= weights[i]
    
    utility_scores = preds_norm.sum(axis=1) + max(0.0, float(curiosity)) * unc_norm.sum(axis=1)
    candidate_df["Utility"] = utility_scores
    candidate_df["Uncertainty"] = np.mean(uncertainties, axis=1)

    X_candidate_np = candidate_inputs.values
    X_labeled_np = labeled_data[input_columns].values
    novelty_scores = calculate_novelty(X_candidate_np, X_labeled_np)
    candidate_df["Novelty"] = novelty_scores

    candidate_df["Selected for Testing"] = False
    if not candidate_df.empty:
        max_utility_idx = candidate_df["Utility"].idxmax()
        candidate_df.loc[max_utility_idx, "Selected for Testing"] = True

    result_df = candidate_df.sort_values(by="Utility", ascending=False).reset_index(drop=True)
    return result_df
