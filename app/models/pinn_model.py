import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import streamlit as st
from app.utils import calculate_utility, calculate_novelty
from app.pinn_utils import compute_physics_loss

class PINNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=3, dropout_rate=0.3):
        super(PINNModel, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def predict_with_uncertainty(self, X_input: pd.DataFrame | np.ndarray, input_columns: list[str], num_samples=30, dropout_rate=0.3):
        if not getattr(self, 'is_trained', False) or self.scaler_x is None or self.scaler_y is None:
            raise RuntimeError("Model is not trained yet or scalers are missing. Call pinn_train first.")

        self.train()  # Enable dropout for MC samples

        # Set dropout rate for all dropout layers
        for module in self.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout_rate

        # Handle both DataFrame and numpy array inputs
        if isinstance(X_input, pd.DataFrame):
            # Ensure the column order matches the order used to fit the scaler
            X_processed = X_input[self.scaler_x.feature_names_in_].values
        else:  # Assumes numpy array
            X_processed = X_input

        X_scaled_np = self.scaler_x.transform(X_processed)
        X_tensor = torch.tensor(X_scaled_np, dtype=torch.float32)

        with torch.no_grad():
            predictions_scaled = [self(X_tensor).numpy() for _ in range(num_samples)]

        predictions_scaled = np.array(predictions_scaled) # Shape: (num_samples, num_points, num_targets)

        # Calculate mean and standard deviation in scaled space
        mean_predictions_scaled = predictions_scaled.mean(axis=0) # Shape: (num_points, num_targets)
        std_dev_scaled = predictions_scaled.std(axis=0) # Shape: (num_points, num_targets)

        # Inverse transform mean predictions to original scale
        mean_predictions_original_scale = self.scaler_y.inverse_transform(mean_predictions_scaled)

        # Inverse transform standard deviation
        std_dev_original_scale = std_dev_scaled * self.scaler_y.scale_

        # For utility calculation, we need a single uncertainty value per sample.
        # If we have multiple targets, we average the uncertainty (std dev) across them.
        if std_dev_original_scale.ndim > 1 and std_dev_original_scale.shape[1] > 1:
            uncertainty_per_sample = np.mean(std_dev_original_scale, axis=1)
        else:
            uncertainty_per_sample = std_dev_original_scale

        # Ensure the final uncertainty is a column vector
        uncertainty_per_sample = uncertainty_per_sample.reshape(-1, 1)

        return mean_predictions_original_scale, uncertainty_per_sample

def pinn_train(model, data, input_columns, target_columns, epochs, learning_rate, physics_loss_weight, batch_size):
    labeled_data = data.dropna(subset=target_columns)
    scaler_x = RobustScaler().fit(data[input_columns])
    scaler_y = RobustScaler().fit(labeled_data[target_columns])

    inputs = scaler_x.transform(labeled_data[input_columns])
    targets = scaler_y.transform(labeled_data[target_columns])

    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    for epoch in range(epochs):
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]

            optimizer.zero_grad()
            predictions = model(batch_inputs)
            data_loss = loss_function(predictions, batch_targets)

            # Placeholder for physics loss
            physics_loss = compute_physics_loss(predictions, batch_inputs)

            loss = data_loss + physics_loss_weight * physics_loss
            loss.backward()
            optimizer.step()

    model.is_trained = True
    model.scaler_x = scaler_x
    model.scaler_y = scaler_y

    return model, scaler_x, scaler_y

def evaluate_pinn(model, data, input_columns, target_columns, curiosity, weights, max_or_min):
    unlabeled_data = data[data[target_columns].isna().any(axis=1)]
    if unlabeled_data.empty:
        st.warning("No unlabeled samples available for evaluation.")
        return None

    predictions, uncertainty_scores = model.predict_with_uncertainty(unlabeled_data, input_columns)

    labeled_inputs = model.scaler_x.transform(data.dropna(subset=target_columns)[input_columns])
    novelty_scores = calculate_novelty(model.scaler_x.transform(unlabeled_data[input_columns]), labeled_inputs)

    utility_scores = calculate_utility(
        predictions,
        uncertainty_scores,
        novelty_scores,
        curiosity,
        weights,
        max_or_min
    )

    result_df = unlabeled_data.copy()
    result_df["Utility"] = utility_scores
    result_df["Uncertainty"] = uncertainty_scores
    result_df["Novelty"] = novelty_scores
    result_df["Exploration"] = result_df["Uncertainty"] * result_df["Novelty"]
    result_df["Exploitation"] = 1.0 - result_df["Uncertainty"]
    for i, col in enumerate(target_columns):
        result_df[col] = predictions[:, i]

    result_df = result_df.sort_values(by="Utility", ascending=False)
    result_df["Selected for Testing"] = False
    if not result_df.empty:
        result_df.iloc[0, result_df.columns.get_loc("Selected for Testing")] = True
    result_df.reset_index(drop=True, inplace=True)

    # Reorder columns for consistency
    columns_to_front = ["Idx_Sample"] if "Idx_Sample" in result_df.columns else []
    metrics_columns = ["Utility", "Exploration", "Exploitation", "Novelty", "Uncertainty"]
    remaining_columns = [col for col in result_df.columns if col not in columns_to_front + metrics_columns + target_columns + ["Selected for Testing"]]
    new_column_order = columns_to_front + metrics_columns + target_columns + remaining_columns + ["Selected for Testing"]
    new_column_order = [col for col in new_column_order if col in result_df.columns]

    result_df = result_df[new_column_order]
    return result_df
