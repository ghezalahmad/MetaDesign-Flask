
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import streamlit as st
from app.models.bayesian_optimizer import multi_objective_bayesian_optimization
from app.utils.utils import calculate_novelty

class ReptileModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256, num_layers=3, dropout_rate=0.3):
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
        mean_preds, _ = self.predict_with_uncertainty(X_input, num_samples=1)
        return mean_preds

    def predict_with_uncertainty(self, X_input: pd.DataFrame | np.ndarray, input_columns=None, num_samples=30, dropout_rate=0.3):
        if not self.is_trained or self.scaler_x is None or self.scaler_y is None:
            raise RuntimeError("Model is not trained yet or scalers are missing.")

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

        return mean_predictions_original, std_dev_original

    def _get_input_columns(self):
        return self.scaler_x.feature_names_in_ if self.scaler_x else None

def reptile_train(model, data, input_columns, target_columns, epochs, learning_rate, num_tasks, batch_size):
    labeled_data = data.dropna(subset=target_columns)
    if len(labeled_data) < 3:
        st.error("Not enough labeled data for Reptile training.")
        return model, None, None

    scaler_x = RobustScaler().fit(data[input_columns])
    scaler_y = RobustScaler().fit(labeled_data[target_columns])

    inputs = scaler_x.transform(labeled_data[input_columns])
    targets = scaler_y.transform(labeled_data[target_columns])

    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    # Meta-learning loop
    for meta_epoch in range(epochs):
        # Create a copy of the model
        weights_before = model.state_dict()
        
        # Sample a task (a batch of data)
        indices = torch.randperm(len(inputs_tensor))
        task_indices = indices[:batch_size]
        task_inputs = inputs_tensor[task_indices]
        task_targets = targets_tensor[task_indices]

        # Inner loop optimization
        for _ in range(5):  # 5 steps per task
            optimizer.zero_grad()
            predictions = model(task_inputs)
            loss = loss_function(predictions, task_targets)
            loss.backward()
            optimizer.step()

        # Reptile update
        weights_after = model.state_dict()
        model.load_state_dict({name: weights_before[name] + (weights_after[name] - weights_before[name]) * 0.1 for name in weights_before})


    model.is_trained = True
    model.scaler_x = scaler_x
    model.scaler_y = scaler_y
    model.scaler_x.feature_names_in_ = input_columns

    return model, scaler_x, scaler_y

def evaluate_reptile(model, data, input_columns, target_columns, curiosity, weights, max_or_min):
    labeled_data = data.dropna(subset=target_columns)
    candidate_df = data[data[target_columns].isnull().any(axis=1)].copy()

    if candidate_df.empty:
        st.warning("No candidate samples to evaluate.")
        return pd.DataFrame()

    train_inputs = labeled_data[input_columns]
    train_targets = labeled_data[target_columns].values
    candidate_inputs = candidate_df[input_columns]

    st.info("Using Bayesian Optimization with Reptile surrogate to score candidates.")
    
    utility_scores = multi_objective_bayesian_optimization(
        train_inputs=train_inputs,
        train_targets=train_targets,
        candidate_inputs=candidate_inputs,
        weights=np.array(weights),
        max_or_min=max_or_min,
        curiosity=curiosity,
        acquisition="UCB",
        strategy="weighted_sum",
        surrogate_model=model,
        input_columns=input_columns
    )

    predictions, uncertainties = model.predict_with_uncertainty(candidate_inputs)

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
