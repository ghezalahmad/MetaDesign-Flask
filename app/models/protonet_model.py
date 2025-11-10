
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
import streamlit as st

from app.models.bayesian_optimizer import multi_objective_bayesian_optimization
from app.utils import calculate_novelty
from app.pinn_utils import compute_physics_loss # Assuming this is a typo and not needed

class ProtoNetModel(nn.Module):
    def __init__(self, input_size, output_size, embedding_size=256, num_layers=3, dropout_rate=0.3):
        super(ProtoNetModel, self).__init__()
        
        self.encoder = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        self.encoder.append(nn.Linear(input_size, embedding_size))
        self.norm_layers.append(nn.LayerNorm(embedding_size))
        self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        for _ in range(num_layers - 2):
            self.encoder.append(nn.Linear(embedding_size, embedding_size))
            self.norm_layers.append(nn.LayerNorm(embedding_size))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        self.encoder.append(nn.Linear(embedding_size, embedding_size))
        self.norm_layers.append(nn.LayerNorm(embedding_size))
        self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        self.projector = nn.Linear(embedding_size, output_size)
        self.is_trained = False
        self.scaler_x = None
        self.scaler_y = None

    def encode(self, x):
        h = x
        for i, (layer, norm, dropout) in enumerate(zip(self.encoder, self.norm_layers, self.dropout_layers)):
            z = layer(h)
            z = norm(z)
            z = torch.relu(z)
            z = dropout(z)
            if i > 0 and h.shape == z.shape:
                h = h + z
            else:
                h = z
        return h
    
    def forward(self, x):
        embedding = self.encode(x)
        return torch.nn.functional.softplus(self.projector(embedding))

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

def protonet_train(model, data, input_columns, target_columns, epochs=50, learning_rate=0.001, batch_size=16):
    labeled_data = data.dropna(subset=target_columns).reset_index(drop=True)
    if len(labeled_data) < 3:
        st.error("Not enough labeled samples for Prototypical Network training.")
        return model, None, None

    scaler_x = RobustScaler().fit(data[input_columns])
    scaler_y = RobustScaler().fit(labeled_data[target_columns])
    
    inputs = scaler_x.transform(labeled_data[input_columns].values)
    targets = scaler_y.transform(labeled_data[target_columns].values)
    
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    progress_bar = st.progress(0)
    for epoch in range(epochs):
        progress_bar.progress((epoch + 1) / epochs)
        model.train()
        for i in range(0, len(inputs_tensor), batch_size):
            batch_inputs = inputs_tensor[i:i+batch_size]
            batch_targets = targets_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = loss_function(predictions, batch_targets)
            loss.backward()
            optimizer.step()
    
    model.is_trained = True
    model.scaler_x = scaler_x
    model.scaler_y = scaler_y
    model.scaler_x.feature_names_in_ = input_columns

    return model, scaler_x, scaler_y

def evaluate_protonet(model, data, input_columns, target_columns, curiosity, weights, max_or_min):
    labeled_data = data.dropna(subset=target_columns)
    candidate_df = data[data[target_columns[0]].isnull()].copy()

    if candidate_df.empty:
        st.warning("No candidate samples to evaluate.")
        return pd.DataFrame()

    train_inputs = labeled_data[input_columns]
    train_targets = labeled_data[target_columns].values
    candidate_inputs = candidate_df[input_columns]

    st.info("Using Bayesian Optimization with ProtoNet surrogate to score candidates.")
    
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
