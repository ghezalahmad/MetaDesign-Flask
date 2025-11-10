
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
import streamlit as st

from app.models.bayesian_optimizer import multi_objective_bayesian_optimization
from app.utils.utils import calculate_novelty

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

def protonet_train(model, data, input_columns, target_columns, epochs=50, learning_rate=0.001, num_tasks=5, num_shot=5, num_query=5):
    labeled_data = data.dropna(subset=target_columns).reset_index(drop=True)
    if len(labeled_data) < 10:
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

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for task in range(num_tasks):
            # Sample support and query sets
            indices = torch.randperm(len(inputs_tensor))
            support_indices = indices[:num_shot]
            query_indices = indices[num_shot:num_shot + num_query]

            support_inputs = inputs_tensor[support_indices]
            support_targets = targets_tensor[support_indices]
            query_inputs = inputs_tensor[query_indices]
            query_targets = targets_tensor[query_indices]

            # Compute prototypes
            support_embeddings = model.encode(support_inputs)
            prototypes = torch.mean(support_embeddings, dim=0, keepdim=True)
            prototype_targets = torch.mean(support_targets, dim=0, keepdim=True)

            # Predict query targets
            query_embeddings = model.encode(query_inputs)
            distances = torch.cdist(query_embeddings, prototypes)
            weights = torch.softmax(-distances, dim=1)
            predicted_targets = torch.matmul(weights, prototype_targets)

            loss = loss_function(predicted_targets, query_targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

    model.is_trained = True
    model.scaler_x = scaler_x
    model.scaler_y = scaler_y
    model.scaler_x.feature_names_in_ = input_columns

    return model, scaler_x, scaler_y

def evaluate_protonet(model, data, input_columns, target_columns, curiosity, weights, max_or_min):
    labeled_data = data.dropna(subset=target_columns)
    candidate_df = data[data[target_columns][0].isnull()].copy()

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
