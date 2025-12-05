import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import streamlit as st # Using Streamlit components for logging/warnings

# Assuming these internal imports are correct for your setup
from app.models.bayesian_optimizer import multi_objective_bayesian_optimization
from app.pinn_utils import compute_physics_loss
from app.utils.utils import calculate_novelty

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
        self.is_trained = False
        self.scaler_x = None
        self.scaler_y = None
        self.default_dropout_rate = dropout_rate # Store default rate

    def forward(self, x):
        return self.net(x)

    def predict(self, X_input: pd.DataFrame | np.ndarray):
        """Generates predictions without uncertainty."""
        mean_preds, _, _ = self.predict_with_uncertainty(X_input, num_samples=1)
        return mean_preds

    def predict_with_uncertainty(self, X_input: pd.DataFrame | np.ndarray, input_columns=None, num_samples=30, dropout_rate=None):
        """
        Generates predictions and uncertainties (Std Dev) using MC Dropout.
        dropout_rate is optional; uses the model's default if not provided.
        
        Returns:
            mean_predictions_original: (n_samples, n_targets)
            std_dev_original: (n_samples, n_targets)
            predictions_scaled: (num_samples, n_samples, n_targets) - posterior samples
        """
        if not self.is_trained or self.scaler_x is None or self.scaler_y is None:
            # Handle case where model is not ready
            n = len(X_input)
            out_targets = self.scaler_y.scale_.shape[0] if hasattr(self.scaler_y, 'scale_') else 1
            return np.zeros((n, out_targets)), np.zeros((n, out_targets)), None

        # 1. Enable dropout during inference (MC Dropout)
        self.train()
        
        # Handle None value for num_samples
        if num_samples is None:
            num_samples = 30  # Default value
        
        rate = dropout_rate if dropout_rate is not None else self.default_dropout_rate
        for module in self.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = rate

        # 2. Prepare Input
        if isinstance(X_input, pd.DataFrame):
            # Ensure columns are ordered correctly if scaler_x has feature_names_in_
            if hasattr(self.scaler_x, 'feature_names_in_'):
                X_processed = X_input[self.scaler_x.feature_names_in_].values
            else:
                X_processed = X_input.values
        else:
            X_processed = X_input

        X_scaled_np = self.scaler_x.transform(X_processed)
        X_tensor = torch.tensor(X_scaled_np, dtype=torch.float32)

        # 3. MC Sampling
        with torch.no_grad():
            predictions_scaled = [self(X_tensor).numpy() for _ in range(num_samples)]

        predictions_scaled = np.array(predictions_scaled) # Shape (num_samples, n_samples, n_targets)
        
        # 4. Calculate Mean and Std Dev in Scaled Space
        mean_predictions_scaled = predictions_scaled.mean(axis=0) # (n_samples, n_targets)
        std_dev_scaled = predictions_scaled.std(axis=0)           # (n_samples, n_targets)

        # 5. Inverse Transform Mean and Uncertainty (Std Dev)
        mean_predictions_original = self.scaler_y.inverse_transform(mean_predictions_scaled)
        # Var(k*X) = k^2 * Var(X) -> Std(k*X) = k * Std(X)
        std_dev_original = std_dev_scaled * self.scaler_y.scale_ 

        # 6. Inverse transform posterior samples for compatibility with BayesianOptimizer
        posterior_samples_original = np.array([
            self.scaler_y.inverse_transform(predictions_scaled[i])
            for i in range(num_samples)
        ])

        # 7. Set model back to eval mode (optional, but good practice if not done externally)
        self.eval() 
        
        return mean_predictions_original, std_dev_original, posterior_samples_original

    def _get_input_columns(self):
        """Returns the input columns used for training."""
        return self.scaler_x.feature_names_in_ if self.scaler_x else None

def pinn_train(model, data, input_columns, target_columns, epochs, learning_rate, physics_loss_weight, batch_size):
    """Trains a PINN model using data loss and physics loss."""
    
    # 1. Data Preparation and Scaling
    labeled_data = data.dropna(subset=target_columns)
    
    # Handle insufficient labeled data
    if labeled_data.empty:
        st.warning("PINN: Not enough labeled data to train.")
        return model, None, None

    # RobustScaler is good for outliers
    scaler_x = RobustScaler().fit(data[input_columns])
    scaler_y = RobustScaler().fit(labeled_data[target_columns])

    inputs = scaler_x.transform(labeled_data[input_columns])
    targets = scaler_y.transform(labeled_data[target_columns])

    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    # 2. Training Loop
    print(f"Starting PINN training: {len(labeled_data)} samples, {epochs} epochs, Physics Weight: {physics_loss_weight}")
    for epoch in range(epochs):
        # Set model to train mode
        model.train()
        
        for i in range(0, len(inputs_tensor), batch_size):
            batch_inputs = inputs_tensor[i:i+batch_size]
            batch_targets = targets_tensor[i:i+batch_size]

            optimizer.zero_grad()
            predictions = model(batch_inputs)
            
            # Data Loss (fitting the measured points)
            data_loss = loss_function(predictions, batch_targets)
            
            # Physics Loss (enforcing physical laws, requires internal function)
            physics_loss = compute_physics_loss(predictions, batch_inputs)
            
            # Total Loss
            loss = data_loss + physics_loss_weight * physics_loss
            loss.backward()
            optimizer.step()

    # 3. Finalization
    model.is_trained = True
    model.scaler_x = scaler_x
    model.scaler_y = scaler_y
    # Store feature names in scaler_x for compatibility with predict_with_uncertainty
    if hasattr(scaler_x, 'feature_names_in_'):
        model.scaler_x.feature_names_in_ = input_columns
    
    # Set model to evaluation mode after training
    model.eval()

    print("PINN Training Completed.")
    return model, scaler_x, scaler_y

def evaluate_pinn(model, data, input_columns, target_columns, curiosity, weights, max_or_min):
    """Evaluates the PINN model on candidate samples using Bayesian Optimization."""

    labeled_data = data.dropna(subset=target_columns)
    
    # Identify unlabeled (candidate) samples
    if isinstance(target_columns, list) and len(target_columns) > 0:
        candidate_df = data[data[target_columns[0]].isnull()].copy()
    else:
        candidate_df = data[data[target_columns].isnull()].copy()

    if candidate_df.empty:
        st.warning("No candidate samples to evaluate.")
        return pd.DataFrame()
    
    # Check model training status and train if necessary
    if not getattr(model, 'is_trained', False):
        st.info("PINN: Model untrained. Training now...")
        # Since this function only takes the model, we can't fully train here
        # Assuming the caller has trained the model before calling evaluate
        # If not, the prediction method will return zeros, which is safe.

    train_inputs = labeled_data[input_columns]
    train_targets = labeled_data[target_columns].values
    candidate_inputs = candidate_df[input_columns]

    st.info("Using Bayesian Optimization with PINN surrogate to score candidates.")

    # 1. Run Bayesian Optimization to get Utility Scores
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

    # 2. Get Predictions and Uncertainties
    predictions, uncertainties, _ = model.predict_with_uncertainty(candidate_inputs, input_columns=input_columns)

    # 3. Map Predictions and Target-Specific Uncertainties
    for i, col in enumerate(target_columns):
        # Predictions
        candidate_df[col] = predictions[:, i]
        # Target-specific Uncertainties (Std Dev)
        candidate_df[f"Uncertainty ({col})"] = uncertainties[:, i]

    # 4. Assign Utility Scores
    if utility_scores is None:
        candidate_df["Utility"] = 0.0
    else:
        utility_scores = np.array(utility_scores, dtype=np.float64).flatten()
        # Safety truncation
        n = min(len(utility_scores), len(candidate_df))
        if len(utility_scores) != len(candidate_df):
            candidate_df = candidate_df.iloc[:n].copy()
            utility_scores = utility_scores[:n]
        candidate_df["Utility"] = utility_scores
    
    candidate_df["Utility"] = pd.to_numeric(candidate_df["Utility"], errors="coerce").fillna(0.0).astype(float)


    # 5. Calculate Aggregate Uncertainty (Mean of target uncertainties)
    candidate_df["Uncertainty"] = np.mean(uncertainties, axis=1)

    # 6. Calculate Novelty
    X_candidate_np = candidate_inputs.values
    X_labeled_np = labeled_data[input_columns].values
    novelty_scores = calculate_novelty(X_candidate_np, X_labeled_np)
    candidate_df["Novelty"] = novelty_scores

    # 7. Exploration / Exploitation
    candidate_df["Exploration"] = candidate_df["Uncertainty"] * (1.0 + curiosity)
    candidate_df["Exploitation"] = candidate_df[target_columns].mean(axis=1)

    # 8. Selection Flag
    candidate_df["Selected for Testing"] = False
    if not candidate_df["Utility"].empty:
        max_utility_idx = candidate_df["Utility"].idxmax()
        candidate_df.loc[max_utility_idx, "Selected for Testing"] = True

    # 9. Final Sort
    result_df = candidate_df.sort_values(by="Utility", ascending=False).reset_index(drop=True)
    return result_df