import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from sklearn.preprocessing import RobustScaler
import streamlit as st # Using Streamlit components for logging/warnings

# Assuming these internal imports are correct for your setup
from app.models.base import SurrogateModel
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
        
        # 6. Apply SOFT CLIPPING: allow 50% beyond training range for exploration
        if hasattr(self, 'train_min') and hasattr(self, 'train_max') and self.train_min is not None:
            train_range = self.train_max - self.train_min
            soft_min = self.train_min - 0.5 * train_range
            soft_max = self.train_max + 0.5 * train_range
            mean_predictions_original = np.clip(mean_predictions_original, soft_min, soft_max)

        # 7. Inverse transform posterior samples for compatibility with BayesianOptimizer
        posterior_samples_original = np.array([
            self.scaler_y.inverse_transform(predictions_scaled[i])
            for i in range(num_samples)
        ])

        # 8. Set model back to eval mode (optional, but good practice if not done externally)
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
    
    # Store training range for soft clipping (like MAML/Reptile)
    model.train_min = labeled_data[target_columns].min().values
    model.train_max = labeled_data[target_columns].max().values

    # 2. Training Loop
    print(f"Starting PINN training: {len(labeled_data)} samples, {epochs} epochs, Physics Weight: {physics_loss_weight}")
    
    total_data_loss = 0.0
    total_physics_loss = 0.0
    
    for epoch in range(epochs):
        # Set model to train mode
        model.train()
        epoch_data_loss = 0.0
        epoch_physics_loss = 0.0
        
        for i in range(0, len(inputs_tensor), batch_size):
            batch_inputs = inputs_tensor[i:i+batch_size]
            batch_targets = targets_tensor[i:i+batch_size]

            optimizer.zero_grad()
            predictions = model(batch_inputs)
            
            # Data Loss (fitting the measured points)
            data_loss = loss_function(predictions, batch_targets)
            
            # Physics Loss (enforcing physical laws)
            physics_loss = compute_physics_loss(predictions, batch_inputs, physics_weight=1.0)
            
            # Total Loss (physics_loss already weighted internally, but we apply user weight)
            loss = data_loss + physics_loss_weight * physics_loss
            loss.backward()
            optimizer.step()
            
            epoch_data_loss += data_loss.item()
            epoch_physics_loss += physics_loss.item()
        
        total_data_loss = epoch_data_loss
        total_physics_loss = epoch_physics_loss
        
        # Log progress every 25 epochs
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Data Loss={total_data_loss:.4f}, Physics Loss={total_physics_loss:.4f}")

    # 3. Finalization
    model.is_trained = True
    model.scaler_x = scaler_x
    model.scaler_y = scaler_y
    # Store feature names in scaler_x for compatibility with predict_with_uncertainty
    model.scaler_x.feature_names_in_ = input_columns
    
    # Set model to evaluation mode after training
    model.eval()

    print(f"PINN Training Completed. Final Data Loss: {total_data_loss:.4f}, Physics Loss: {total_physics_loss:.4f}")
    return model, scaler_x, scaler_y

def evaluate_pinn(model, data, input_columns, target_columns, curiosity, weights, max_or_min):
    """Evaluates the PINN model on candidate samples using WEBSLAMD utility formula."""
    from app.utils.webslamd_utility import calculate_webslamd_utility

    labeled_data = data.dropna(subset=target_columns)
    
    # Identify unlabeled (candidate) samples
    if isinstance(target_columns, list) and len(target_columns) > 0:
        candidate_df = data[data[target_columns[0]].isnull()].copy()
    else:
        candidate_df = data[data[target_columns].isnull()].copy()

    if candidate_df.empty:
        st.warning("No candidate samples to evaluate.")
        return pd.DataFrame()
    
    candidate_inputs = candidate_df[input_columns]

    # 1. Get Predictions and Uncertainties
    predictions, uncertainties, _ = model.predict_with_uncertainty(candidate_inputs, input_columns=input_columns)

    # 2. Map Predictions and Target-Specific Uncertainties
    for i, col in enumerate(target_columns):
        candidate_df[col] = predictions[:, i]
        candidate_df[f"Uncertainty ({col})"] = uncertainties[:, i]

    # 3. Calculate Utility using centralized function
    utility_scores = calculate_webslamd_utility(
        predictions=predictions,
        uncertainties=uncertainties,
        labeled_data=labeled_data,
        target_columns=target_columns,
        max_or_min=max_or_min,
        weights=weights,
        curiosity=curiosity
    )
    candidate_df["Utility"] = utility_scores
    candidate_df["Utility"] = pd.to_numeric(candidate_df["Utility"], errors="coerce").fillna(0.0).astype(float)

    # 4. Calculate Aggregate Uncertainty
    candidate_df["Uncertainty"] = np.mean(uncertainties, axis=1)

    # 5. Calculate Novelty
    X_candidate_np = candidate_inputs.values
    X_labeled_np = labeled_data[input_columns].values
    novelty_scores = calculate_novelty(X_candidate_np, X_labeled_np)
    candidate_df["Novelty"] = novelty_scores

    # 6. Selection Flag
    candidate_df["Selected for Testing"] = False
    if not candidate_df["Utility"].empty:
        max_utility_idx = candidate_df["Utility"].idxmax()
        candidate_df.loc[max_utility_idx, "Selected for Testing"] = True

    # 7. Final Sort
    result_df = candidate_df.sort_values(by="Utility", ascending=False).reset_index(drop=True)
    return result_df


# =============================================================================
# SURROGATE MODEL WRAPPER (Implements SurrogateModel Interface)
# =============================================================================

class PINNSurrogate(SurrogateModel):
    """
    PINN Surrogate Model that implements the SurrogateModel interface.
    
    This wrapper class composes the PINNModel (nn.Module) and provides
    a standardized interface for training and inference.
    """
    
    def __init__(self, hidden_size: int = 128, num_layers: int = 3, 
                 dropout_rate: float = 0.3, epochs: int = 100,
                 learning_rate: float = 0.001, physics_loss_weight: float = 0.1,
                 batch_size: int = 32):
        """
        Initialize PINNSurrogate.
        
        Args:
            hidden_size: Number of hidden units per layer
            num_layers: Number of hidden layers
            dropout_rate: Dropout rate for regularization
            epochs: Training epochs
            learning_rate: Optimizer learning rate
            physics_loss_weight: Weight for physics loss component
            batch_size: Training batch size
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.physics_loss_weight = physics_loss_weight
        self.batch_size = batch_size
        
        # The underlying PyTorch model
        self._model: Optional[PINNModel] = None
    
    def train(self, data: pd.DataFrame, input_columns: List[str],
              target_columns: List[str], **kwargs) -> 'PINNSurrogate':
        """
        Train the PINN model on labeled data.
        
        Args:
            data: Full dataset (labeled + unlabeled)
            input_columns: List of feature column names
            target_columns: List of target column names
            **kwargs: Additional hyperparameters (override defaults)
        
        Returns:
            self: The trained surrogate instance
        """
        # Override hyperparameters if provided
        epochs = kwargs.get('epochs', self.epochs)
        learning_rate = kwargs.get('learning_rate', self.learning_rate)
        physics_loss_weight = kwargs.get('physics_loss_weight', self.physics_loss_weight)
        batch_size = kwargs.get('batch_size', self.batch_size)
        
        # Create the underlying PyTorch model
        input_size = len(input_columns)
        output_size = len(target_columns)
        self._model = PINNModel(
            input_size=input_size,
            output_size=output_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate
        )
        
        # Train using the existing pinn_train function
        self._model, scaler_x, scaler_y = pinn_train(
            model=self._model,
            data=data,
            input_columns=input_columns,
            target_columns=target_columns,
            epochs=epochs,
            learning_rate=learning_rate,
            physics_loss_weight=physics_loss_weight,
            batch_size=batch_size
        )
        
        # Store training metadata
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.input_columns = input_columns
        self.target_columns = target_columns
        
        # Store training stats for soft clipping
        labeled_data = data.dropna(subset=target_columns)
        self.store_training_stats(labeled_data, target_columns)
        
        self.is_trained = self._model.is_trained
        return self
    
    def predict_with_uncertainty(self, X: pd.DataFrame,
                                  input_columns: Optional[List[str]] = None,
                                  num_samples: int = 50
                                  ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Generate predictions with uncertainty estimates using MC Dropout.
        
        Args:
            X: Input features (DataFrame)
            input_columns: Optional column names for ordering
            num_samples: Number of MC samples for uncertainty
        
        Returns:
            tuple: (mean_predictions, uncertainties, posterior_samples)
        """
        if not self.is_trained or self._model is None:
            raise RuntimeError("Model is not trained yet.")
        
        # Use input_columns from training if not provided
        cols = input_columns or self.input_columns
        
        # Delegate to the underlying PINNModel
        return self._model.predict_with_uncertainty(X, input_columns=cols, num_samples=num_samples)
    
    def get_underlying_model(self) -> PINNModel:
        """Get the underlying PyTorch PINNModel for advanced use."""
        return self._model