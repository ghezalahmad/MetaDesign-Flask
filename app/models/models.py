import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Internal Imports
from app.models.bayesian_optimizer import multi_objective_bayesian_optimization
from app.utils.utils import calculate_novelty

# ==========================================
# 1. THE MAML MODEL CLASS
# ==========================================

class MAMLModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=3, dropout_rate=0.3):
        super(MAMLModel, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # Attributes expected by BayesianOptimizer
        self.is_trained = False
        self.scaler_x = None
        self.scaler_y = None
        
        # --- Architecture ---
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.input_bn = nn.BatchNorm1d(hidden_size)
        self.input_dropout = nn.Dropout(dropout_rate)
        
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.bn_layers.append(nn.BatchNorm1d(hidden_size))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # He Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.input_layer(x)
        # Batch Norm requires > 1 sample. If 1 sample, skip BN to avoid error.
        if x.shape[0] > 1:
            x = self.input_bn(x)
        x = torch.relu(x)
        x = self.input_dropout(x)
        
        for i in range(len(self.layers)):
            identity = x
            x = self.layers[i](x)
            if x.shape[0] > 1:
                x = self.bn_layers[i](x)
            x = torch.relu(x)
            x = self.dropout_layers[i](x)
            if identity.shape == x.shape:
                x = x + identity
        
        output = self.output_layer(x)
        return torch.nn.functional.softplus(output) # Ensure positive outputs for material properties

    def fit(self, X, y):
        """
        Dummy method to satisfy BayesianOptimizer checks, though training 
        should happen via 'meta_train'.
        """
        pass

    def predict_with_uncertainty(self, X, input_columns=None, num_samples=30):
        """
        Primary inference method called by BayesianOptimizer.
        
        Args:
            X: DataFrame or NumPy array.
            input_columns: List of column names (optional).
            num_samples: Number of MC Dropout samples for posterior estimation.
            
        Returns:
            (mean_predictions, std_deviations, posterior_samples) in original scale.
        """
        if not self.is_trained or self.scaler_x is None or self.scaler_y is None:
             # Return zero-arrays if called before training (fallback)
             n = len(X)
             out = self.output_size
             return np.zeros((n, out)), np.zeros((n, out)), np.zeros((num_samples or 1, n, out))

        # 1. Preprocess Input
        X_np = None
        if isinstance(X, pd.DataFrame):
            # If input_columns provided, filter. Else use all.
            if input_columns:
                X_np = self.scaler_x.transform(X[input_columns])
            else:
                # Fallback: assume DataFrame columns match training exactly
                X_np = self.scaler_x.transform(X)
        else:
            # Assume Numpy array is already in correct column order
            X_np = self.scaler_x.transform(X)
            
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        
        # 2. Mean Prediction (Eval Mode - Dropout OFF)
        self.eval()
        with torch.no_grad():
            mean_preds_scaled = self(X_tensor).numpy()
        
        # 3. Uncertainty / Posterior (Train Mode - Dropout ON)
        self.train() 
        mc_preds_list = []
        
        # If num_samples is None, default to 30
        ns = num_samples if num_samples is not None else 30
        
        with torch.no_grad():
            for _ in range(ns):
                mc_preds_list.append(self(X_tensor).numpy())
        
        self.eval() # Reset to eval
        
        # Stack: (n_samples, n_points, n_targets)
        posterior_samples_scaled = np.stack(mc_preds_list)
        
        # 4. Calculate Statistics in Scaled Space
        variance_scaled = np.var(posterior_samples_scaled, axis=0) # (n_points, n_targets)
        
        # 5. Inverse Transform to Original Scale
        mean_preds_original = self.scaler_y.inverse_transform(mean_preds_scaled)
        
        # Inverse transform posterior samples
        # We need to iterate because inverse_transform expects 2D (n_points, n_targets)
        posterior_samples_original_list = []
        for i in range(ns):
            posterior_samples_original_list.append(
                self.scaler_y.inverse_transform(posterior_samples_scaled[i])
            )
        posterior_samples_original = np.stack(posterior_samples_original_list)

        # Inverse transform variance -> Std Dev
        # Var(k*X) = k^2 * Var(X)
        scale_sq = self.scaler_y.scale_ ** 2
        variance_original = variance_scaled * scale_sq
        std_original = np.sqrt(np.maximum(variance_original, 1e-12))
        
        # Return exact signature required by BayesianOptimizer
        return mean_preds_original, std_original, posterior_samples_original
       


    def predict(self, X):
        """Scikit-learn compatibility wrapper."""
        mu, _, _ = self.predict_with_uncertainty(X, num_samples=1)
        return mu


# ==========================================
# 2. TRAINING LOGIC
# ==========================================

def meta_train(meta_model: MAMLModel, data: pd.DataFrame, input_columns: list, target_columns: list,
               epochs: int = 100, inner_lr: float = 0.01, outer_lr: float = 0.001,
               num_tasks: int = 4, inner_lr_decay: float = 0.95, curiosity: float = 0,
               min_samples_per_task: int = 3, early_stopping_patience: int = 10):
    
    # Optimizer Setup
    optimizer = optim.AdamW(meta_model.parameters(), lr=outer_lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Prepare Data
    labeled_data = data.dropna(subset=target_columns).copy()
    if not labeled_data.empty:
         labeled_data = labeled_data.sort_values(by=target_columns[0]).reset_index(drop=True)

    # Data Tiling (Robustness for small data < 8 rows)
    if 0 < len(labeled_data) < 8:
        tile_factor = int(np.ceil(8 / len(labeled_data)))
        labeled_data = pd.concat([labeled_data] * tile_factor, ignore_index=True)
        # Add noise to prevent singular matrices or overfitting identical rows
        for col in input_columns + target_columns:
            if col in labeled_data.columns:
                std = labeled_data[col].std()
                noise = np.random.normal(0, std * 0.05 if std > 0 else 1e-4, size=len(labeled_data))
                labeled_data[col] += noise

    if labeled_data.empty:
        return meta_model, None, None

    # Scaling
    scaler_inputs = RobustScaler().fit(labeled_data[input_columns])
    scaler_targets = RobustScaler().fit(labeled_data[target_columns])
    
    # Attach scalers immediately (useful if training crashes early, at least object has them)
    meta_model.scaler_x = scaler_inputs
    meta_model.scaler_y = scaler_targets
    
    inputs = torch.tensor(scaler_inputs.transform(labeled_data[input_columns]), dtype=torch.float32)
    targets = torch.tensor(scaler_targets.transform(labeled_data[target_columns]), dtype=torch.float32)
    
    loss_function = nn.SmoothL1Loss() if len(labeled_data) >= 10 else nn.MSELoss()
    batch_size = max(2, min(8, len(inputs) // 4))
    
    # --- Main Loop ---
    best_loss = float('inf')
    best_model_state = None
    patience = 0
    
    print(f"Starting MAML training: {len(inputs)} samples, {epochs} epochs.")

    for epoch in range(epochs):
        meta_model.train()
        meta_losses = []
        
        # Adjust Inner LR
        curr_inner_lr = max(inner_lr * (inner_lr_decay ** (epoch / 10)), inner_lr * 0.1)
        
        # Task Creation (Sliding Window)
        window_size = max(len(inputs) // num_tasks, 1)
        for i in range(num_tasks):
            start = i * window_size
            end = min(start + window_size, len(inputs))
            val_idx = np.arange(start, end)
            train_idx = np.setdiff1d(np.arange(len(inputs)), val_idx)
            
            if len(train_idx) < min_samples_per_task: continue

            s_x, s_y = inputs[train_idx], targets[train_idx]
            q_x, q_y = inputs[val_idx], targets[val_idx]
            
            # Inner Loop (FO-MAML)
            fast_model = copy.deepcopy(meta_model)
            inner_opt = optim.SGD(fast_model.parameters(), lr=curr_inner_lr)
            
            # Adaptation steps
            for _ in range(3): 
                loss = loss_function(fast_model(s_x), s_y)
                inner_opt.zero_grad()
                loss.backward()
                inner_opt.step()
            
            # Outer Loop (Query set)
            q_preds = fast_model(q_x)
            task_loss = loss_function(q_preds, q_y)
            meta_losses.append(task_loss)
            
        if meta_losses:
            optimizer.zero_grad()
            loss = torch.stack(meta_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(meta_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            current_loss = loss.item()
            
            if current_loss < best_loss:
                best_loss = current_loss
                best_model_state = copy.deepcopy(meta_model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    if best_model_state:
        meta_model.load_state_dict(best_model_state)

    meta_model.is_trained = True
    print("MAML Training Completed.")
    return meta_model, scaler_inputs, scaler_targets


# ==========================================
# 3. EVALUATION LOGIC
# ==========================================

def evaluate_maml(model: MAMLModel, data: pd.DataFrame, input_columns: list,
                  target_columns: list, curiosity: float, weights_targets: np.ndarray,
                  max_or_min_targets: list[str]):
    """
    Evaluates candidates using MAML as the surrogate.
    Mirrors evaluate_lolopy_model structure.
    """
    
    # 1. Split labeled vs candidate rows
    labeled_data = data.dropna(subset=target_columns)
    candidate_df = data[data[target_columns[0]].isnull()].copy()

    if candidate_df.empty:
        return pd.DataFrame()

    # 2. Ensure model is trained
    if not getattr(model, 'is_trained', False):
        print("evaluate_maml: Model untrained. Training now...")
        model, _, _ = meta_train(model, data, input_columns, target_columns)

    train_inputs = labeled_data[input_columns]
    train_targets = labeled_data[target_columns].values
    candidate_inputs = candidate_df[input_columns]

    # 3. Run Bayesian Optimization
    # This will call model.predict_with_uncertainty internally
    utility_scores = multi_objective_bayesian_optimization(
        train_inputs=train_inputs,
        train_targets=train_targets,
        candidate_inputs=candidate_inputs,
        weights=weights_targets,
        max_or_min=max_or_min_targets,
        curiosity=curiosity,
        acquisition="UCB",
        strategy="weighted_sum",
        surrogate_model=model, 
        input_columns=input_columns 
    )

    # 4. Get Predictions explicitly to fill DataFrame columns
    predictions, uncertainties, _ = model.predict_with_uncertainty(
        candidate_inputs, 
        input_columns=input_columns,
        num_samples=30
    )

    # 5. Map to columns
    for i, col in enumerate(target_columns):
        # Predictions
        if predictions.ndim == 1:
            candidate_df[col] = predictions
        else:
            candidate_df[col] = predictions[:, i]
            
        # Uncertainties (Broadcasting check)
        if uncertainties.ndim == 1:
             candidate_df[f"Uncertainty ({col})"] = uncertainties
        elif uncertainties.shape[1] == 1:
             candidate_df[f"Uncertainty ({col})"] = uncertainties[:, 0]
        else:
             candidate_df[f"Uncertainty ({col})"] = uncertainties[:, i]

    # 6. Assign Utility Scores
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

    # 7. Calculate Aggregate Uncertainty
    if uncertainties.ndim > 1 and uncertainties.shape[1] > 1:
        candidate_df["Uncertainty"] = np.mean(uncertainties, axis=1)
    else:
        candidate_df["Uncertainty"] = uncertainties.flatten()

    # 8. Calculate Novelty
    X_candidate_np = candidate_inputs.values
    X_labeled_np = labeled_data[input_columns].values
    novelty_scores = calculate_novelty(X_candidate_np, X_labeled_np)
    candidate_df["Novelty"] = novelty_scores

    # 9. Exploration / Exploitation
    candidate_df["Exploration"] = candidate_df["Uncertainty"] * (1.0 + curiosity)
    candidate_df["Exploitation"] = candidate_df[target_columns].mean(axis=1)

    # 10. Selection Flag
    candidate_df["Selected for Testing"] = False
    if not candidate_df["Utility"].empty:
        max_utility_idx = candidate_df["Utility"].idxmax()
        candidate_df.loc[max_utility_idx, "Selected for Testing"] = True

    # 11. Final Sort
    result_df = candidate_df.sort_values(by="Utility", ascending=False).reset_index(drop=True)

    return result_df