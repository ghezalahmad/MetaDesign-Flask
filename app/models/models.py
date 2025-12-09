import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
# Import both scalers, but use StandardScaler below
from sklearn.preprocessing import RobustScaler, StandardScaler 

# Internal Imports
from app.models.bayesian_optimizer import multi_objective_bayesian_optimization
from app.utils.utils import calculate_novelty

# ==========================================
# 1. THE MAML MODEL CLASS (Simplified Architecture)
# ==========================================

class MAMLModel(nn.Module):
    # Hidden size of 64 balances model capacity and generalization
    def __init__(self, input_size, output_size, hidden_size=64, num_layers=2, dropout_rate=0.3):
        super(MAMLModel, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # Attributes expected by BayesianOptimizer
        self.is_trained = False
        self.scaler_x = None
        self.scaler_y = None
        # NEW: Attributes to store min/max of training targets for potential clipping
        self.y_min_train = None
        self.y_max_train = None
        
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

    def predict_with_uncertainty(self, X, input_columns=None, num_samples=50):
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
        
        # Soft clipping: Allow predictions up to 50% beyond training range
        # This allows exploration while preventing unrealistic extrapolation
        if self.y_min_train is not None and self.y_max_train is not None:
            buffer_factor = 0.5  # 50% beyond training range
            range_size = self.y_max_train - self.y_min_train
            soft_min = self.y_min_train - (range_size * buffer_factor)
            soft_max = self.y_max_train + (range_size * buffer_factor)
            
            mean_preds_original = np.clip(mean_preds_original, soft_min, soft_max)

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
# 1.5 MULTI-TARGET WRAPPER (Like Lolopy)
# Trains separate NN per target for better diversity
# ==========================================

class MAMLMultiTargetWrapper:
    """
    Wrapper that trains separate MAMLModel per target (like Lolopy RF).
    This prevents prediction collapse common with multi-output NNs.
    """
    def __init__(self, input_size, target_columns, hidden_size=64, num_layers=2, dropout_rate=0.3):
        self.input_size = input_size
        self.target_columns = target_columns
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # One model per target
        self.models = []
        self.scalers_x = []
        self.scalers_y = []
        self.is_trained = False
        
    def train(self, X, y, epochs=100):
        """Train one NN per target column."""
        from sklearn.preprocessing import StandardScaler
        
        if isinstance(y, pd.DataFrame):
            y_np = y.values
        else:
            y_np = y
            
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)
            
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = X
            
        self.models = []
        self.scalers_x = []
        self.scalers_y = []
        
        num_targets = y_np.shape[1]
        print(f"MAMLMultiTarget: Training {num_targets} separate models...")
        
        for i in range(num_targets):
            # Create single-output model
            model = MAMLModel(
                input_size=self.input_size,
                output_size=1,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout_rate=self.dropout_rate
            )
            
            # Fit scalers
            scaler_x = StandardScaler().fit(X_np)
            scaler_y = StandardScaler().fit(y_np[:, i:i+1])
            
            model.scaler_x = scaler_x
            model.scaler_y = scaler_y
            model.y_min_train = np.min(y_np[:, i])
            model.y_max_train = np.max(y_np[:, i])
            
            # Train with mini MAML loop
            self._train_single_model(model, X_np, y_np[:, i], scaler_x, scaler_y, epochs)
            
            model.is_trained = True
            self.models.append(model)
            self.scalers_x.append(scaler_x)
            self.scalers_y.append(scaler_y)
            
        self.is_trained = True
        print(f"MAMLMultiTarget: All {num_targets} models trained.")
        
    def _train_single_model(self, model, X_np, y_target, scaler_x, scaler_y, epochs):
        """Train a single target model using standard NN training."""
        X_scaled = scaler_x.transform(X_np)
        y_scaled = scaler_y.transform(y_target.reshape(-1, 1))
        
        inputs = torch.tensor(X_scaled, dtype=torch.float32)
        targets = torch.tensor(y_scaled, dtype=torch.float32)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = model(inputs)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            
        model.eval()
        
    def predict_with_uncertainty(self, X, input_columns=None, num_samples=50):
        """Aggregate predictions from all per-target models."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
            
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = X
            
        all_preds = []
        all_stds = []
        
        for i, model in enumerate(self.models):
            # Use the per-model prediction
            preds, stds, _ = model.predict_with_uncertainty(X, input_columns, num_samples)
            all_preds.append(preds[:, 0:1])
            all_stds.append(stds[:, 0:1])
            
        final_preds = np.hstack(all_preds)
        final_stds = np.hstack(all_stds)
        
        return final_preds, final_stds, None


# ==========================================
# 2. TRAINING LOGIC (Scaler and Min/Max Storage Updated)
# ==========================================

def meta_train(meta_model: MAMLModel, data: pd.DataFrame, input_columns: list, target_columns: list,
               epochs: int = 100, inner_lr: float = 0.005, outer_lr: float = 0.001,
               num_tasks: int = 4, inner_lr_decay: float = 0.95, curiosity: float = 0,
               min_samples_per_task: int = 3, early_stopping_patience: int = 20):
    
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

    # --- SCALING CHANGE: Using StandardScaler for better NN stability ---
    scaler_inputs = StandardScaler().fit(labeled_data[input_columns])
    scaler_targets = StandardScaler().fit(labeled_data[target_columns])
    
    # Attach scalers immediately
    meta_model.scaler_x = scaler_inputs
    meta_model.scaler_y = scaler_targets

    # --- NEW: Store min/max for clipping ---
    y_train_orig = labeled_data[target_columns].values
    meta_model.y_min_train = np.min(y_train_orig, axis=0)
    meta_model.y_max_train = np.max(y_train_orig, axis=0)
    # ---------------------------------------
    
    inputs = torch.tensor(scaler_inputs.transform(labeled_data[input_columns]), dtype=torch.float32)
    targets = torch.tensor(scaler_targets.transform(labeled_data[target_columns]), dtype=torch.float32)
    
    loss_function = nn.SmoothL1Loss() if len(labeled_data) >= 10 else nn.MSELoss()
    # batch_size is defined but not used here, keep it for reference
    # batch_size = max(2, min(8, len(inputs) // 4)) 
    
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

def evaluate_maml(model, data: pd.DataFrame, input_columns: list,
                  target_columns: list, curiosity: float, weights_targets: np.ndarray,
                  max_or_min_targets: list[str]):
    """
    Evaluates candidates using MAML with per-target training (like Lolopy).
    Uses MAMLMultiTargetWrapper for better prediction diversity.
    """
    
    # 1. Split labeled vs candidate rows
    labeled_data = data.dropna(subset=target_columns)
    candidate_df = data[data[target_columns[0]].isnull()].copy()

    if candidate_df.empty:
        return pd.DataFrame()

    # 2. Use MAMLMultiTargetWrapper (like Lolopy RF)
    if not getattr(model, 'is_trained', False):
        print("evaluate_maml: Using MAMLMultiTargetWrapper for per-target training...")
        wrapper = MAMLMultiTargetWrapper(
            input_size=len(input_columns),
            target_columns=target_columns,
            hidden_size=64,
            num_layers=2,
            dropout_rate=0.3
        )
        wrapper.train(
            labeled_data[input_columns],
            labeled_data[target_columns],
            epochs=100
        )
        model = wrapper

    candidate_inputs = candidate_df[input_columns]

    # 3. Get Predictions
    predictions, uncertainties, _ = model.predict_with_uncertainty(
        candidate_inputs, 
        input_columns=input_columns,
        num_samples=50
    )

    # 4. Map to columns
    for i, col in enumerate(target_columns):
        if predictions.ndim == 1:
            candidate_df[col] = predictions
        else:
            candidate_df[col] = predictions[:, i]
            
        if uncertainties.ndim == 1:
             candidate_df[f"Uncertainty ({col})"] = uncertainties
        elif uncertainties.shape[1] == 1:
             candidate_df[f"Uncertainty ({col})"] = uncertainties[:, 0]
        else:
             candidate_df[f"Uncertainty ({col})"] = uncertainties[:, i]

    # 5. WEBSLAMD-EXACT UTILITY CALCULATION
    labels_mean = labeled_data[target_columns].mean(skipna=True)
    labels_std = labeled_data[target_columns].std(skipna=True).replace(0, 1)
    
    n_targets = len(target_columns)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if uncertainties.ndim == 1:
        uncertainties = uncertainties.reshape(-1, 1)
        
    preds_norm = np.zeros_like(predictions, dtype=float)
    unc_norm = np.zeros_like(uncertainties, dtype=float)
    
    for i, col in enumerate(target_columns):
        mean_val = labels_mean.iloc[i]
        std_val = labels_std.iloc[i]
        preds_norm[:, i] = (predictions[:, i] - mean_val) / std_val
        if max_or_min_targets[i].lower() == "min":
            preds_norm[:, i] *= -1
        preds_norm[:, i] *= weights_targets[i]
        unc_norm[:, i] = uncertainties[:, i] / std_val
        unc_norm[:, i] *= weights_targets[i]
    
    utility_scores = preds_norm.sum(axis=1) + curiosity * unc_norm.sum(axis=1)
    candidate_df["Utility"] = utility_scores
    candidate_df["Utility"] = pd.to_numeric(candidate_df["Utility"], errors="coerce").fillna(0.0).astype(float)

    # 6. Calculate Aggregate Uncertainty
    if uncertainties.ndim > 1 and uncertainties.shape[1] > 1:
        candidate_df["Uncertainty"] = np.mean(uncertainties, axis=1)
    else:
        candidate_df["Uncertainty"] = uncertainties.flatten()

    # 7. Calculate Novelty
    X_candidate_np = candidate_inputs.values
    X_labeled_np = labeled_data[input_columns].values
    novelty_scores = calculate_novelty(X_candidate_np, X_labeled_np)
    candidate_df["Novelty"] = novelty_scores

    # 8. Selection Flag
    candidate_df["Selected for Testing"] = False
    if not candidate_df["Utility"].empty:
        max_utility_idx = candidate_df["Utility"].idxmax()
        candidate_df.loc[max_utility_idx, "Selected for Testing"] = True

    # 9. Final Sort
    result_df = candidate_df.sort_values(by="Utility", ascending=False).reset_index(drop=True)

    return result_df