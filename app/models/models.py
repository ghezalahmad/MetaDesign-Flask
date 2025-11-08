import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from scipy.stats import norm
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from app.utils.utils import calculate_utility, calculate_novelty, select_acquisition_function

class MAMLModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=3, dropout_rate=0.3):
        super(MAMLModel, self).__init__()
        
        # Improved architecture with residual connections and batch normalization
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.input_bn = nn.BatchNorm1d(hidden_size)
        self.input_dropout = nn.Dropout(dropout_rate)
        
        # Middle layers with residual connections
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.bn_layers.append(nn.BatchNorm1d(hidden_size))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Initialize weights with He initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Input layer
        x = self.input_layer(x)
        if x.shape[0] > 1:  # Only apply batch norm for batch size > 1
            x = self.input_bn(x)
        x = torch.relu(x)
        x = self.input_dropout(x)
        
        # Middle layers with residual connections
        for i in range(len(self.layers)):
            identity = x
            x = self.layers[i](x)
            if x.shape[0] > 1:
                x = self.bn_layers[i](x)
            x = torch.relu(x)
            x = self.dropout_layers[i](x)
            
            # Add residual connection if shapes match
            if identity.shape == x.shape:
                x = x + identity
        
        # Output layer
        output = self.output_layer(x)
        return torch.nn.functional.softplus(output)

    def predict_with_uncertainty(self, X_df_original_scale: pd.DataFrame, input_columns: list[str], num_samples=30, dropout_rate=0.3):
        """
        Makes predictions and estimates uncertainty using MC dropout.
        Assumes the model has been trained and scalers (self.scaler_x, self.scaler_y) are set.

        Args:
            X_df_original_scale (pd.DataFrame): DataFrame with input features in their original scale.
            input_columns (list[str]): List of input feature column names to use from X_df_original_scale.
            num_samples (int): Number of Monte Carlo samples for uncertainty estimation.
            dropout_rate (float): Dropout rate to use during MC sampling.

        Returns:
            tuple: (mean_predictions_original_scale, std_dev_original_scale, Optional[mc_samples_original_scale])
                   mean_predictions_original_scale (np.ndarray): Predictions in the original target scale.
                   std_dev_original_scale (np.ndarray): Standard deviation of predictions in original target scale.
                   mc_samples_original_scale (np.ndarray, optional): MC samples in original target scale. Shape (num_mc_samples, num_input_points, num_targets).
        """
        if not getattr(self, 'is_trained', False) or self.scaler_x is None or self.scaler_y is None:
            raise RuntimeError("Model is not trained yet or scalers are missing. Call meta_train first.")

        # self.eval() # eval mode is set inside calculate_uncertainty_ensemble if needed for dropout
        # For mean prediction, it should be eval. For MC samples, it's train.
        # Let's get mean pred first in eval mode.
        self.eval()
        X_processed = X_df_original_scale[input_columns]
        X_scaled_np = self.scaler_x.transform(X_processed)
        X_tensor = torch.tensor(X_scaled_np, dtype=torch.float32)

        with torch.no_grad():
            mean_predictions_scaled = self(X_tensor).numpy()

        # Get uncertainty (variance, scaled) AND MC samples using existing ensemble/dropout method
        # calculate_uncertainty_ensemble sets model.train() internally for dropout
        variance_scaled_per_sample, mc_samples_scaled_torch = calculate_uncertainty_ensemble(
            self, X_tensor, num_samples=num_samples, dropout_rate=dropout_rate,
            use_input_perturbation=getattr(self, 'use_input_perturbation_for_uncertainty', False), # Assuming this attribute could be set
            noise_scale=getattr(self, 'noise_scale_for_uncertainty', 0.05)
        ) # variance_scaled_per_sample is (n_points,) or (n_points,1), mc_samples_scaled_torch is (n_mc_samples, n_points, n_targets)

        # Inverse transform mean predictions
        mean_predictions_original_scale = self.scaler_y.inverse_transform(mean_predictions_scaled)

        # Inverse transform MC samples
        # mc_samples_scaled_torch is (num_samples, num_X_points, num_targets)
        # scaler_y.inverse_transform expects (num_X_points, num_targets)
        mc_samples_original_scale_list = []
        if mc_samples_scaled_torch is not None:
            mc_samples_scaled_np = mc_samples_scaled_torch.cpu().numpy()
            for i in range(mc_samples_scaled_np.shape[0]): # Iterate over MC samples
                sample_slice_scaled = mc_samples_scaled_np[i, :, :]
                mc_samples_original_scale_list.append(self.scaler_y.inverse_transform(sample_slice_scaled))
            mc_samples_original_scale = np.stack(mc_samples_original_scale_list, axis=0)
        else:
            mc_samples_original_scale = None


        # Inverse transform uncertainty (variance -> std_dev)
        # scaler_y.scale_ is an array of scales for each target feature
        # Variance scales by scale_factor^2
        if variance_scaled_per_sample.ndim == 1: # If already averaged over targets
            variance_scaled_per_sample = variance_scaled_per_sample.reshape(-1, 1)

        # If scaler_y was fit on multiple targets, self.scaler_y.scale_ will be an array.
        # If only one target, it's a scalar. We need to handle this for broadcasting.
        scale_sq = self.scaler_y.scale_ ** 2
        if variance_scaled_per_sample.shape[1] == 1 and len(scale_sq) > 1:
            # If uncertainty is a single column but multiple targets, assume it's an average
            # and we can't easily scale it back per target.
            # This implies calculate_uncertainty_ensemble should ideally return per-target variance if possible
            # For now, if it's a single column, we might have to use an average scale factor or just one.
            # Let's assume calculate_uncertainty_ensemble already gives mean variance if multi-target.
            # And if single target, scaler_y.scale_ is scalar.
            # If calculate_uncertainty_ensemble returns (n_samples, n_targets) variance:
            # variance_original_scale = variance_scaled_per_sample * scale_sq
            # std_dev_original_scale = np.sqrt(variance_original_scale)
            # std_dev_original_scale_per_sample = np.mean(std_dev_original_scale, axis=1).reshape(-1, 1)

            # Simpler assumption: if variance_scaled_per_sample is (n_samples, 1),
            # and if scaler_y was multi-target, we use mean of scale_sq. This is an approximation.
            # A better approach: calculate_uncertainty_ensemble should return variance per target.
            # The current calculate_uncertainty_ensemble averages multi-target uncertainty.
            # So, we use the mean of squared scales.
            mean_scale_sq = np.mean(scale_sq) if len(scale_sq) > 0 else 1.0
            variance_original_scale = variance_scaled_per_sample * mean_scale_sq

        elif variance_scaled_per_sample.shape[1] == len(scale_sq): # Per-target variance
             variance_original_scale = variance_scaled_per_sample * scale_sq
        else: # Mismatch, fallback to mean scale or error
            print("Mismatch in uncertainty and target scaler dimensions. Using mean scale for uncertainty.")
            mean_scale_sq = np.mean(scale_sq) if len(scale_sq) > 0 else 1.0
            variance_original_scale = variance_scaled_per_sample * mean_scale_sq


        std_dev_original_scale = np.sqrt(np.maximum(variance_original_scale, 1e-12)) # ensure non-negative before sqrt

        # Ensure std_dev_original_scale is (n_samples, 1) by averaging if it's multi-column
        if std_dev_original_scale.ndim > 1 and std_dev_original_scale.shape[1] > 1:
            std_dev_original_scale_per_sample = np.mean(std_dev_original_scale, axis=1).reshape(-1, 1)
        else:
            std_dev_original_scale_per_sample = std_dev_original_scale.reshape(-1, 1)
        # Ensuring this return statement is clean
        return mean_predictions_original_scale, std_dev_original_scale_per_sample, mc_samples_original_scale


def meta_train(meta_model: MAMLModel, data: pd.DataFrame, input_columns: list[str], target_columns: list[str],
               epochs: int = 100, inner_lr: float = 0.01, outer_lr: float = 0.001,
               num_tasks: int = 4, inner_lr_decay: float = 0.95, curiosity: float = 0,
               min_samples_per_task: int = 3, early_stopping_patience: int = 10) -> tuple[MAMLModel, RobustScaler | None, RobustScaler | None]:
    """
    Trains a MAMLModel using meta-learning.

    The function preprocesses data, sets up optimizers and schedulers,
    and then runs the meta-training loop. Inside the loop, it samples tasks,
    adapts a copy of the model on the support set of each task, and computes
    a meta-loss on the query set to update the original meta-model.
    Includes features like adaptive loss, data tiling for small datasets,
    episodic memory, and early stopping.

    Args:
        meta_model: The MAMLModel instance to be trained.
        data: DataFrame containing all training data.
        input_columns: List of column names for input features.
        target_columns: List of column names for target properties.
        epochs: Number of meta-training epochs.
        inner_lr: Initial learning rate for the inner loop (task adaptation).
        outer_lr: Learning rate for the outer loop (meta-model update).
        num_tasks: Number of tasks to sample per epoch.
        inner_lr_decay: Decay rate for the inner loop learning rate.
        curiosity: Factor to adjust loss, potentially encouraging exploration.
        min_samples_per_task: Minimum samples required to form a task.
        early_stopping_patience: Number of epochs with no improvement to wait before stopping.

    Returns:
        A tuple containing:
            - The trained MAMLModel.
            - The scaler used for input features (RobustScaler).
            - The scaler used for target properties (RobustScaler).
        Returns (meta_model, None, None) if training cannot proceed (e.g., no labeled data).
    """
    # Configure optimizers with weight decay to prevent overfitting on small datasets
    optimizer = optim.AdamW(meta_model.parameters(), lr=outer_lr, 
                           weight_decay=1e-3, betas=(0.9, 0.999))
    
    # Cosine annealing scheduler with warm restarts for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=outer_lr * 0.1
    )
    
    # Adaptive loss function based on dataset size
    # Extract labeled data
    labeled_data = data.dropna(subset=target_columns).copy()

    # Sort data by the first target variable to create more structured tasks
    if not labeled_data.empty and len(target_columns) > 0:
        labeled_data = labeled_data.sort_values(by=target_columns[0]).reset_index(drop=True)

    # ✅ Ensure target columns are numeric
    for col in target_columns:
        labeled_data[col] = pd.to_numeric(labeled_data[col], errors='coerce')

    # ✅ Drop rows with NaN values (if conversion failed for some rows)
    labeled_data = labeled_data.dropna(subset=target_columns)
    
    # Check if we have enough labeled data
    if labeled_data.empty:
        print("⚠️ No labeled data available for training.")
        return meta_model, None, None
    
    # ✅✅ NEW - Enhanced handling for extremely small datasets
    MINIMUM_DATA_POINTS = 8
    if len(labeled_data) > 0 and len(labeled_data) < MINIMUM_DATA_POINTS:
        print(f"🔄 Few labeled samples ({len(labeled_data)}). Applying data tiling for more robust evaluation.")
        tile_factor = int(np.ceil(MINIMUM_DATA_POINTS / len(labeled_data)))
        labeled_data_eval = pd.concat([labeled_data] * tile_factor, ignore_index=True)
        
        for col in input_columns:
            std = labeled_data[col].std()
            if std > 0:
                noise = np.random.normal(0, std * 0.05, size=len(labeled_data_eval))
                labeled_data_eval[col] += noise
        for col in target_columns:
            std = labeled_data[col].std()
            if std > 0:
                noise = np.random.normal(0, max(std * 0.02, 1e-6), size=len(labeled_data_eval))
                labeled_data_eval[col] += noise

            
        print(f"✅ Data tiled from {len(labeled_data)} to {len(labeled_data_eval)} samples.")
        labeled_data = labeled_data_eval  # 🔄 Apply tiling + noise

      
    # ✅ Now, safely compute min and max
    print(f"Target range: {labeled_data[target_columns].min().values} to {labeled_data[target_columns].max().values}")
    if len(labeled_data) < 10:
        # For very small datasets, use MSE which is more stable
        loss_function = nn.MSELoss()
        print("Using MSE loss for small dataset stability")
    else:
        # For larger datasets, use Huber loss for robustness to outliers
        loss_function = nn.SmoothL1Loss()
        print("Using Huber loss for robustness to outliers")

    # Validate if we have enough samples for few-shot learning
    if len(labeled_data) < min_samples_per_task * 2:
        print(f"⚠️ Limited labeled samples ({len(labeled_data)}) - using few-shot adaptations.")
        # Reduce task complexity for very small datasets
        num_tasks = max(2, min(num_tasks, len(labeled_data) // 2))
    
    # Use RobustScaler for better handling of outliers in small datasets
    scaler_inputs = RobustScaler().fit(labeled_data[input_columns])
    scaler_targets = RobustScaler().fit(labeled_data[target_columns])
    
    # Transform the data
    inputs = torch.tensor(scaler_inputs.transform(labeled_data[input_columns]), dtype=torch.float32)
    targets = torch.tensor(scaler_targets.transform(labeled_data[target_columns]), dtype=torch.float32)
    
    # Ensure we have data to work with
    if len(inputs) == 0 or len(targets) == 0:
        print("❌ No valid data available for training after preprocessing.")
        return meta_model, scaler_inputs, scaler_targets
    
    # Dynamic batch size based on available data
    batch_size = max(2, min(8, len(inputs) // 4))
    
    # Adjust number of tasks based on available data
    num_tasks = max(2, min(num_tasks, len(inputs) // batch_size))
    
    # Cross-validation strategy for small datasets
    kf = KFold(n_splits=min(5, len(labeled_data)), shuffle=True)
    
    # Initialize tracking variables for early stopping
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Set minimum learning rate for inner loop
    min_inner_lr = inner_lr * 0.1
    current_inner_lr = inner_lr
    
    # Create a dictionary to store validation info for each fold
    fold_metrics = {}
    
    # Create episodic memory for knowledge retention
    episodic_memory = []
    episodic_memory_size = min(10, len(inputs))
    
    print(f"Starting meta-training with {len(labeled_data)} labeled samples")
    print(f"Batch size: {batch_size}, Number of tasks: {num_tasks}")
    
    # Main training loop
    for epoch in range(epochs):
        meta_model.train()
        meta_losses = []
        
        # Decay inner learning rate with a smoother curve for material properties
        current_inner_lr = max(inner_lr * (inner_lr_decay ** (epoch / 10)), min_inner_lr)
        
        # Dynamically adjust inner loop steps based on epoch and dataset size
        inner_loop_steps = max(2, min(5, int(len(inputs) / batch_size / 2)))
        if epoch > epochs // 2:
            inner_loop_steps = max(inner_loop_steps, 3)  # Increase steps in later epochs
        
        # New task creation: Sliding window over sorted data
        task_indices = []
        window_size = len(inputs) // num_tasks
        for i in range(num_tasks):
            # Define the window for the current task
            start_idx = i * window_size
            end_idx = start_idx + window_size
            
            # The validation set is the current window
            val_idx = np.arange(start_idx, end_idx)

            # The training set is everything outside the window
            train_idx = np.setdiff1d(np.arange(len(inputs)), val_idx)

            if len(train_idx) >= min_samples_per_task and len(val_idx) >= 1:
                task_indices.append((train_idx, val_idx))
        
        # Process each task
        for task_idx, (support_indices, query_indices) in enumerate(task_indices[:num_tasks]):
            # Convert to tensors
            support_indices = torch.tensor(support_indices)
            query_indices = torch.tensor(query_indices)
            
            # Get support and query data
            support_inputs = inputs[support_indices]
            support_targets = targets[support_indices]
            query_inputs = inputs[query_indices]
            query_targets = targets[query_indices]
            
            # Add samples from episodic memory for knowledge retention
            if epoch > 0 and len(episodic_memory) > 0:
                memory_samples = min(len(episodic_memory), batch_size // 2)
                memory_indices = np.random.choice(len(episodic_memory), size=memory_samples, replace=False)
                
                for idx in memory_indices:
                    mem_input, mem_target = episodic_memory[idx]
                    support_inputs = torch.cat([support_inputs, mem_input.unsqueeze(0)], dim=0)
                    support_targets = torch.cat([support_targets, mem_target.unsqueeze(0)], dim=0)
            
            # Create a copy of the meta-model for this task
            task_model = copy.deepcopy(meta_model)
            
            # Adaptive optimizer selection based on sample size
            if len(support_inputs) < 10:
                task_optimizer = optim.SGD(task_model.parameters(), 
                                          lr=current_inner_lr, 
                                          momentum=0.9, 
                                          weight_decay=1e-4)
            else:
                task_optimizer = optim.Adam(task_model.parameters(), 
                                           lr=current_inner_lr, 
                                           weight_decay=1e-4)
            
            # Inner loop adaptation
            task_losses = []
            for step in range(inner_loop_steps):
                # For very small datasets, use all support data
                if len(support_inputs) <= batch_size:
                    batch_inputs = support_inputs
                    batch_targets = support_targets
                else:
                    # Sample a batch with importance weighting for difficult samples
                    if step > 0 and len(task_losses) > 0:
                        # Focus on difficult samples in later steps
                        with torch.no_grad():
                            difficulties = torch.abs(task_model(support_inputs) - support_targets).sum(dim=1)
                            probs = difficulties / difficulties.sum()
                            batch_indices = torch.multinomial(probs, min(batch_size, len(probs)), replacement=False)
                    else:
                        # Random sampling in first step
                        batch_indices = torch.randperm(len(support_inputs))[:batch_size]
                    
                    batch_inputs = support_inputs[batch_indices]
                    batch_targets = support_targets[batch_indices]
                
                # Forward pass
                task_predictions = task_model(batch_inputs)
                task_loss = loss_function(task_predictions, batch_targets)
                
                # Apply curiosity factor
                if curiosity != 0:
                    # Refined curiosity scaling for material discovery domain
                    scaling_factor = 1.0 + (0.1 * curiosity * (1.0 - (step / inner_loop_steps)))
                    task_loss = task_loss * scaling_factor
                
                # Backward pass
                task_optimizer.zero_grad()
                task_loss.backward()
                
                # Gradient clipping to stabilize training with small batches
                torch.nn.utils.clip_grad_norm_(task_model.parameters(), max_norm=0.5)
                task_optimizer.step()
                
                task_losses.append(task_loss.item())
            
            # Evaluate on query set
            task_model.eval()
            query_predictions = task_model(query_inputs)
            query_loss = loss_function(query_predictions, query_targets)
            
            # Apply curiosity to query loss as well for consistent optimization
            if curiosity != 0:
                query_loss = query_loss * (1 + 0.1 * curiosity)
            
            # Store good examples in episodic memory
            with torch.no_grad():
                for i in range(min(2, len(query_inputs))):
                    pred = task_model(query_inputs[i:i+1])
                    error = torch.abs(pred - query_targets[i:i+1]).mean().item()
                    
                    # Store examples with low error
                    if error < 0.2:
                        if len(episodic_memory) < episodic_memory_size:
                            episodic_memory.append((query_inputs[i], query_targets[i]))
                        else:
                            # Replace a random item
                            replace_idx = np.random.randint(0, len(episodic_memory))
                            episodic_memory[replace_idx] = (query_inputs[i], query_targets[i])
            
            # Meta-update (first-order approximation for efficiency)
            if task_idx == 0:
                meta_loss = query_loss
            else:
                meta_loss = meta_loss + query_loss
            
            # Store metrics
            fold_metrics[f"task_{task_idx}"] = {
                "support_size": len(support_inputs),
                "query_size": len(query_inputs),
                "final_task_loss": task_losses[-1] if task_losses else float('inf'),
                "query_loss": query_loss.item()
            }
            
            meta_losses.append(query_loss.item())
        
        # Average meta loss
        meta_loss = meta_loss / len(task_indices[:num_tasks])
        
        # Update meta-model
        optimizer.zero_grad()
        meta_loss.backward()
        
        # Gradient clipping for meta-update
        torch.nn.utils.clip_grad_norm_(meta_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update learning rate scheduler
        scheduler.step()
        
        # Calculate average loss for this epoch
        avg_meta_loss = np.mean(meta_losses) if meta_losses else float('inf')
        
        # Early stopping check
        if avg_meta_loss < best_loss:
            best_loss = avg_meta_loss
            best_model_state = copy.deepcopy(meta_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs} - Meta Loss: {avg_meta_loss:.4f}, Best Loss: {best_loss:.4f}")
            print(f"Inner LR: {current_inner_lr:.6f}, Outer LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Dynamic patience based on dataset size and epoch
        if len(labeled_data) < 20:
            dynamic_patience = max(5, early_stopping_patience - (epoch // 10))
        else:
            dynamic_patience = early_stopping_patience
        
        # Check for early stopping
        if patience_counter >= dynamic_patience:
            print(f"🛑 Early stopping at epoch {epoch+1} due to {dynamic_patience} non-improving epochs.")
            break
    
    # Restore best model weights
    if best_model_state:
        meta_model.load_state_dict(best_model_state)
        print(f"✅ Restored best model with loss: {best_loss:.4f}")
    
    # Final evaluation on all labeled data
    meta_model.eval()
    with torch.no_grad():
        all_predictions = meta_model(inputs)
        final_loss = loss_function(all_predictions, targets).item()
    
    print(f"Final evaluation loss on all labeled data: {final_loss:.4f}")

    meta_model.is_trained = True
    meta_model.scaler_x = scaler_inputs
    meta_model.scaler_y = scaler_targets
    
    return meta_model, scaler_inputs, scaler_targets


def evaluate_maml(meta_model: MAMLModel, data: pd.DataFrame, input_columns: list[str], target_columns: list[str],
                  curiosity: float, weights: np.ndarray, max_or_min: list[str],
                  acquisition: str = None, min_labeled_samples: int = 5, dynamic_acquisition: bool = True) -> pd.DataFrame | None:
    """
    Evaluates a trained MAML model on unlabeled data to suggest new candidates.

    It handles predictions (with GP fallback for very small labeled sets),
    uncertainty estimation, novelty calculation, and utility computation using
    a specified acquisition function.

    Args:
        meta_model: The trained MAMLModel instance.
        data: DataFrame containing all data (labeled and unlabeled).
        input_columns: List of column names for input features.
        target_columns: List of column names for target properties.
        curiosity: Float indicating exploration vs. exploitation tendency.
        weights: Numpy array of weights for each target property in utility calculation.
        max_or_min: List of strings ('max' or 'min') indicating optimization direction for each target.
        acquisition: Name of the acquisition function to use (e.g., "UCB", "EI").
                     If None, selected dynamically based on curiosity and data size.
        min_labeled_samples: Minimum number of labeled samples required to use MAML directly;
                             below this, a Gaussian Process fallback is used.
        dynamic_acquisition: If True and acquisition is None, it's selected dynamically.

    Returns:
        A pandas DataFrame with suggestions ranked by utility, including predicted properties,
        uncertainty, novelty, and other metrics. Returns None if no unlabeled data.
    """
    # Get labeled and unlabeled data
    unlabeled_data = data[data[target_columns].isna().any(axis=1)].copy() # Use .copy() to avoid SettingWithCopyWarning
    labeled_data = data.dropna(subset=target_columns)

    if unlabeled_data.empty:
        print("No unlabeled samples available for evaluation.")
        return None

    print(f"Evaluating with {len(labeled_data)} labeled samples and {len(unlabeled_data)} unlabeled samples")

    # Set fallback strategy based on labeled data count
    use_fallback = len(labeled_data) < min_labeled_samples

    # Enhanced handling for very small datasets (tiling + noise)
    MINIMUM_DATA_POINTS = 8
    if len(labeled_data) > 0 and len(labeled_data) < MINIMUM_DATA_POINTS and not use_fallback:
        print(f"🔄 Few labeled samples ({len(labeled_data)}). Applying data tiling for robust evaluation.")
        tile_factor = int(np.ceil(MINIMUM_DATA_POINTS / len(labeled_data)))
        labeled_data_eval = pd.concat([labeled_data] * tile_factor, ignore_index=True)

        # Add noise only if std > 0
        for col in input_columns:
            std = labeled_data[col].std()
            if std > 0:
                noise = np.random.normal(0, std * 0.05, size=len(labeled_data_eval))
                labeled_data_eval[col] += noise

        for col in target_columns:
            std = labeled_data[col].std()
            if std > 0:
                noise = np.random.normal(0, max(std * 0.02, 1e-6), size=len(labeled_data_eval))
                labeled_data_eval[col] += noise

        print(f"✅ Data tiled from {len(labeled_data)} to {len(labeled_data_eval)} samples.")
    else:
        labeled_data_eval = labeled_data

    # Prediction using fallback GP or meta-model
    if use_fallback:
        print(f"Using Gaussian Process fallback strategy due to limited data ({len(labeled_data)} samples)")
        result_df = unlabeled_data.copy()
        all_predictions = np.zeros((len(unlabeled_data), len(target_columns)))
        all_uncertainties = np.zeros((len(unlabeled_data), len(target_columns)))

        for i, target_col in enumerate(target_columns):
            kernel = Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True,
                                          n_restarts_optimizer=5, random_state=42)
            X_train = labeled_data[input_columns].values
            y_train = labeled_data[target_col].values.reshape(-1, 1)
            gp.fit(X_train, y_train)
            X_test = unlabeled_data[input_columns].values
            y_pred, y_std = gp.predict(X_test, return_std=True)

            # Ensure GP predictions are non-negative, consistent with NN model outputs
            y_pred = np.maximum(y_pred, 0)

            all_predictions[:, i] = y_pred
            all_uncertainties[:, i] = y_std
            result_df[target_col] = y_pred

        uncertainty_scores = np.mean(all_uncertainties, axis=1).reshape(-1, 1)
    else:
        print("Using meta-trained model for predictions")
        scaler_inputs = RobustScaler().fit(labeled_data[input_columns])
        scaler_targets = RobustScaler().fit(labeled_data[target_columns])
        inputs_infer = scaler_inputs.transform(unlabeled_data[input_columns])
        inputs_infer_tensor = torch.tensor(inputs_infer, dtype=torch.float32)

        meta_model.eval()
        with torch.no_grad():
            predictions_scaled = meta_model(inputs_infer_tensor).numpy()

        # calculate_uncertainty_ensemble returns a tuple (uncertainty_np, predictions_stack)
        # We only need the first element here for uncertainty_scores.
        uncertainty_scores, _ = calculate_uncertainty_ensemble(
            meta_model, inputs_infer_tensor,
            num_samples=30, dropout_rate=0.3
        )

        predictions = scaler_targets.inverse_transform(predictions_scaled)

        # New safeguard: Clamp predictions to a reasonable range based on training data
        min_target_val = labeled_data[target_columns].min().values
        max_target_val = labeled_data[target_columns].max().values
        predictions = np.clip(predictions, min_target_val, max_target_val * 1.5) # Allow 50% extrapolation

        result_df = unlabeled_data.copy()
        for i, col in enumerate(target_columns):
            result_df[col] = predictions[:, i]

    # Novelty calculation
    X_labeled_scaled = scaler_inputs.transform(labeled_data[input_columns])
    novelty_scores = calculate_novelty(inputs_infer, X_labeled_scaled)

    if acquisition is None:
        acquisition = select_acquisition_function(curiosity, len(labeled_data))
    print(f"Using SLAMD-style acquisition function: {acquisition}")

    weights = np.clip(np.array(weights) + np.random.uniform(0.01, 0.1, size=len(weights)), 0.1, 1.0)
    if max_or_min is None or not isinstance(max_or_min, list):
        max_or_min = ['max'] * len(target_columns)

    # Using the consolidated calculate_utility from app.utils
    # thresholds parameter can be passed as None if not used here.
    # for_visualization is False as these are final utility scores for ranking.
    utility_scores = calculate_utility(
        predictions=(predictions if not use_fallback else all_predictions),
        uncertainties=uncertainty_scores,
        novelty=novelty_scores,
        curiosity=curiosity,
        weights=weights,
        max_or_min=max_or_min,
        thresholds=None,  # Or pass actual thresholds if they become available here
        acquisition=acquisition,
        for_visualization=False
    )
    # The calculate_utility function already applies log1p and clipping internally.

    result_df["Utility"] = utility_scores.flatten()
    result_df["Uncertainty"] = np.clip(uncertainty_scores.flatten(), 1e-6, None) # Ensure uncertainty_scores is flattened
    result_df["Novelty"] = novelty_scores.flatten()
    result_df["Exploration"] = result_df["Uncertainty"] * result_df["Novelty"]
    result_df["Exploitation"] = 1.0 - result_df["Uncertainty"]

    result_df = result_df.sort_values(by="Utility", ascending=False)
    result_df["Selected for Testing"] = False
    result_df.iloc[0, result_df.columns.get_loc("Selected for Testing")] = True
    result_df.reset_index(drop=True, inplace=True)

    columns_to_front = ["Idx_Sample"] if "Idx_Sample" in result_df.columns else []
    metrics_columns = ["Utility", "Exploration", "Exploitation", "Novelty", "Uncertainty"]
    remaining_columns = [col for col in result_df.columns if col not in columns_to_front + metrics_columns + target_columns + ["Selected for Testing"]]
    new_column_order = columns_to_front + metrics_columns + target_columns + remaining_columns + ["Selected for Testing"]
    new_column_order = [col for col in new_column_order if col in result_df.columns]

    result_df = result_df[new_column_order]
    return result_df


def calculate_uncertainty_ensemble(model: nn.Module, X: torch.Tensor, num_samples: int = 30,
                                   dropout_rate: float = 0.3, use_input_perturbation: bool = False,
                                   noise_scale: float = 0.05) -> tuple[np.ndarray, torch.Tensor]:
    """
    Estimates prediction uncertainty using Monte Carlo (MC) dropout and optional input perturbation.

    The method involves performing multiple forward passes with dropout layers enabled
    (and optionally with noise added to inputs). The variance of these predictions
    is used as a measure of uncertainty. A small term proportional to the prediction
    magnitude is added to this variance.

    Args:
        model: The PyTorch nn.Module for which to estimate uncertainty.
               Dropout layers should be part of this model.
        X: Input data as a PyTorch Tensor.
        num_samples: The number of forward passes (MC samples) to perform.
        dropout_rate: The dropout rate to apply. This function attempts to set
                      the `p` attribute of `nn.Dropout` layers in the model.
        use_input_perturbation: If True, Gaussian noise is added to the inputs
                                for each MC sample.
        noise_scale: Standard deviation of the Gaussian noise for input perturbation.

    Returns:
        A tuple containing:
            - uncertainty_np (np.ndarray): An array of uncertainty scores (one per input point).
                                           If the model is multi-target, this is the mean
                                           uncertainty across targets.
            - predictions_stack (torch.Tensor): The stack of raw predictions from all MC samples.
                                                Shape: (num_samples, num_input_points, num_targets).
    """
    # Set model to training mode to enable dropout
    model.train()
    
    # For models with configurable dropout, try to update the rate
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout_rate
    
    # Collect predictions from multiple forward passes
    predictions = []
    for _ in range(num_samples):
        with torch.no_grad():
            if use_input_perturbation:
                # Add Gaussian noise to inputs
                noise = torch.normal(0, noise_scale, size=X.shape, device=X.device) # Ensure noise is on same device
                perturbed_input = X + noise
                pred = model(perturbed_input)
            else:
                pred = model(X)
            predictions.append(pred)
    
    # Stack predictions and compute statistics
    predictions_stack = torch.stack(predictions)
    mean_pred = torch.mean(predictions_stack, dim=0)
    var_pred = torch.var(predictions_stack, dim=0)
    
    # Combine epistemic (model) and aleatoric (data) uncertainty
    total_uncertainty = var_pred + torch.abs(mean_pred) * 0.05
    
    # Convert to numpy array and ensure proper shape for single uncertainty per sample
    uncertainty_np = total_uncertainty.numpy()
    
    # If we have multi-target predictions, average the uncertainties across targets
    if uncertainty_np.ndim > 1 and uncertainty_np.shape[1] > 1:
        uncertainty_np = np.mean(uncertainty_np, axis=1) # This is for the aggregated uncertainty output
    
    # Ensure uncertainty is non-negative
    uncertainty_np = np.maximum(uncertainty_np, 1e-9) # Adding a clip for safety

    # predictions_stack is (num_samples, num_X_points, num_targets)
    # For q-acquisition functions, we might need the raw samples.
    # The function's primary return is still the single uncertainty array.
    # We can return the stack as an optional second value if needed by a wrapper.
    # For now, let's modify predict_with_uncertainty to handle this.
    return uncertainty_np, predictions_stack # Return stack for potential use

# Note: The following function `calculate_utility_with_acquisition` was removed as part of consolidation.
# The primary utility calculation is now done by `app.utils.calculate_utility`.