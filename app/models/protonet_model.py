import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
import streamlit as st
from app.utils import (
    calculate_utility, calculate_novelty, # calculate_uncertainty removed
    initialize_scheduler, balance_exploration_exploitation
)
# calculate_uncertainty_ensemble will be imported from app.models where needed

class ProtoNetModel(nn.Module):
    def __init__(self, input_size, output_size, embedding_size=256, num_layers=3, dropout_rate=0.3):
        super(ProtoNetModel, self).__init__()
        
        # Add this line to define the use_layer_norm attribute
        self.use_layer_norm = True
        
        # Add this line to define the input_norm attribute
        self.input_norm = nn.LayerNorm(input_size)
        
        # Embedding network with residual connections
        self.encoder = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # First layer
        self.encoder.append(nn.Linear(input_size, embedding_size))
        self.norm_layers.append(nn.LayerNorm(embedding_size))
        self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers with residual connections
        for _ in range(num_layers - 2):
            self.encoder.append(nn.Linear(embedding_size, embedding_size))
            self.norm_layers.append(nn.LayerNorm(embedding_size))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        # Final embedding layer
        self.encoder.append(nn.Linear(embedding_size, embedding_size))
        self.norm_layers.append(nn.LayerNorm(embedding_size))
        self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        # Projection head for property prediction
        self.projector = nn.Linear(embedding_size, output_size)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x):
        """
        Encode inputs into the embedding space.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
            
        Returns:
        --------
        embedding : torch.Tensor
            Encoded embedding
        """
        if self.use_layer_norm:
            x = self.input_norm(x)
        
        h = x
        for i, (layer, norm, dropout) in enumerate(zip(self.encoder, self.norm_layers, self.dropout_layers)):
            # Forward pass through layer
            z = layer(h)
            z = norm(z)
            z = torch.relu(z)
            z = dropout(z)
            
            # Apply residual connection if shapes match
            if i > 0 and h.shape == z.shape:
                h = h + z
            else:
                h = z
        
        return h
    
    def forward(self, x):
        """
        Forward pass: encode and project to property space.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
            
        Returns:
        --------
        properties : torch.Tensor
            Predicted material properties
        """
        # Get embedding
        embedding = self.encode(x)
        
        # Project to property space with softplus activation for positive properties
        return torch.nn.functional.softplus(self.projector(embedding))

    def predict_with_uncertainty(self, X_df_original_scale: pd.DataFrame, input_columns: list[str], num_perturbations=50, noise_scale=0.1, dropout_rate=0.3):
        """
        Makes predictions and estimates uncertainty.
        Assumes the model has been trained and scalers (self.scaler_x, self.scaler_y) are set.
        Uses the model's direct forward pass for mean prediction and app.utils.calculate_uncertainty for uncertainty.

        Args:
            X_df_original_scale (pd.DataFrame): DataFrame with input features in their original scale.
            input_columns (list[str]): List of input feature column names to use.
            num_perturbations (int): Number of Monte Carlo samples for uncertainty estimation.
            noise_scale (float): Scale of Gaussian noise for input perturbation for uncertainty.
            dropout_rate (float): Dropout rate to use during MC sampling for uncertainty.

        Returns:
            tuple: (mean_predictions_original_scale, std_dev_original_scale_per_sample, Optional[mc_samples_original_scale])
                   mean_predictions_original_scale (np.ndarray): Predictions in the original target scale.
                   std_dev_original_scale_per_sample (np.ndarray): Standard deviation of predictions (per sample) in original target scale.
                   mc_samples_original_scale (np.ndarray, optional): MC samples in original target scale.
        """
        if not getattr(self, 'is_trained', False) or self.scaler_x is None or self.scaler_y is None:
            raise RuntimeError("Model is not trained yet or scalers are missing. Call protonet_train first.")

        self.eval() # Ensure model is in eval mode for mean prediction
        X_processed = X_df_original_scale[input_columns]
        X_scaled_np = self.scaler_x.transform(X_processed)
        X_tensor = torch.tensor(X_scaled_np, dtype=torch.float32)

        with torch.no_grad():
            mean_predictions_scaled = self(X_tensor).numpy()

        mean_predictions_original_scale = self.scaler_y.inverse_transform(mean_predictions_scaled)

        from app.models import calculate_uncertainty_ensemble
        variance_scaled_per_sample, mc_samples_scaled_torch = calculate_uncertainty_ensemble(
            model=self, X=X_tensor,
            num_samples=num_perturbations,
            dropout_rate=dropout_rate,
            use_input_perturbation=True, # ProtoNet model was using this
            noise_scale=noise_scale
        )
        if variance_scaled_per_sample.ndim == 1:
            variance_scaled_per_sample = variance_scaled_per_sample.reshape(-1, 1)

        mc_samples_original_scale = None
        if mc_samples_scaled_torch is not None:
            mc_samples_scaled_np = mc_samples_scaled_torch.cpu().numpy()
            mc_samples_original_scale_list = []
            for i in range(mc_samples_scaled_np.shape[0]):
                sample_slice_scaled = mc_samples_scaled_np[i, :, :]
                mc_samples_original_scale_list.append(self.scaler_y.inverse_transform(sample_slice_scaled))
            mc_samples_original_scale = np.stack(mc_samples_original_scale_list, axis=0)

        # Inverse transform uncertainty (variance -> std_dev)
        scale_sq = self.scaler_y.scale_ ** 2

        if variance_scaled_per_sample.shape[1] == 1: # As returned by util_calc_uncertainty
            mean_scale_sq = np.mean(scale_sq) if len(scale_sq) > 0 else 1.0
            variance_original_scale = variance_scaled_per_sample * mean_scale_sq
        else:
            # This case should ideally not be hit if util_calc_uncertainty behaves as expected
            st.warning("Unexpected shape for variance_scaled_per_sample in ProtoNetModel. Using mean scale.")
            mean_scale_sq = np.mean(scale_sq) if len(scale_sq) > 0 else 1.0
            variance_original_scale = variance_scaled_per_sample * mean_scale_sq

        std_dev_original_scale_per_sample = np.sqrt(np.maximum(variance_original_scale, 1e-12))

        return mean_predictions_original_scale, std_dev_original_scale_per_sample.reshape(-1, 1), mc_samples_original_scale


def euclidean_distance(x, y):
    """
    Compute pairwise Euclidean distance between two sets of vectors.
    
    Parameters:
    -----------
    x : torch.Tensor
        First set of vectors [n, d]
    y : torch.Tensor
        Second set of vectors [m, d]
        
    Returns:
    --------
    distances : torch.Tensor
        Pairwise distances [n, m]
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    
    return torch.sqrt(torch.pow(x - y, 2).sum(2) + 1e-8)


def compute_prototypes(embeddings, targets, num_clusters=None, use_kmeans=False):
    """
    Compute prototypes from embeddings and targets.
    
    Parameters:
    -----------
    embeddings : torch.Tensor
        Embeddings from the model
    targets : torch.Tensor
        Target values corresponding to embeddings
    num_clusters : int
        Number of clusters for K-means (or None to use mean)
    use_kmeans : bool
        Whether to use K-means clustering
        
    Returns:
    --------
    prototypes : torch.Tensor
        Prototype embeddings
    prototype_targets : torch.Tensor
        Target values corresponding to prototypes
    """
    if use_kmeans and num_clusters and num_clusters < len(embeddings) // 2:
        # Use K-means clustering to find prototypes
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings.detach().numpy())
        
        # Initialize prototype lists
        prototypes = []
        prototype_targets = []
        
        # Compute prototype as mean of embeddings in each cluster
        for cluster_idx in range(num_clusters):
            cluster_mask = (cluster_labels == cluster_idx)
            
            # Skip empty clusters
            if not np.any(cluster_mask):
                continue
                
            # Convert mask to tensor
            cluster_mask_tensor = torch.tensor(cluster_mask, dtype=torch.bool)
            
            # Compute prototype embedding as mean of cluster embeddings
            prototype = embeddings[cluster_mask_tensor].mean(dim=0, keepdim=True)
            
            # Compute prototype target as mean of cluster targets
            prototype_target = targets[cluster_mask_tensor].mean(dim=0, keepdim=True)
            
            prototypes.append(prototype)
            prototype_targets.append(prototype_target)
        
        # Concatenate prototypes and targets
        if prototypes:
            return torch.cat(prototypes, dim=0), torch.cat(prototype_targets, dim=0)
    
    # Default: use mean of all embeddings as single prototype
    prototype = embeddings.mean(dim=0, keepdim=True)
    prototype_target = targets.mean(dim=0, keepdim=True)
    
    return prototype, prototype_target


def protonet_train(model, data, input_columns, target_columns, epochs=50, 
                  learning_rate=0.001, num_tasks=5, num_shot=None, num_query=None,
                  batch_size=16, scaler_x=None, scaler_y=None, early_stopping=True):
  
    # Get labeled data
    labeled_data = data.dropna(subset=target_columns).reset_index(drop=True)
    MINIMUM_DATA_POINTS = 8
    if len(labeled_data) > 0 and len(labeled_data) < MINIMUM_DATA_POINTS:
        st.warning(f"🔄 Few labeled samples ({len(labeled_data)}). Tiling for robust training.")
        tile_factor = int(np.ceil(MINIMUM_DATA_POINTS / len(labeled_data)))
        labeled_data = pd.concat([labeled_data] * tile_factor, ignore_index=True)

        for col in input_columns:
            noise_scale = labeled_data[col].std() * 0.05
            labeled_data[col] += np.random.normal(0, noise_scale, size=len(labeled_data))
        for col in target_columns:
            noise_scale = max(labeled_data[col].std() * 0.02, 1e-6)
            labeled_data[col] += np.random.normal(0, noise_scale, size=len(labeled_data))

    
    # Initialize scalers
    if scaler_x is None:
        scaler_x = RobustScaler() if len(labeled_data) < 20 else StandardScaler()
    if scaler_y is None:
        scaler_y = RobustScaler() if len(labeled_data) < 20 else StandardScaler()
    
    total_samples = len(labeled_data)
    
    # Check if we have enough data to proceed
    if total_samples < 3:
        st.error(f"Not enough labeled samples. Need at least 3, but got {total_samples}.")
        return model, scaler_x, scaler_y
    
    # Dynamically adjust num_shot and num_query based on available samples
    if num_shot is None or num_query is None:
        max_samples = total_samples // (2 * max(1, num_tasks))
        num_shot = max(1, int(0.4 * max_samples))  # Ensure at least 1 sample
        num_query = max(1, int(0.6 * max_samples)) # Ensure at least 1 sample
    
    # Ensure num_shot + num_query does not exceed available samples
    if num_shot + num_query > total_samples:
        st.warning(f"Adjusting sample counts: total_samples={total_samples}")
        num_shot = max(1, total_samples // 2)
        num_query = max(1, total_samples - num_shot)
    
    # Calculate maximum possible tasks
    max_possible_tasks = max(1, total_samples // max(1, (num_shot + num_query)))
    num_tasks = max(1, min(num_tasks, max_possible_tasks))
    
    st.info(f"Training with {total_samples} samples: {num_shot} support, {num_query} query, {num_tasks} tasks")
    
    # Transform data
    inputs = scaler_x.fit_transform(labeled_data[input_columns].values)
    targets = scaler_y.fit_transform(labeled_data[target_columns].values)
    
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    
    # Select loss function based on dataset size
    if total_samples < 10:
        # For very small datasets, use MSE loss
        loss_function = nn.MSELoss()
    else:
        # For larger datasets, use Huber loss for robustness
        loss_function = nn.SmoothL1Loss()
    
    # Optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Initialize scheduler for adaptive learning rate
    scheduler = initialize_scheduler(
        optimizer, 
        scheduler_type="ReduceLROnPlateau", 
        patience=5, 
        factor=0.5, 
        min_lr=1e-6
    )
    
    # Early stopping parameters
    early_stopping_patience = 15
    no_improvement_epochs = 0
    best_loss = float('inf')
    best_model_state = None
    
    # Progress bar
    progress_bar = st.progress(0)
    
    # Training loop
    for epoch in range(epochs):
        # Update progress bar
        progress_bar.progress((epoch + 1) / epochs)
        
        model.train()
        epoch_loss = 0.0
        
        # Process tasks
        for task in range(num_tasks):
            # Sample support and query sets
            indices = torch.randperm(len(inputs))
            support_indices = indices[:num_shot]
            query_indices = indices[num_shot:num_shot + num_query]
            
            # Handle edge case: not enough samples for query set
            if len(query_indices) == 0:
                st.warning("Not enough samples for query set. Adjusting to continue training.")
                query_indices = indices[-1:] if len(indices) > 1 else indices  # Use at least one sample
            
            # Get support and query sets
            support_inputs = inputs[support_indices]
            support_targets = targets[support_indices]
            query_inputs = inputs[query_indices]
            query_targets = targets[query_indices]
            
            # Forward pass: encode support and query sets
            support_embeddings = model.encode(support_inputs)
            query_embeddings = model.encode(query_inputs)
            
            # Generate prototypes from support set
            use_kmeans = (len(support_indices) >= 10)  # Only use K-means with enough samples
            num_clusters = min(5, len(support_indices) // 2) if use_kmeans else None
            
            prototypes, prototype_targets = compute_prototypes(
                support_embeddings, 
                support_targets,
                num_clusters=num_clusters,
                use_kmeans=use_kmeans
            )
            
            # Hybrid loss calculation
            if total_samples < 10:
                # For very small datasets: Simple prediction from prototype
                query_predictions = model.projector(query_embeddings)
                loss = loss_function(query_predictions, query_targets)
            else:
                # For larger datasets: Distance-based predictions
                distances = euclidean_distance(query_embeddings, prototypes)
                
                # Convert distances to weights (closer = higher weight)
                similarity = torch.exp(-distances)
                weights = similarity / (similarity.sum(dim=1, keepdim=True) + 1e-8)
                
                # Predict query targets as weighted average of prototype targets
                predicted_targets = torch.matmul(weights, prototype_targets)
                
                # Compute loss
                loss = loss_function(predicted_targets, query_targets)
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                st.warning("NaN or Inf loss detected, skipping update for this task")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent instability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update model parameters
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
        
        # Calculate average loss for this epoch
        avg_epoch_loss = epoch_loss / max(1, num_tasks)
        
        # Update learning rate scheduler
        scheduler.step(avg_epoch_loss)
        
        # Print progress occasionally
        if epoch % 5 == 0 or epoch == epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            st.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}, LR: {current_lr:.6f}")
        
        # Early stopping check
        if early_stopping:
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1
                
            if no_improvement_epochs >= early_stopping_patience:
                st.warning(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Restore best model
    if early_stopping and best_model_state:
        model.load_state_dict(best_model_state)
        st.success(f"Restored best model with loss: {best_loss:.4f}")
    
    # Fine-tune the projection head for direct prediction
    if total_samples >= 5:
        model = fine_tune_projection(model, inputs, targets, epochs=min(20, epochs//2), learning_rate=learning_rate/5)
    
    model.is_trained = True
    model.scaler_x = scaler_x
    model.scaler_y = scaler_y
    return model, scaler_x, scaler_y


def fine_tune_projection(model, inputs, targets, epochs=20, learning_rate=0.0001):

    # Freeze encoder to preserve learned embeddings
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # Only optimize projector
    optimizer = optim.Adam(model.projector.parameters(), lr=learning_rate)
    loss_function = nn.SmoothL1Loss()
    
    # Use all data at once for very small datasets
    batch_size = min(16, len(inputs))
    
    # Fine-tuning loop
    for epoch in range(epochs):
        # Shuffle data
        indices = torch.randperm(len(inputs))
        inputs_shuffled = inputs[indices]
        targets_shuffled = targets[indices]
        
        # Process in batches
        total_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs_shuffled[i:i+batch_size]
            batch_targets = targets_shuffled[i:i+batch_size]
            
            # Forward pass
            predictions = model(batch_inputs)
            loss = loss_function(predictions, batch_targets)
            
            # Backward pass (only updating projection head)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Print progress occasionally
        if epoch % 5 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / max(1, num_batches)
            st.info(f"Fine-tuning epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Unfreeze encoder for future updates
    for param in model.encoder.parameters():
        param.requires_grad = True

    model.is_trained = True
    # Note: protonet_train returns scaler_x, scaler_y but doesn't set them on model.
    # This should ideally happen inside protonet_train, or the calling function needs to set them.
    # For now, we'll assume they are set if the model is to be used by BayesianOptimizer.
    # Let's modify protonet_train to set them.
    
    return model

# Duplicated protonet_train definition and orphaned lines removed from here.
def evaluate_protonet(model, data, input_columns, target_columns, curiosity, weights, max_or_min, acquisition="UCB"):

    # Get labeled and unlabeled data
    unlabeled_data = data[data[target_columns].isna().any(axis=1)]
    labeled_data = data.dropna(subset=target_columns)
    
    if unlabeled_data.empty:
        st.warning("No unlabeled samples available for evaluation.")
        return None
    
    if len(labeled_data) < 2:
        st.warning("Need at least 2 labeled samples for reliable evaluation.")
        return None
    
    # Scale inputs and targets
    scaler_inputs = RobustScaler().fit(data[input_columns]) if len(labeled_data) < 20 else StandardScaler().fit(data[input_columns])
    scaler_targets = RobustScaler().fit(labeled_data[target_columns]) if len(labeled_data) < 20 else StandardScaler().fit(labeled_data[target_columns])
    
    # Transform data
    labeled_inputs = torch.tensor(scaler_inputs.transform(labeled_data[input_columns]), dtype=torch.float32)
    labeled_targets = torch.tensor(scaler_targets.transform(labeled_data[target_columns]), dtype=torch.float32)
    
    inputs_infer = scaler_inputs.transform(unlabeled_data[input_columns])
    inputs_infer_tensor = torch.tensor(inputs_infer, dtype=torch.float32)
    
    # Set evaluation strategy based on dataset size
    use_prototypes = (len(labeled_data) < 15)
    
    model.eval()
    with torch.no_grad():
        if use_prototypes:
            # For smaller datasets: Use prototype-based prediction
            st.info("📊 Using prototype-based prediction (fallback strategy for low data)")            
            # Encode inputs
            labeled_embeddings = model.encode(labeled_inputs)
            unlabeled_embeddings = model.encode(inputs_infer_tensor)
            
            # Create prototypes (either multiple clusters or single mean)
            use_kmeans = (len(labeled_data) >= 8)
            num_clusters = min(5, len(labeled_data) // 2) if use_kmeans else None
            
            prototypes, prototype_targets = compute_prototypes(
                labeled_embeddings, 
                labeled_targets,
                num_clusters=num_clusters,
                use_kmeans=use_kmeans
            )
            
            # Compute distances to prototypes
            distances = euclidean_distance(unlabeled_embeddings, prototypes)
            
            # Convert distances to weights
            similarity = torch.exp(-distances)
            weights_tensor = similarity / (similarity.sum(dim=1, keepdim=True) + 1e-8)
            
            # Predict targets as weighted average of prototype targets
            predictions_scaled = torch.matmul(weights_tensor, prototype_targets).numpy()
            
            # Also get direct predictions for blending
            direct_predictions = model(inputs_infer_tensor).numpy()
            
            # Blend prototype-based and direct predictions
            alpha = min(0.8, max(0.4, 7 / len(labeled_data)))  # Adjust weight based on data size
            predictions_scaled = alpha * predictions_scaled + (1 - alpha) * direct_predictions
        else:
            # For larger datasets: Use direct neural network prediction
            st.info("📈 Using direct neural network prediction (sufficient data)")
            predictions_scaled = model(inputs_infer_tensor).numpy()
    
    # Calculate uncertainty using Monte Carlo dropout from app.models
    from app.models import calculate_uncertainty_ensemble
    # model.train() is handled by calculate_uncertainty_ensemble
    # Unpack the tuple correctly: (uncertainty_array, mc_samples_stack)
    uncertainty_scores, _ = calculate_uncertainty_ensemble(
        model=model,
        X=inputs_infer_tensor,
        num_samples=50, # Corresponds to num_perturbations
        dropout_rate=0.3, # Default dropout rate
        use_input_perturbation=True, # ProtoNet's old uncertainty used input perturbation
        noise_scale=0.05 # Default noise scale
    )
    if uncertainty_scores.ndim == 1: # Ensure shape (n_samples, 1)
        uncertainty_scores = uncertainty_scores.reshape(-1, 1)
    
    # Inverse transform predictions to original scale
    predictions = scaler_targets.inverse_transform(predictions_scaled)
    # 🔒 Ensure non-negative predictions, aligning with expected output characteristics.
    predictions = np.maximum(predictions, 0)

    # 🚫 Threshold enforcement
    # Define a minimum acceptable strength/property threshold.
    min_strength_threshold = 10
    # Identify rows where predictions are below the threshold for utility penalization.
    invalid_rows = np.any(predictions < min_strength_threshold, axis=1)

    
    # Calculate novelty
    X_labeled_scaled = scaler_inputs.transform(labeled_data[input_columns])
    novelty_scores = calculate_novelty(inputs_infer, X_labeled_scaled)
    
    # Ensure weights and max_or_min are properly formatted
    if not isinstance(weights, np.ndarray):
        weights = np.array(weights)
    
    if max_or_min is None or not isinstance(max_or_min, list):
        max_or_min = ['max'] * len(target_columns)
    
    # ⚖️ SLAMD-style acquisition weights adjustment
    weights = weights + np.random.uniform(0.01, 0.1, size=weights.shape)
    weights = np.clip(weights, 0.1, 1.0)
    
    from app.utils import select_acquisition_function
    if acquisition is None:
        acquisition = select_acquisition_function(curiosity, len(labeled_data))
    print(f"🎯 Target optimization direction: {max_or_min[0]}")
    print(f"🔍 Acquisition function: {acquisition}, Curiosity level: {curiosity}")


    
    # Calculate utility with specified acquisition function
    utility_scores = calculate_utility(
        predictions,
        uncertainty_scores,
        novelty_scores,
        curiosity,
        weights,
        max_or_min,
        acquisition=acquisition
    )
    # 🛑 Apply penalty for invalid predictions
    # Penalize utility for predictions that fall below the min_strength_threshold.
    if invalid_rows.any():
        utility_scores[invalid_rows] = -np.inf

    # Create result DataFrame
    result_df = unlabeled_data.copy()
    
    # Add calculated metrics
    result_df["Utility"] = utility_scores.flatten()
    result_df["Uncertainty"] = np.clip(uncertainty_scores, 1e-6, None).flatten()
    result_df["Novelty"] = novelty_scores.flatten()
    
    # Add exploration-exploitation balance metrics
    result_df["Exploration"] = result_df["Uncertainty"] * result_df["Novelty"]
    result_df["Exploitation"] = 1.0 - result_df["Uncertainty"]
    
    # Add predicted values for target columns
    for i, col in enumerate(target_columns):
        result_df[col] = predictions[:, i]
    
    # Sort by utility and mark the best candidate
    result_df = result_df.sort_values(by="Utility", ascending=False)
    result_df["Selected for Testing"] = False
    result_df.iloc[0, result_df.columns.get_loc("Selected for Testing")] = True

    # SLAMD-style override for maximization
    if max_or_min[0] == "max":
        highest_value_idx = np.argmax(predictions[:, 0])
        highest_value = predictions[highest_value_idx, 0]
        selected_value = result_df.iloc[0][target_columns[0]]
        
        if highest_value > selected_value * 1.1:
            print(f"⚠️ Overriding selection to ensure highest value ({highest_value})")
            result_df["Selected for Testing"] = False
            result_df.iloc[highest_value_idx, result_df.columns.get_loc("Selected for Testing")] = True

    
    # Reset index for better display
    result_df.reset_index(drop=True, inplace=True)
    
    
    # Organize columns for better presentation
    columns_to_front = ["Row Number", "Idx_Sample"] if "Idx_Sample" in result_df.columns else ["Row Number"]
    metrics_columns = ["Utility", "Exploration", "Exploitation", "Novelty", "Uncertainty"]
    
    # Get all other columns excluding metrics and targets
    remaining_columns = [col for col in result_df.columns 
                         if col not in columns_to_front + metrics_columns + target_columns + ["Selected for Testing"]]
    
    # Set the new column order, ensuring all columns exist
    new_column_order = [col for col in (columns_to_front + metrics_columns + target_columns + 
                                       remaining_columns + ["Selected for Testing"])
                        if col in result_df.columns]
    
    result_df = result_df[new_column_order]
    
    return result_df