import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# Assuming these internal imports are correct for your setup
from app.models.bayesian_optimizer import multi_objective_bayesian_optimization
from app.utils.utils import calculate_novelty
import logging

logging.basicConfig(level=logging.INFO)

class FeatureExtractor(nn.Module):
    """The Deep Neural Network component for feature extraction."""
    def __init__(self, input_size, feature_dim, hidden_size=64):
        super(FeatureExtractor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_dim) # Output is the feature dimension
        )

    def forward(self, x):
        return self.net(x)

class DKLModel:
    """
    Implements Deep Kernel Learning (DKL) using a NN for feature extraction 
    and independent Gaussian Processes (GPs) on the features for multi-output targets.
    """
    def __init__(self, input_size, output_size, hidden_size=64, alpha=1e-6):
        self.input_size = input_size
        self.output_size = output_size
        self.feature_extractor = FeatureExtractor(input_size, output_size, hidden_size=hidden_size)
        
        # A list to hold one GP model for each target
        self.gp_models = []
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.target_columns = []
        self.is_trained = False
        self.alpha = alpha

        # Default Kernel for GPs in feature space
        self.base_kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=alpha, noise_level_bounds=(1e-10, 1e-1))

    def train(self, X: pd.DataFrame, y: pd.DataFrame, epochs=100, lr=0.001):
        """
        Trains the DKL model. This involves training the feature extractor (NN)
        and then fitting independent GPs on the extracted features.
        
        NOTE: In a true DKL implementation (e.g., GPyTorch), the NN and GP would be trained jointly.
        Here, we use a simpler sequential approach for easier integration and stability.
        """
        self.target_columns = y.columns.tolist()
        
        # 1. Prepare data (Scaling)
        X_scaled = self.scaler_x.fit_transform(X.values)
        y_scaled = self.scaler_y.fit_transform(y.values)
        
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

        # 2. Train Feature Extractor (NN)
        logging.info("Training DKL Feature Extractor...")
        optimizer = optim.Adam(self.feature_extractor.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            self.feature_extractor.train()
            optimizer.zero_grad()
            features = self.feature_extractor(X_tensor)
            
            # Use MSE between features and scaled targets as a proxy loss
            loss = criterion(features, y_tensor)
            loss.backward()
            optimizer.step()
        
        logging.info(f"Feature Extractor Training Complete. Final MSE Loss: {loss.item():.4f}")

        # 3. Extract Features and Train GPs
        self.feature_extractor.eval()
        with torch.no_grad():
            X_features_np = self.feature_extractor(X_tensor).numpy()
        
        self.gp_models = []
        for i in range(self.output_size):
            gp = GaussianProcessRegressor(
                kernel=self.base_kernel, 
                alpha=self.alpha,
                normalize_y=True,
                random_state=42
            )
            # Fit the GP on the extracted features (X_features_np) and the i-th target (y_scaled[:, i])
            gp.fit(X_features_np, y_scaled[:, i])
            self.gp_models.append(gp)

        self.is_trained = True

    def predict_with_uncertainty(self, X_input: pd.DataFrame, input_columns=None, num_samples=None):
        """
        Generates predictions and associated standard deviations (uncertainties).
        
        Args:
            X_input: Input features as DataFrame
            input_columns: Optional, for compatibility with other models
            num_samples: Optional, for compatibility with other models (not used by DKL)
        
        Returns:
            predictions_orig: (n_samples, n_targets)
            uncertainties_orig: (n_samples, n_targets)
            None: DKL doesn't provide posterior samples
        """
        if not self.is_trained:
            raise Exception("DKLModel not trained. Call train() first.")

        X_scaled = self.scaler_x.transform(X_input.values)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        self.feature_extractor.eval()
        with torch.no_grad():
            X_features_np = self.feature_extractor(X_tensor).numpy()

        predictions = []
        uncertainties = []

        for gp in self.gp_models:
            # Predict mean and std in the feature space
            y_pred_scaled, sigma_scaled = gp.predict(X_features_np, return_std=True)
            
            predictions.append(y_pred_scaled)
            # The uncertainty from the GP is the standard deviation (sigma)
            uncertainties.append(sigma_scaled)

        # 1. Stack and inverse transform predictions (means)
        predictions_scaled = np.vstack(predictions).T
        predictions_orig = self.scaler_y.inverse_transform(predictions_scaled)
        
        # 2. The uncertainty (sigma) from GP is on the *scaled* target space.
        # To convert to the original space, we multiply by the standard deviation of the original targets (y).
        # scaler_y.scale_ is the standard deviation used for scaling
        uncertainties_scaled = np.vstack(uncertainties).T
        # Broadcasting the scale factor: (N_samples, N_targets) * (N_targets,)
        uncertainties_orig = uncertainties_scaled * self.scaler_y.scale_
        
        return predictions_orig, uncertainties_orig, None  # Add None for posterior samples

# =======================================================
# Wrapper Functions for routes.py integration
# =======================================================

def train_dkl_model(data: pd.DataFrame, input_columns: list[str], target_columns: list[str], model_params: dict = None):
    """
    Trains the DKL model.
    """
    if model_params is None:
        model_params = {}
    
    train_data = data.dropna(subset=target_columns)
    
    if train_data.empty or len(train_data) < 2:
        logging.warning("Insufficient clean data to train DKL model.")
        # Return a dummy model or raise error, depending on desired application behavior
        # Here we raise the error so the calling route handles it.
        raise ValueError("Insufficient data for DKL training.")
    
    X = train_data[input_columns]
    y = train_data[target_columns]
    
    input_size = len(input_columns)
    output_size = len(target_columns)

    model = DKLModel(input_size=input_size, output_size=output_size, **model_params)
    
    # Train the model (using default or passed parameters for DKL training)
    # Note: DKL internal training parameters (epochs, lr) need to be part of model_params 
    # if you want to customize them, otherwise defaults are used.
    model.train(X, y)
    
    return model, None, None  # Return 3 values for consistency


def evaluate_dkl_model(model: DKLModel, labeled_data: pd.DataFrame, candidate_inputs: pd.DataFrame, 
                       input_columns: list[str], target_columns: list[str], 
                       weights: list[float], max_or_min: list[str], curiosity: float) -> pd.DataFrame:
    """
    Uses the trained DKL model as a surrogate for Bayesian Optimization 
    to evaluate and score candidate materials.
    """
    logging.info("Running Bayesian Optimization with DKL surrogate to score candidates.")
    
    # Extract training inputs and targets for BO context (needed for novelty and utility)
    train_inputs = labeled_data[input_columns]
    train_targets = labeled_data[target_columns]
    
    # 1. Calculate Utility Scores using BO
    utility_scores = multi_objective_bayesian_optimization(
        train_inputs=train_inputs,
        train_targets=train_targets,
        candidate_inputs=candidate_inputs,
        weights=np.array(weights),
        max_or_min=max_or_min,
        curiosity=curiosity,
        acquisition="UCB", # UCB is typically best with NN-based models like DKL
        strategy="weighted_sum",
        surrogate_model=model, # Pass the DKLModel instance
        input_columns=input_columns
    )

    # 2. Get predictions and uncertainties from the DKL model
    predictions, uncertainties, _ = model.predict_with_uncertainty(candidate_inputs)

    # 3. Prepare the results DataFrame
    candidate_df = candidate_inputs.copy()
    
    # Add prediction and uncertainty for each target
    for i, col in enumerate(target_columns):
        candidate_df[col] = predictions[:, i]
        # Store individual target uncertainty
        candidate_df[f"Uncertainty ({col})"] = uncertainties[:, i]

    # 4. Assign Utility Scores
    if utility_scores is None:
        candidate_df["Utility"] = 0.0
    else:
        utility_scores = np.array(utility_scores, dtype=np.float64).flatten()
        n = min(len(utility_scores), len(candidate_df))
        if len(utility_scores) != len(candidate_df):
            # Trim if BO returned an uneven number of scores
            candidate_df = candidate_df.iloc[:n].copy()
            utility_scores = utility_scores[:n]
        candidate_df["Utility"] = utility_scores

    candidate_df["Utility"] = pd.to_numeric(candidate_df["Utility"], errors="coerce").fillna(0.0).astype(float)
    
    # 5. Calculate Aggregate Uncertainty (Mean of all target uncertainties)
    candidate_df["Uncertainty"] = np.mean(uncertainties, axis=1)

    # 6. Calculate Novelty
    X_candidate_np = candidate_inputs.values
    X_labeled_np = labeled_data[input_columns].values
    novelty_scores = calculate_novelty(X_candidate_np, X_labeled_np)
    candidate_df["Novelty"] = novelty_scores

    # 7. Exploration / Exploitation
    # Exploration: UCB's exploration term is already incorporated into Utility, 
    # but we track uncertainty contribution separately.
    candidate_df["Exploration"] = candidate_df["Uncertainty"] * (1.0 + curiosity)
    # Exploitation: Mean predicted value of the primary target(s)
    candidate_df["Exploitation"] = candidate_df[target_columns].mean(axis=1)

    # 8. Selection Flag
    candidate_df["Selected for Testing"] = False
    if not candidate_df["Utility"].empty:
        max_utility_idx = candidate_df["Utility"].idxmax()
        candidate_df.loc[max_utility_idx, "Selected for Testing"] = True

    # 9. Final sorting
    result_df = candidate_df.sort_values(by="Utility", ascending=False).reset_index(drop=True)
    
    return result_df