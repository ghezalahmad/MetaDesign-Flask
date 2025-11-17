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

class FeatureExtractor(nn.Module):
    """The Deep Neural Network component for feature extraction."""
    def __init__(self, input_size, feature_dim, hidden_size=64):
        # NOTE: feature_dim must match the number of output targets (output_size) for the MSE loss to work.
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
    def __init__(self, input_size, output_size, hidden_size=64):
        # FIX APPLIED HERE: The feature extractor output size must match the number of targets (output_size)
        # because its output is used directly for the MSE loss calculation against Y_tensor.
        self.feature_dim = output_size # Set feature dimension equal to the output size for training stability
        self.feature_extractor = FeatureExtractor(input_size, self.feature_dim, hidden_size)
        
        self.gp_models = []
        self.scaler_x = StandardScaler()
        self.target_columns = []
        self.is_trained = False
        self.input_size = input_size
        self.output_size = output_size
        
        # Define the base kernel for the GP (RBF is common for DKL)
        self.base_kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-10, noise_level_bounds=(1e-10, 1e-1))

    def _extract_features(self, X_scaled: np.ndarray) -> np.ndarray:
        """Passes scaled data through the trained NN to get features."""
        self.feature_extractor.eval()
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            features = self.feature_extractor(X_tensor).numpy()
        return features

    def train(self, data, input_columns, target_columns, fe_epochs=50, fe_lr=0.001):
        """
        Trains the Feature Extractor NN and then fits the GPs on the extracted features.
        """
        train_data = data.dropna(subset=target_columns)
        self.target_columns = target_columns

        if train_data.empty or len(train_data) < 2:
            print("DKLModel: Insufficient training data. Skipping fit.")
            self.is_trained = False
            return

        X = train_data[input_columns]
        Y = train_data[target_columns]

        # 1. Scale Inputs
        X_scaled = self.scaler_x.fit_transform(X)
        
        # 2. Train Feature Extractor NN (using all targets combined for feature loss)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        Y_tensor = torch.tensor(Y.values, dtype=torch.float32)
        
        optimizer = optim.Adam(self.feature_extractor.parameters(), lr=fe_lr)
        loss_fn = nn.MSELoss()
        
        self.feature_extractor.train()
        for epoch in range(fe_epochs):
            optimizer.zero_grad()
            # Predictions now match the Y_tensor size (self.output_size)
            predictions = self.feature_extractor(X_tensor) 
            loss = loss_fn(predictions, Y_tensor)
            loss.backward()
            optimizer.step()
        
        # 3. Extract Features for GP Training
        # Note: The 'features' now have the size of self.output_size (e.g., 2), 
        # which is sufficient to guide the GP if feature_dim == output_size.
        features = self._extract_features(X_scaled)
        
        # 4. Train Independent GPs on the Extracted Features
        self.gp_models = []
        for i, col in enumerate(target_columns):
            y_i = Y.iloc[:, i].values.reshape(-1, 1)
            
            gp = GaussianProcessRegressor(
                kernel=self.base_kernel,
                alpha=1e-10,
                n_restarts_optimizer=5,
                normalize_y=True,
                random_state=42
            )
            gp.fit(features, y_i)
            self.gp_models.append(gp)

        self.is_trained = True

    def predict(self, X_input: pd.DataFrame | np.ndarray):
        """Generates mean predictions."""
        mean_preds, _ = self.predict_with_uncertainty(X_input)
        return mean_preds

    def predict_with_uncertainty(self, X_input: pd.DataFrame | np.ndarray, input_columns=None, num_samples=None):
        """
        Generates mean predictions and standard deviations for all targets.
        """
        if not self.is_trained or not self.gp_models:
            n = len(X_input)
            out_targets = len(self.target_columns) if self.target_columns else self.output_size
            return np.zeros((n, out_targets)), np.zeros((n, out_targets))

        # 1. Scale Input
        X_processed = X_input.values if isinstance(X_input, pd.DataFrame) else X_input
        X_scaled = self.scaler_x.transform(X_processed)
        
        # 2. Extract Features
        features = self._extract_features(X_scaled)
        
        predictions = []
        std_deviations = []

        # 3. Predict using each independent GP
        for gp in self.gp_models:
            mean_pred, std_dev = gp.predict(features, return_std=True)
            predictions.append(mean_pred.reshape(-1, 1))
            std_deviations.append(std_dev.reshape(-1, 1))

        predictions_combined = np.hstack(predictions)
        std_deviations_combined = np.hstack(std_deviations)

        return predictions_combined, std_deviations_combined


def train_dkl_model(data, input_columns, target_columns):
    """Initializes and trains the DKL model."""
    input_size = len(input_columns)
    output_size = len(target_columns)
    model = DKLModel(input_size, output_size)
    # Default training parameters for feature extractor
    model.train(data, input_columns, target_columns, fe_epochs=100, fe_lr=0.005)
    return model, model.scaler_x, None


def evaluate_dkl_model(model, data, input_columns, target_columns, curiosity, weights, max_or_min):
    """Evaluates the DKL model on candidate samples using Bayesian Optimization."""

    labeled_data = data.dropna(subset=target_columns)
    candidate_df = data[data[target_columns[0]].isnull()].copy()

    if candidate_df.empty:
        return pd.DataFrame()
    
    # Ensure model is trained
    if not getattr(model, 'is_trained', False):
        print("evaluate_dkl_model: Model untrained. Training now...")
        # Use default training parameters if not trained
        model.train(data, input_columns, target_columns, fe_epochs=100, fe_lr=0.005)


    train_inputs = labeled_data[input_columns]
    train_targets = labeled_data[target_columns].values
    candidate_inputs = candidate_df[input_columns]
    
    # 1. Run Bayesian Optimization to get Utility Scores
    utility_scores = multi_objective_bayesian_optimization(
        train_inputs=train_inputs,
        train_targets=train_targets,
        candidate_inputs=candidate_inputs,
        weights=weights,
        max_or_min=max_or_min,
        curiosity=curiosity,
        acquisition="UCB",
        strategy="weighted_sum",
        surrogate_model=model,
        input_columns=input_columns
    )

    # 2. Get Predictions and Uncertainties
    predictions, uncertainties = model.predict_with_uncertainty(candidate_inputs)
    
    # 3. Map Predictions and Uncertainties to columns
    for i, col in enumerate(target_columns):
        candidate_df[col] = predictions[:, i]
        candidate_df[f"Uncertainty ({col})"] = uncertainties[:, i]

    # 4. Assign Utility Scores
    if utility_scores is None:
        candidate_df["Utility"] = 0.0
    else:
        utility_scores = np.array(utility_scores, dtype=np.float64).flatten()
        n = min(len(utility_scores), len(candidate_df))
        if len(utility_scores) != len(candidate_df):
            candidate_df = candidate_df.iloc[:n].copy()
            utility_scores = utility_scores[:n]
        candidate_df["Utility"] = utility_scores

    candidate_df["Utility"] = pd.to_numeric(candidate_df["Utility"], errors="coerce").fillna(0.0).astype(float)
    candidate_df["Uncertainty"] = np.mean(uncertainties, axis=1)

    # 5. Calculate Novelty
    X_candidate_np = candidate_inputs.values
    X_labeled_np = labeled_data[input_columns].values
    novelty_scores = calculate_novelty(X_candidate_np, X_labeled_np)
    candidate_df["Novelty"] = novelty_scores

    # 6. Exploration / Exploitation
    candidate_df["Exploration"] = candidate_df["Uncertainty"] * (1.0 + curiosity)
    candidate_df["Exploitation"] = candidate_df[target_columns].mean(axis=1)

    # 7. Selection Flag
    candidate_df["Selected for Testing"] = False
    if not candidate_df["Utility"].empty:
        max_utility_idx = candidate_df["Utility"].idxmax()
        candidate_df.loc[max_utility_idx, "Selected for Testing"] = True

    # 8. Final Sort
    result_df = candidate_df.sort_values(by="Utility", ascending=False).reset_index(drop=True)
    return result_df