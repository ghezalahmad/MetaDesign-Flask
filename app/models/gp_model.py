import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.neighbors import NearestNeighbors

# Assuming this internal import path is correct for your setup
from app.models.bayesian_optimizer import multi_objective_bayesian_optimization
from app.utils.utils import calculate_novelty # Reusing the shared utility function

class GPModel:
    """
    A wrapper for GaussianProcessRegressor handling multi-output targets
    via multiple independent GP models.
    """
    def __init__(self, kernel=None, alpha=1e-10, normalize_y=True, random_state=42):
        # A list to hold one GP model for each target
        self.gp_models = []
        self.scaler_x = StandardScaler()
        self.target_columns = []
        self.is_trained = False
        
        # Default Kernel: A robust combination of constant, RBF, and noise
        if kernel is None:
            # C: constant amplitude, RBF: non-linearity, WhiteKernel: handles noise/jitter
            self.base_kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=alpha, noise_level_bounds=(1e-10, 1e-1))
        else:
            self.base_kernel = kernel

        self.alpha = alpha
        self.normalize_y = normalize_y
        self.random_state = random_state

    def train(self, data, input_columns, target_columns):
        """Trains one GPR model for each target column."""
        
        train_data = data.dropna(subset=target_columns)
        self.target_columns = target_columns

        if train_data.empty or len(train_data) < 2:
            print("GPModel: Insufficient training data. Skipping fit.")
            self.is_trained = False
            return

        X = train_data[input_columns]
        
        # 1. Scale inputs
        X_scaled = self.scaler_x.fit_transform(X)
        
        self.gp_models = []
        for col in target_columns:
            y = train_data[col].values.reshape(-1, 1)
            
            # Initialize a new GP for each target
            gp = GaussianProcessRegressor(
                kernel=self.base_kernel,
                alpha=self.alpha,
                n_restarts_optimizer=10,
                normalize_y=self.normalize_y,
                random_state=self.random_state
            )
            
            # Fit the GP model
            gp.fit(X_scaled, y)
            self.gp_models.append(gp)

        self.is_trained = True

    def predict(self, X_input: pd.DataFrame | np.ndarray):
        """Generates mean predictions."""
        mean_preds, _ = self.predict_with_uncertainty(X_input)
        return mean_preds

    def predict_with_uncertainty(self, X_input: pd.DataFrame | np.ndarray, input_columns=None, num_samples=None):
        """
        Generates mean predictions and standard deviations for all targets.
        Returns:
            (predictions, std_deviations) shape (n_samples, n_targets)
        """
        if not self.is_trained or not self.gp_models:
            n = len(X_input)
            out_targets = len(self.target_columns) if self.target_columns else 1
            return np.zeros((n, out_targets)), np.zeros((n, out_targets))

        # 1. Prepare input
        if isinstance(X_input, pd.DataFrame):
            X_processed = X_input[self.scaler_x.feature_names_in_].values
        else:
            X_processed = X_input
            
        X_scaled = self.scaler_x.transform(X_processed)
        
        predictions = []
        std_deviations = []

        # 2. Predict using each independent GP
        for gp in self.gp_models:
            # return_std=True provides the standard deviation
            mean_pred, std_dev = gp.predict(X_scaled, return_std=True)
            predictions.append(mean_pred.reshape(-1, 1))
            std_deviations.append(std_dev.reshape(-1, 1))

        # 3. Combine results into (n_samples, n_targets) arrays
        # np.hstack stacks them horizontally
        predictions_combined = np.hstack(predictions)
        std_deviations_combined = np.hstack(std_deviations)

        return predictions_combined, std_deviations_combined


def train_gp_model(data, input_columns, target_columns):
    """Trains a Gaussian Process model."""
    model = GPModel()
    model.train(data, input_columns, target_columns)
    return model, model.scaler_x, None # No separate y-scaler needed due to normalize_y=True


def evaluate_gp_model(model, data, input_columns, target_columns, curiosity, weights, max_or_min):
    """Evaluates the GP model on candidate samples and calculates utility scores."""

    labeled_data = data.dropna(subset=target_columns)
    # Candidate samples are those where the first target column is null
    candidate_df = data[data[target_columns[0]].isnull()].copy()

    if candidate_df.empty:
        return pd.DataFrame()
    
    # Ensure model is trained
    if not getattr(model, 'is_trained', False):
        print("evaluate_gp_model: Model untrained. Training now...")
        model.train(data, input_columns, target_columns)


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
    
    # 3. Map Predictions and Target-Specific Uncertainties to columns
    for i, col in enumerate(target_columns):
        # Predictions
        candidate_df[col] = predictions[:, i]
            
        # Uncertainties 
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