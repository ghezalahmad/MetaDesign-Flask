import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.neighbors import NearestNeighbors
import streamlit as st # Assuming Streamlit components for logging/warnings

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
        self.scaler_y = StandardScaler() # Added scaler for consistency
        self.target_columns = []
        self.input_columns = []
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

    def train(self, data: pd.DataFrame, input_columns: list, target_columns: list):
        """Trains the model on the provided data."""
        self.input_columns = input_columns
        self.target_columns = target_columns
        train_data = data.dropna(subset=target_columns)

        if train_data.empty or len(train_data) < 2:
            st.warning("Insufficient data to train GPModel. Using dummy scalers.")
            X_dummy = pd.DataFrame(np.zeros((2, len(input_columns))), columns=input_columns)
            y_dummy = pd.DataFrame(np.zeros((2, len(target_columns))), columns=target_columns)
            self.scaler_x.fit(X_dummy)
            self.scaler_y.fit(y_dummy)
            self.is_trained = False
            return

        X = train_data[input_columns].values
        Y = train_data[target_columns].values

        X_scaled = self.scaler_x.fit_transform(X)
        Y_scaled = self.scaler_y.fit_transform(Y)
        
        self.gp_models = []
        for i in range(Y_scaled.shape[1]):
            # Create a new GP model instance for each target
            gp = GaussianProcessRegressor(
                kernel=self.base_kernel,
                alpha=self.alpha,
                normalize_y=self.normalize_y,
                random_state=self.random_state,
                n_restarts_optimizer=10 # Optimize hyperparameters better
            )
            # Train the model on the scaled inputs (X) and the scaled i-th target (Y_scaled[:, i])
            gp.fit(X_scaled, Y_scaled[:, i])
            self.gp_models.append(gp)

        self.is_trained = True
        print(f"GPModel trained successfully with {len(self.gp_models)} independent models.")

    def predict_with_uncertainty(self, X_input: pd.DataFrame, return_std=True):
        """
        Generates predictions and associated standard deviations (uncertainty).
        
        Returns:
            tuple: (np.ndarray of predictions, np.ndarray of standard deviations)
        """
        if not self.is_trained:
            st.warning("GPModel is not trained. Returning zeros for predictions and uncertainties.")
            num_targets = len(self.target_columns) if self.target_columns else 1
            num_samples = len(X_input)
            return np.zeros((num_samples, num_targets)), np.ones((num_samples, num_targets)) * 1.0
        
        X_np = X_input[self.input_columns].values
        X_scaled = self.scaler_x.transform(X_np)
        
        # Collect predictions and std devs for all targets
        predictions_scaled = []
        std_devs_scaled = []
        
        for gp in self.gp_models:
            # Predict mean (mu_scaled) and standard deviation (std_scaled)
            mu_scaled, std_scaled = gp.predict(X_scaled, return_std=return_std)
            predictions_scaled.append(mu_scaled.reshape(-1, 1))
            std_devs_scaled.append(std_scaled.reshape(-1, 1))

        # Combine and inverse transform predictions
        predictions_scaled_combined = np.hstack(predictions_scaled)
        predictions_unscaled = self.scaler_y.inverse_transform(predictions_scaled_combined)

        # The standard deviation from GP is of the SCALED data. 
        # To get the uncertainty in the UNCALED space, we must rescale the standard deviation.
        # std_unscaled = std_scaled * scale_factor_y
        
        # Get the scaling factor for each target column (standard deviation of the training data)
        scale_factors_y = np.sqrt(self.scaler_y.var_)
        
        # Combine the scaled standard deviations
        std_devs_scaled_combined = np.hstack(std_devs_scaled)
        
        # Apply the scaling factor element-wise to get unscaled uncertainties
        # Reshape the scale factors to broadcast correctly
        uncertainties_unscaled = std_devs_scaled_combined * scale_factors_y

        return predictions_unscaled, uncertainties_unscaled, None  # Add None for posterior samples

def train_gp_model(data: pd.DataFrame, input_columns: list, target_columns: list, model_params: dict):
    """
    Standalone function to create and train a GPModel instance, 
    matching the signature of other model training functions.
    """
    model = GPModel(**model_params)
    model.train(data, input_columns, target_columns)
    return model, None, None  # Return 3 values for consistency

def evaluate_gp_model(
    model: GPModel,
    labeled_data: pd.DataFrame,
    candidate_inputs: pd.DataFrame,
    input_columns: list,
    target_columns: list,
    weights: list,
    max_or_min: list,
    curiosity: float
) -> pd.DataFrame:
    """
    Uses the trained GP model to evaluate candidate inputs based on a 
    multi-objective Bayesian Optimization utility score.
    """
    if not model.is_trained:
        st.warning("GPModel is not trained. Evaluation skipped.")
        return candidate_inputs.copy()

    # 1. Run multi-objective Bayesian Optimization to get utility scores
    st.info("🎯 Running multi-objective Bayesian Optimization with GP surrogate to score candidates.")
    
    utility_scores = multi_objective_bayesian_optimization(
        train_inputs=labeled_data[input_columns],
        train_targets=labeled_data[target_columns],
        candidate_inputs=candidate_inputs,
        weights=np.array(weights),
        max_or_min=max_or_min,
        curiosity=curiosity,
        acquisition="UCB", # UCB is a good default for GP
        strategy="weighted_sum",
        surrogate_model=model,
        input_columns=input_columns
    )

    # 2. Get predictions and uncertainties for all candidates
    predictions, uncertainties, _ = model.predict_with_uncertainty(candidate_inputs)

    # 3. Build the results DataFrame
    candidate_df = candidate_inputs.copy()
    
    # 4. Assign predictions and uncertainty per target
    for i, col in enumerate(target_columns):
        candidate_df[col] = predictions[:, i]
        candidate_df[f"Uncertainty ({col})"] = uncertainties[:, i]

    # 5. Assign Utility Scores
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

    # 6. Calculate Aggregate Uncertainty (Mean of target uncertainties)
    candidate_df["Uncertainty"] = np.mean(uncertainties, axis=1)

    # 7. Calculate Novelty
    X_candidate_np = candidate_inputs.values
    X_labeled_np = labeled_data[input_columns].values
    novelty_scores = calculate_novelty(X_candidate_np, X_labeled_np)
    candidate_df["Novelty"] = novelty_scores

    # 8. Exploration / Exploitation
    candidate_df["Exploration"] = candidate_df["Uncertainty"] * (1.0 + curiosity)
    candidate_df["Exploitation"] = candidate_df[target_columns].mean(axis=1)

    # 9. Selection Flag
    candidate_df["Selected for Testing"] = False
    if not candidate_df["Utility"].empty:
        max_utility_idx = candidate_df["Utility"].idxmax()
        candidate_df.loc[max_utility_idx, "Selected for Testing"] = True

    # 10. Final sorting
    result_df = candidate_df.sort_values(by="Utility", ascending=False).reset_index(drop=True)
    
    return result_df