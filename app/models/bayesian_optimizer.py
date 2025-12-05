import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
import streamlit as st
from app.utils.utils import calculate_novelty, enforce_diversity


def prepare_train_candidate_data(
    data: pd.DataFrame,
    input_columns: list,
    target_columns: list,
    status_column: str = "Status",
    tested_labels: tuple = ("tested", "labeled", "labelled", "train")
):

    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")

    missing_inputs = [c for c in input_columns if c not in data.columns]
    missing_targets = [c for c in target_columns if c not in data.columns]
    if missing_inputs or missing_targets:
        raise ValueError(
            f"Missing columns in data. "
            f"Inputs missing: {missing_inputs}, Targets missing: {missing_targets}"
        )

    # 1) Use explicit Status column if present
    if status_column in data.columns:
        status_series = data[status_column].astype(str).str.lower().str.strip()
        tested_mask = status_series.isin([s.lower() for s in tested_labels])
        candidate_mask = ~tested_mask
    else:
        # 2) Fallback: tested rows have all targets non-null
        tested_mask = ~data[target_columns].isnull().any(axis=1)
        candidate_mask = ~tested_mask

    n_tested = tested_mask.sum()
    if n_tested == 0:
        raise ValueError(
            "No TESTED samples found. "
            "Ensure your dataset has either a 'Status' column with appropriate "
            "values or non-null targets for tested rows."
        )

    train_inputs = data.loc[tested_mask, input_columns].copy()
    train_targets = data.loc[tested_mask, target_columns].copy()
    candidate_inputs = data.loc[candidate_mask, input_columns].copy()

    return train_inputs, train_targets, candidate_inputs, tested_mask, candidate_mask


# Module-level helper functions for acquisition calculations
def _calculate_ucb(mu, sigma, kappa_adjusted):
    return mu + kappa_adjusted * sigma

def _calculate_ei(mu, sigma, y_max_of_current_obj, xi_adjusted):
    sigma_safe = np.maximum(sigma, 1e-9) # Avoid division by zero / sqrt of negative
    imp = mu - y_max_of_current_obj - xi_adjusted
    z = imp / sigma_safe
    ei = imp * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei[imp < 0] = 0.0 # Expected improvement cannot be negative
    return ei

def _calculate_pi(mu, sigma, y_max_of_current_obj, xi_adjusted):
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (mu - y_max_of_current_obj - xi_adjusted) / sigma_safe
    return norm.cdf(z)


class BayesianOptimizer:

    
    def __init__(self, surrogate_model=None, bounds=None, kernel=None, alpha=1e-6, n_restarts=10, normalize_y=True, target_index_for_surrogate=0):

        self.surrogate_model = surrogate_model
        self.bounds = bounds
        self.X_train = None
        self.y_train = None # Will store original scale y_train for acquisition functions like EI/PI
        self.is_fitted = False
        self.target_index = target_index_for_surrogate

        if self.surrogate_model is not None:
            if not hasattr(self.surrogate_model, 'predict_with_uncertainty') or \
               not callable(self.surrogate_model.predict_with_uncertainty):
                raise ValueError("Provided surrogate_model must have a 'predict_with_uncertainty' method.")
            self.gp = None
            self.kernel = None
        else:
            if kernel is None:
                self.kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
            else:
                self.kernel = kernel
            self.gp = GaussianProcessRegressor(
                kernel=self.kernel, alpha=alpha, n_restarts_optimizer=n_restarts,
                normalize_y=normalize_y, random_state=42
            )
    
    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray):

        # Handle 1D y
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Store original X and y for reference and potential use by surrogates
        # self.X_raw_train = X
        # self.y_raw_train = y

        # Filter out NaN/Inf values from y for fitting GP or storing y_train
        valid_idx = np.isfinite(y).all(axis=1)
        X_valid_np = X[valid_idx] if isinstance(X, np.ndarray) else X.iloc[valid_idx].values
        y_valid = y[valid_idx]
        
        if len(X_valid_np) == 0:
            raise ValueError("No valid data points after filtering NaN/Inf values for y.")
        
        # Store X_train as DataFrame if original X was a DataFrame, otherwise as numpy.
        # This helps in _get_surrogate_prediction if column names are needed.
        if isinstance(X, pd.DataFrame):
            self.X_train = X.iloc[valid_idx].copy()
        else:
            self.X_train = X_valid_np
        
        # Store original-scale y_train
        self.y_train = y_valid
        
        # Fit internal GP if no external surrogate_model is provided
        if self.surrogate_model is None:
            self.gp.fit(X_valid_np, y_valid[:, self.target_index])
        
        self.is_fitted = True

    def _get_surrogate_prediction(self, X: pd.DataFrame | np.ndarray, return_posterior_samples: bool = False):
  
        if not self.is_fitted:
            raise ValueError("BayesianOptimizer must be fitted before calling _get_surrogate_prediction.")
        
        # If we have an external surrogate model, use its predict_with_uncertainty
        if self.surrogate_model is not None:
            # Determine input_columns if we have a scaler with feature_names
            input_columns = None
            if hasattr(self.surrogate_model, 'scaler_x') and hasattr(self.surrogate_model.scaler_x, 'feature_names_in_'):
                input_columns = list(self.surrogate_model.scaler_x.feature_names_in_)
            
            # Ensure DataFrame if surrogate expects named columns
            if isinstance(X, np.ndarray) and input_columns is not None:
                X_df = pd.DataFrame(X, columns=input_columns)
            else:
                X_df = X
            
            # Choose how many MC samples to request if return_posterior_samples
            num_samples = 200 if return_posterior_samples else None
            
            preds = self.surrogate_model.predict_with_uncertainty(X_df, input_columns=input_columns, num_samples=num_samples)
            
            if return_posterior_samples:
                # Expect (mean_preds, std_devs, posterior_samples)
                if len(preds) == 3:
                    mean_preds, std_devs, posterior_samples = preds
                else:
                    mean_preds, std_devs = preds
                    posterior_samples = None
            else:
                # Expect (mean_preds, std_devs)
                mean_preds, std_devs = preds
                posterior_samples = None

            # Ensure shapes
            mean_preds = np.atleast_2d(mean_preds)
            std_devs = np.atleast_2d(std_devs)

            # If multiple surrogate targets exist, we might select one via target_index
            if mean_preds.ndim == 2 and mean_preds.shape[1] > 1:
                mean_target = mean_preds[:, self.target_index:self.target_index+1]
                std_target = std_devs[:, self.target_index:self.target_index+1]
            else:
                mean_target = mean_preds.reshape(-1, 1)
                std_target = std_devs.reshape(-1, 1)
            
            return mean_target, std_target, posterior_samples
        
        # Otherwise use internal GP
        X_np = X if isinstance(X, np.ndarray) else X.values
        mu, sigma = self.gp.predict(X_np, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        
        return mu, sigma, None

    def acquisition_function(
        self,
        X: pd.DataFrame | np.ndarray,
        acquisition: str = "UCB",
        xi: float = 0.01,
        kappa: float = 2.5,
        curiosity: float = 0.0
    ) -> np.ndarray:
 
        if not self.is_fitted: # This check is also in _get_surrogate_prediction, but good for early exit.
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Get predictions and uncertainties using the new unified method
        # We don't need posterior samples for these standard acquisition functions.
        mu, sigma, _ = self._get_surrogate_prediction(X_np, return_posterior_samples=False)

        # Ensure sigma is non-negative (it's std_dev) and not zero to avoid division errors.
        sigma = np.maximum(sigma, 1e-9) # Changed from 1e-6 to 1e-9 for potentially smaller std devs

        # Adjust parameters based on curiosity
        kappa_adjusted = kappa * (1.0 + 0.5 * curiosity)
        xi_adjusted = xi * (1.0 + 0.5 * curiosity)
        
        # Compute acquisition function based on type
        if acquisition == "UCB":
            # Upper Confidence Bound
            return mu + kappa_adjusted * sigma

        elif acquisition == "EI":
            # Expected Improvement
            y_max = np.max(self.y_train)

            # Calculate improvement
            imp = mu - y_max - xi_adjusted

            # Calculate Z-score
            z = imp / sigma

            # Calculate Expected Improvement
            ei = imp * norm.cdf(z) + sigma * norm.pdf(z)

            # Set EI to 0 where it's negative
            ei[imp < 0] = 0.0

            return ei

        elif acquisition == "PI":
            # Probability of Improvement
            y_max = np.max(self.y_train)

            # Calculate Z-score
            z = (mu - y_max - xi_adjusted) / sigma

            # Calculate Probability of Improvement
            return norm.cdf(z)

        elif acquisition == "MaxEntropy":
            # Maximum Entropy (pure exploration)
            return sigma

        else:
            raise ValueError(f"Unknown acquisition function: {acquisition}")

    def optimize(self, bounds_dict: dict[str, tuple[float, float]], n_restarts: int = 20,
                 acquisition: str = "UCB", curiosity: float = 0.0,
                 n_points: int = 1, min_distance: float = 0.1, max_iter: int = 500) -> tuple[np.ndarray, np.ndarray]:

        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Extract bounds as list of (lower, upper) tuples
        feature_names = list(bounds_dict.keys())
        bounds = [bounds_dict[name] for name in feature_names]

        # Function to minimize (negative of acquisition function)
        def objective(x):
            x_reshaped = x.reshape(1, -1)
            return -self.acquisition_function(x_reshaped, acquisition, curiosity=curiosity).item()

        # Initialize best points and values
        X_best = []
        values_best = []

        # Run optimization from multiple starting points
        for _ in range(n_restarts):
            # Generate random starting point
            x0 = np.array([np.random.uniform(low, high) for low, high in bounds])

            # Run optimization
            result = minimize(
                objective,
                x0,
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter": max_iter}
            )

            # Get optimal point and value
            if result.success:
                X_best.append(result.x)
                values_best.append(-result.fun)

        # Convert to arrays
        X_best = np.array(X_best)
        values_best = np.array(values_best)

        # Sort by acquisition function value
        sorted_indices = np.argsort(-values_best)
        X_best = X_best[sorted_indices]
        values_best = values_best[sorted_indices]

        # Enforce diversity between returned points
        if n_points > 1:
            # Initialize with the best point
            diverse_indices = [0]

            # Add diverse points
            for i in range(1, len(X_best)):
                is_diverse = True

                # Check distance to all selected points
                for j in diverse_indices:
                    distance = np.linalg.norm(X_best[i] - X_best[j])

                    if distance < min_distance:
                        is_diverse = False
                        break

                # Add if diverse enough
                if is_diverse:
                    diverse_indices.append(i)

                    # Break if we have enough points
                    if len(diverse_indices) >= n_points:
                        break

            # Select diverse points
            X_diverse = X_best[diverse_indices]
            values_diverse = values_best[diverse_indices]

            return X_diverse, values_diverse

        # Return top n_points
        return X_best[:n_points], values_best[:n_points]


def multi_objective_bayesian_optimization(
    train_inputs,
    train_targets,
    candidate_inputs,
    weights,
    max_or_min,
    curiosity: float = 0.0,
    acquisition: str = "UCB",
    strategy: str = "weighted_sum",
    surrogate_model=None,
    input_columns=None
):
    # Handle data dimensionality
    train_targets = np.array(train_targets)
    weights = np.array(weights)

    if train_targets.ndim == 1:
        train_targets = train_targets.reshape(-1, 1)
        weights = np.array([1.0])
        max_or_min = ["max"]

    n_objectives_from_targets = train_targets.shape[1]

    # Filter out invalid targets
    valid_indices = np.isfinite(train_targets).all(axis=1)
    train_targets = train_targets[valid_indices]

    # Handle train_inputs being a DataFrame or numpy array
    if isinstance(train_inputs, pd.DataFrame):
        train_inputs = train_inputs.iloc[valid_indices]
    else:
        train_inputs = train_inputs[valid_indices]

    if len(train_inputs) == 0:
        st.warning("No valid training data. Returning random scores.")
        return np.random.rand(len(candidate_inputs))

    # Determine effective weights for scalarization
    if strategy == "parego":
        if n_objectives_from_targets > 0:
            random_w = np.random.dirichlet(np.ones(n_objectives_from_targets), size=1).ravel()
            effective_weights = random_w
            st.info(f"ParEGO strategy: using random weights: {np.round(effective_weights, 3)}")
        else:
            effective_weights = np.array([1.0]) # Fallback
    elif strategy == "pareto":
        effective_weights = None # Weights are not used in this strategy
    else: # 'weighted_sum' or default
        effective_weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(len(weights)) / len(weights)

    # Prepare candidate inputs
    if isinstance(candidate_inputs, np.ndarray) and input_columns:
        candidate_inputs_df = pd.DataFrame(candidate_inputs, columns=input_columns)
    elif isinstance(candidate_inputs, pd.DataFrame):
        candidate_inputs_df = candidate_inputs
    else:
        raise ValueError("candidate_inputs format issue or missing input_columns.")

    num_candidates = len(candidate_inputs_df)
    acq_values_total = np.zeros(num_candidates)

    # Loop over each objective, build/acquire its own acquisition, and combine
    for obj_idx in range(n_objectives_from_targets):
        y_obj = train_targets[:, obj_idx]

        # Multi-objective: transform y for minimization problems
        if max_or_min[obj_idx].lower().startswith("min"):
            y_obj = -y_obj

        # Internal BO or external surrogate
        if surrogate_model is None:
            # Fit a separate GP for each objective
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                n_restarts_optimizer=5,
                normalize_y=True,
                random_state=42
            )

            if isinstance(train_inputs, pd.DataFrame):
                X_train_np = train_inputs.values
            else:
                X_train_np = train_inputs

            gp.fit(X_train_np, y_obj)

            if isinstance(candidate_inputs_df, pd.DataFrame):
                X_cand_np = candidate_inputs_df.values
            else:
                X_cand_np = candidate_inputs_df

            mu, sigma = gp.predict(X_cand_np, return_std=True)
            mu = mu.reshape(-1, 1)
            sigma = sigma.reshape(-1, 1)

            sigma_safe = np.maximum(sigma, 1e-9)

            # Adjust parameters based on curiosity
            kappa = 2.5
            xi = 0.01
            kappa_adjusted = kappa * (1.0 + 0.5 * curiosity)
            xi_adjusted = xi * (1.0 + 0.5 * curiosity)

            if acquisition == "UCB":
                acq_obj = _calculate_ucb(mu, sigma_safe, kappa_adjusted)
            elif acquisition == "EI":
                acq_obj = _calculate_ei(mu, sigma_safe, np.max(y_obj), xi_adjusted)
            elif acquisition == "PI":
                acq_obj = _calculate_pi(mu, sigma_safe, np.max(y_obj), xi_adjusted)
            else:
                # MaxEntropy or fallback
                acq_obj = sigma_safe

        else:
            # Use provided surrogate model
            # The surrogate should have predict_with_uncertainty method
            if not hasattr(surrogate_model, 'predict_with_uncertainty'):
                raise ValueError("Surrogate model must have 'predict_with_uncertainty' method")
            
            # Get predictions for all objectives
            mu_all, sigma_all, _ = surrogate_model.predict_with_uncertainty(
                candidate_inputs_df, 
                input_columns=input_columns,
                num_samples=None  # Use default
            )
            
            # Extract predictions for this specific objective
            if mu_all.ndim == 1 or mu_all.shape[1] == 1:
                # Single objective output
                mu = mu_all.reshape(-1, 1)
                sigma = sigma_all.reshape(-1, 1)
            else:
                # Multi-objective output - select the current objective
                mu = mu_all[:, obj_idx:obj_idx+1]
                sigma = sigma_all[:, obj_idx:obj_idx+1]
            
            # Transform for minimization if needed
            if max_or_min[obj_idx].lower().startswith("min"):
                mu = -mu
            
            sigma_safe = np.maximum(sigma, 1e-9)
            
            # Adjust parameters based on curiosity
            kappa = 2.5
            xi = 0.01
            kappa_adjusted = kappa * (1.0 + 0.5 * curiosity)
            xi_adjusted = xi * (1.0 + 0.5 * curiosity)
            
            if acquisition == "UCB":
                acq_obj = _calculate_ucb(mu, sigma_safe, kappa_adjusted)
            elif acquisition == "EI":
                # For EI, we need the max of training targets for this objective
                acq_obj = _calculate_ei(mu, sigma_safe, np.max(y_obj), xi_adjusted)
            elif acquisition == "PI":
                acq_obj = _calculate_pi(mu, sigma_safe, np.max(y_obj), xi_adjusted)
            else:
                # MaxEntropy or fallback
                acq_obj = sigma_safe

        # Weight this objective's acquisition and add
        w = effective_weights[obj_idx] if effective_weights is not None else 1.0
        acq_values_total += w * acq_obj.flatten()

    # Novelty / diversity term
    try:
        novelty_scores = calculate_novelty(candidate_inputs_df)
        if novelty_scores is not None and np.all(np.isfinite(novelty_scores)):
            novelty_scores = (novelty_scores - np.min(novelty_scores)) / (np.max(novelty_scores) - np.min(novelty_scores) + 1e-12)
            acq_values_total += 0.1 * curiosity * novelty_scores
    except Exception as e:
        st.warning(f"Novelty calculation failed: {e}")

    return acq_values_total.flatten()
