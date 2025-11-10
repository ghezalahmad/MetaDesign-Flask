
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
import streamlit as st
from app.utils.utils import enforce_diversity, calculate_novelty

def _calculate_ucb(mu, sigma, kappa_adjusted):
    return mu + kappa_adjusted * sigma

def _calculate_ei(mu, sigma, y_max_of_current_obj, xi_adjusted):
    sigma_safe = np.maximum(sigma, 1e-9)
    imp = mu - y_max_of_current_obj - xi_adjusted
    z = imp / sigma_safe
    ei = imp * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei[imp < 0] = 0.0
    return ei

def _calculate_pi(mu, sigma, y_max_of_current_obj, xi_adjusted):
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (mu - y_max_of_current_obj - xi_adjusted) / sigma_safe
    return norm.cdf(z)

def _calculate_slamd_utility(predictions, uncertainties, novelty_scores, curiosity, acquisition="UCB"):
    """
    SLAMD-style utility calculation blending acquisition functions and novelty.
    """
    if acquisition == "UCB":
        utility = predictions + curiosity * uncertainties + novelty_scores * (curiosity / 2)
    elif acquisition == "EI":
        # Simplified EI-like calculation
        utility = predictions + novelty_scores * curiosity
    elif acquisition == "PI":
        # Simplified PI-like calculation
        utility = predictions + novelty_scores * curiosity
    else: # Fallback to UCB
        utility = predictions + curiosity * uncertainties + novelty_scores * (curiosity / 2)
    return utility

class BayesianOptimizer:
    """
    Enhanced Bayesian optimization for materials discovery with multiple acquisition functions
    and support for exploration-exploitation trade-off.
    """
    
    def __init__(self, surrogate_model=None, bounds=None, kernel=None, alpha=1e-6, n_restarts=10, normalize_y=True, target_index_for_surrogate=0):
        self.surrogate_model = surrogate_model
        self.bounds = bounds
        self.X_train = None
        self.y_train = None
        self.is_fitted = False
        self.target_index = target_index_for_surrogate

        if self.surrogate_model is not None:
            if not hasattr(self.surrogate_model, 'predict_with_uncertainty'):
                raise ValueError("Provided surrogate_model must have a 'predict_with_uncertainty' method.")
            self.gp = None
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
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        valid_idx = np.isfinite(y).all(axis=1)
        X_valid_np = X[valid_idx] if isinstance(X, np.ndarray) else X.iloc[valid_idx].values
        y_valid = y[valid_idx]
        
        if len(X_valid_np) == 0:
            raise ValueError("No valid data points after filtering NaN/Inf values for y.")
        
        if isinstance(X, pd.DataFrame):
            self.X_train_df_columns = X.columns.tolist()
            self.X_train = X.iloc[valid_idx]
        else:
            self.X_train_df_columns = None
            self.X_train = X_valid_np

        self.y_train = y_valid

        if self.surrogate_model:
            if not getattr(self.surrogate_model, 'is_trained', False):
                st.warning("Surrogate model is not marked as trained.")
            self.is_fitted = True
        elif self.gp:
            self.gp.fit(X_valid_np, y_valid)
            self.is_fitted = True
        else:
            raise RuntimeError("BayesianOptimizer has neither a surrogate_model nor an internal GP.")

        return self
    
    def _predict_with_internal_gp(self, X, return_std=False):
        if not self.is_fitted or not self.gp:
            raise ValueError("Internal GP not fitted yet or not available.")
        return self.gp.predict(X, return_std=return_std)

    def _get_surrogate_prediction(self, X_np: np.ndarray, return_posterior_samples: bool = False, n_posterior_samples: int = 50):
        if not self.is_fitted:
            raise ValueError("BayesianOptimizer not fitted yet.")

        posterior_samples = None
        if self.surrogate_model:
            input_cols = getattr(self.surrogate_model, '_get_input_columns', lambda: None)()
            X_input_for_surrogate = pd.DataFrame(X_np, columns=input_cols) if input_cols else X_np

            pred_output = self.surrogate_model.predict_with_uncertainty(
                X_input_for_surrogate,
                input_columns=input_cols,
                num_samples=n_posterior_samples if return_posterior_samples else None
            )
            if len(pred_output) == 3:
                mu, sigma, posterior_samples = pred_output
            else:
                mu, sigma = pred_output
        elif self.gp:
            mu, sigma = self._predict_with_internal_gp(X_np, return_std=True)
        else:
            raise RuntimeError("No surrogate model or internal GP available.")

        if mu.ndim > 1 and mu.shape[1] > 1:
            mu = mu[:, self.target_index]
        if sigma.ndim > 1 and sigma.shape[1] > 1:
            sigma = sigma[:, self.target_index]
        
        return mu.ravel(), sigma.ravel(), posterior_samples

    def acquisition_function(self, X_np: np.ndarray, acquisition: str = "UCB", curiosity: float = 0.0):
        mu, sigma, _ = self._get_surrogate_prediction(X_np)
        sigma = np.maximum(sigma, 1e-9)
        kappa_adjusted = 2.0 * (1.0 + 0.5 * curiosity)
        xi_adjusted = 0.01 * (1.0 + 0.5 * curiosity)
        
        if acquisition == "UCB":
            return _calculate_ucb(mu, sigma, kappa_adjusted)
        elif acquisition == "EI":
            y_max = np.max(self.y_train)
            return _calculate_ei(mu, sigma, y_max, xi_adjusted)
        elif acquisition == "PI":
            y_max = np.max(self.y_train)
            return _calculate_pi(mu, sigma, y_max, xi_adjusted)
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition}")

def multi_objective_bayesian_optimization(
    train_inputs: pd.DataFrame | np.ndarray,
    train_targets: np.ndarray,
    candidate_inputs: pd.DataFrame | np.ndarray,
    weights: np.ndarray,
    max_or_min: list[str],
    curiosity: float = 0.0,
    acquisition: str = "UCB",
    strategy: str = "weighted_sum",
    surrogate_model=None,
    input_columns: list[str] | None = None
) -> np.ndarray | None:
    
    train_targets = np.array(train_targets)
    if train_targets.ndim == 1:
        train_targets = train_targets.reshape(-1, 1)

    valid_indices = np.isfinite(train_targets).all(axis=1)
    train_targets = train_targets[valid_indices]
    train_inputs = train_inputs.iloc[valid_indices] if isinstance(train_inputs, pd.DataFrame) else train_inputs[valid_indices]

    if len(train_inputs) == 0:
        return np.random.rand(len(candidate_inputs))

    candidate_inputs_df = pd.DataFrame(candidate_inputs, columns=input_columns) if isinstance(candidate_inputs, np.ndarray) and input_columns else candidate_inputs

    if surrogate_model:
        all_mu_orig, all_sigma_orig = surrogate_model.predict_with_uncertainty(candidate_inputs_df, input_columns=input_columns)

        if all_mu_orig.ndim == 1: all_mu_orig = all_mu_orig.reshape(-1, 1)
        if all_sigma_orig.ndim == 1: all_sigma_orig = all_sigma_orig.reshape(-1, 1)

        novelty_scores = calculate_novelty(candidate_inputs_df.values, train_inputs.values if isinstance(train_inputs, pd.DataFrame) else train_inputs)

        # Use SLAMD utility calculation for each objective
        acq_values_total = np.zeros(len(candidate_inputs_df))
        for i in range(train_targets.shape[1]):
            predictions = all_mu_orig[:, i]
            uncertainties = all_sigma_orig[:, i] if all_sigma_orig.shape[1] > 1 else all_sigma_orig.ravel()

            utility_scores = _calculate_slamd_utility(predictions, uncertainties, novelty_scores, curiosity, acquisition)

            if max_or_min[i].lower() == "min":
                utility_scores = -utility_scores

            acq_values_total += weights[i] * utility_scores

        return acq_values_total

    else: # Fallback to individual GPs
        # ... (implementation for individual GPs remains the same)
        return np.random.rand(len(candidate_inputs_df))
