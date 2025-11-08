import numpy as np
from scipy.spatial import distance_matrix
import random
import torch
import torch.optim as optim
from scipy.stats import norm

# Force deterministic behavior in PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def calculate_utility(predictions: np.ndarray, uncertainties: np.ndarray | None, novelty: np.ndarray | None,
                      curiosity: float, weights: np.ndarray, max_or_min: list[str],
                      thresholds: list[float | None] | None = None,
                      acquisition: str = "UCB", for_visualization: bool = False) -> np.ndarray:
    """
    Calculates utility scores for candidate samples based on their predicted properties,
    uncertainties, novelty, and a specified acquisition function.

    This is the primary utility calculation function used across different models.

    Args:
        predictions: Numpy array of predicted target values. Shape (n_samples, n_targets).
        uncertainties: Numpy array of uncertainty estimates for predictions.
                       Shape (n_samples, 1) or (n_samples,). Can be None.
        novelty: Numpy array of novelty scores. Shape (n_samples, 1) or (n_samples,). Can be None.
        curiosity: Factor (-2 to +2) to balance exploration vs. exploitation.
                   Higher values increase exploration.
        weights: Numpy array of weights for each target property. Shape (n_targets,).
        max_or_min: List of strings ('max' or 'min') indicating optimization
                    direction for each target.
        thresholds: Optional list of threshold values for each target. Samples not
                    meeting these are penalized. E.g. [10.0, None, <5.0].
        acquisition: Name of the acquisition function ("UCB", "EI", "PI", "MaxEntropy").
        for_visualization: If True, returns raw utility scores suitable for direct plotting.
                           If False, applies a log1p transformation for ranking.

    Returns:
        Numpy array of utility scores. Shape (n_samples,).
    """
    valid_acquisitions = ["UCB", "PI", "MaxEntropy", "EI"]
    if acquisition not in valid_acquisitions:
        acquisition = "UCB"
    
    predictions = np.array(predictions)
    if np.isnan(predictions).any() or np.isinf(predictions).any():
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if uncertainties is None:
        uncertainties = np.ones((predictions.shape[0], 1)) * 0.1
    else:
        uncertainties = np.array(uncertainties)
        if uncertainties.ndim > 1:
            # If multi-dimensional, average across objectives for a single uncertainty score per sample
            uncertainties = np.mean(uncertainties, axis=1)
        if uncertainties.ndim == 1:
            uncertainties = uncertainties.reshape(-1, 1)
        uncertainties = np.nan_to_num(uncertainties, nan=0.1, posinf=1.0, neginf=0.1)
    
    if novelty is None:
        novelty = np.zeros((predictions.shape[0], 1))
    else:
        novelty = np.array(novelty)
        if novelty.ndim == 1:
            novelty = novelty.reshape(-1, 1)

    weights = np.array(weights).reshape(1, -1)

    # Normalize predictions to [0, 1]
    min_vals = np.min(predictions, axis=0, keepdims=True)
    max_vals = np.max(predictions, axis=0, keepdims=True)
    range_vals = np.where((max_vals - min_vals) < 1e-10, 1.0, (max_vals - min_vals))
    norm_predictions = (predictions - min_vals) / range_vals

    for i, direction in enumerate(max_or_min):
        if direction == "min":
            norm_predictions[:, i] = 1.0 - norm_predictions[:, i]

    # Ensure weights is 1D array for matmul if it's (1, n_obj)
    if weights.ndim > 1:
        weights = weights.flatten()

    # Weighted sum of normalized predictions for the exploitation part of utility
    # norm_predictions is (n_samples, n_objectives), weights is (n_objectives,)
    # exploitation_term is (n_samples,)
    exploitation_term = norm_predictions @ weights

    # Apply thresholds (optional)
    if thresholds is not None:
        thresholds_arr = np.array([t if t is not None else (-np.inf if m == 'max' else np.inf) for t, m in zip(thresholds, max_or_min)])
        penalty = np.ones(predictions.shape[0])
        for i, direction in enumerate(max_or_min):
            if not np.isnan(thresholds_arr[i]):
                if direction == "max":
                    penalty *= (predictions[:, i] >= thresholds_arr[i])
                else: # min
                    penalty *= (predictions[:, i] <= thresholds_arr[i])
        # Apply penalty: full exploitation_term if all thresholds met, 0 otherwise (or could be a factor)
        # For simplicity, making it a hard penalty. Could be softened.
        exploitation_term = exploitation_term * penalty


    norm_uncertainties = (uncertainties / (np.max(uncertainties) + 1e-10)).flatten()
    norm_novelty = (novelty / (np.max(novelty) + 1e-10)).flatten()

    curiosity_factor = np.clip(1.0 + 0.5 * curiosity, 0.1, 2.0) # General factor for exploration strength

    # Specific scaling for UCB's kappa, EI's exploration part, etc.
    # These are typical values, can be tuned or made parameters
    kappa_ucb = 0.5
    exploration_bonus_novelty = 0.1
    exploration_bonus_uncertainty = 0.2

    if acquisition == "UCB":
        utility = exploitation_term + kappa_ucb * curiosity_factor * norm_uncertainties
        if curiosity > 0: # Add novelty bonus if exploring
            utility += exploration_bonus_novelty * curiosity_factor * norm_novelty

    elif acquisition == "EI":
        # Ensure exploitation_term is (n_samples,) for EI/PI logic
        best_observed_exploitation = np.max(exploitation_term) if len(exploitation_term) > 0 else 0.0

        improvement = np.maximum(0, exploitation_term - best_observed_exploitation)
        sigma_eff = norm_uncertainties + 1e-10 # Effective sigma for EI/PI

        z = improvement / sigma_eff
        ei_base = improvement * norm.cdf(z) + sigma_eff * norm.pdf(z)

        # Add curiosity-scaled exploration terms
        utility = ei_base
        if curiosity > 0:
            utility = ei_base * (1 + exploration_bonus_uncertainty * curiosity) + exploration_bonus_novelty * curiosity * norm_novelty
        else: # If not curious, EI focuses on exploitation based on current model
            utility = ei_base


    elif acquisition == "PI":
        best_observed_exploitation = np.max(exploitation_term) if len(exploitation_term) > 0 else 0.0
        sigma_eff = norm_uncertainties + 1e-10 # Effective sigma for EI/PI

        z = (exploitation_term - best_observed_exploitation) / sigma_eff
        pi_base = norm.cdf(z)

        utility = pi_base
        if curiosity > 0:
             utility = pi_base * (1 + exploration_bonus_uncertainty * curiosity) + exploration_bonus_novelty * curiosity * norm_novelty
        else:
             utility = pi_base

    elif acquisition == "MaxEntropy":
        # MaxEntropy focuses on exploration (uncertainty and novelty)
        # Small weight to exploitation_term if curiosity allows some exploitation
        exploitation_weight_max_entropy = 0.1 if curiosity < 0.5 else 0.0 # Only consider exploitation if not strongly exploring
        utility = (exploration_bonus_uncertainty * norm_uncertainties + exploration_bonus_novelty * norm_novelty) * curiosity_factor \
                  + exploitation_weight_max_entropy * exploitation_term
    else: # Fallback, should not happen due to check at the beginning
        utility = exploitation_term

    # Ensure utility is a 1D array
    utility = utility.flatten()

    if for_visualization:
        return np.clip(utility, 0, None) # Clip at 0 for visualization if utility can be negative
    else:
        # Clip to ensure positive values before log1p, then apply log transform
        return np.log1p(np.clip(utility, 1e-8, None))


def calculate_novelty(features: np.ndarray, labeled_features: np.ndarray | None) -> np.ndarray:
    """
    Calculates novelty scores for a set of feature vectors based on their
    minimum Euclidean distance to a set of labeled feature vectors.

    Novelty is normalized by the maximum observed minimum distance. Higher values
    indicate greater novelty (further from known samples).

    Args:
        features: Numpy array of feature vectors for which to calculate novelty.
                  Shape (n_unlabeled_samples, n_features).
        labeled_features: Numpy array of feature vectors of already known/labeled samples.
                          Shape (n_labeled_samples, n_features). Can be None or empty.

    Returns:
        A numpy array of novelty scores (between 0 and 1), one for each sample
        in `features`. Returns an array of ones if `labeled_features` is empty or None.
    """
    # Handle case where no labeled features exist
    if labeled_features is None or labeled_features.shape[0] == 0:
        return np.ones(features.shape[0])
    
    # Check for NaN/Inf values and replace them
    features = np.nan_to_num(features, nan=0.0)
    labeled_features = np.nan_to_num(labeled_features, nan=0.0)
    
    # Calculate distance matrix
    distances = distance_matrix(features, labeled_features)
    
    # Minimum distance to any labeled sample
    min_distances = distances.min(axis=1)
    
    # Normalize by maximum distance to avoid scale issues
    max_distance = min_distances.max()
    if max_distance > 0:
        novelty = min_distances / (max_distance + 1e-6)
    else:
        novelty = np.ones_like(min_distances)
    
    return novelty


# Learning Rate Scheduler Configuration
def initialize_scheduler(optimizer, scheduler_type, **kwargs):
   
    if scheduler_type == "CosineAnnealing":
        # Cosine annealing with warm restarts
        T_max = kwargs.get("T_max", 100)
        eta_min = kwargs.get("eta_min", 1e-5)
        T_mult = kwargs.get("T_mult", 2)
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_max, T_mult=T_mult, eta_min=eta_min
        )
    
    elif scheduler_type == "ReduceLROnPlateau":
        # Reduce learning rate when a metric stops improving
        # Explicitly define all parameters for clarity and to avoid TypeErrors
        # from unexpected interactions with kwargs if a parameter is missing.
        factor = float(kwargs.get("factor", 0.5))
        patience = int(kwargs.get("patience", 10)) # Default from PyTorch docs
        threshold = float(kwargs.get("threshold", 1e-4))
        threshold_mode = str(kwargs.get("threshold_mode", 'rel'))
        cooldown = int(kwargs.get("cooldown", 0))
        min_lr = float(kwargs.get("min_lr", 0)) # Can also be a list
        eps = float(kwargs.get("eps", 1e-8))
        verbose = bool(kwargs.get("verbose", False))

        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps
            # verbose argument removed for simplicity
        )
    
    elif scheduler_type == "StepLR":
        # Step learning rate by a factor every n epochs
        step_size = kwargs.get("step_size", 30)
        gamma = kwargs.get("gamma", 0.1)
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    
    elif scheduler_type == "Adaptive":
        # Custom adaptive learning rate based on validation loss
        class AdaptiveLR:
            def __init__(self, optimizer, factor=0.5, patience=5, min_lr=1e-6):
                self.optimizer = optimizer
                self.factor = factor
                self.patience = patience
                self.min_lr = min_lr
                self.best_loss = float('inf')
                self.bad_epochs = 0
                
            def step(self, metrics=None):
                # If metrics provided, check if we should reduce LR
                if metrics is not None:
                    if metrics < self.best_loss - 1e-4:
                        self.best_loss = metrics
                        self.bad_epochs = 0
                    else:
                        self.bad_epochs += 1
                        
                    if self.bad_epochs >= self.patience:
                        for param_group in self.optimizer.param_groups:
                            old_lr = param_group['lr']
                            new_lr = max(old_lr * self.factor, self.min_lr)
                            param_group['lr'] = new_lr
                            print(f"Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")
                        self.bad_epochs = 0
            
            def get_last_lr(self):
                return [group['lr'] for group in self.optimizer.param_groups]
        
        factor = kwargs.get("factor", 0.5)
        patience = kwargs.get("patience", 5)
        min_lr = kwargs.get("min_lr", 1e-6)
        return AdaptiveLR(optimizer, factor=factor, patience=patience, min_lr=min_lr)
    
    # Default to no scheduler
    return None


# Enforce diversity in sample selection
def enforce_diversity(candidate_inputs, selected_inputs, min_distance=0.1):
   
    if len(selected_inputs) == 0:
        return candidate_inputs
    
    # Compute distances between candidates and selected samples
    distances = distance_matrix(candidate_inputs, selected_inputs)
    
    # Identify candidates that are sufficiently distant from all selected samples
    diverse_indices = np.where(np.min(distances, axis=1) > min_distance)[0]
    
    # If no candidates meet the diversity criterion, gradually reduce distance threshold
    if len(diverse_indices) == 0:
        for reduction in [0.75, 0.5, 0.25, 0.1, 0.05]:
            reduced_threshold = min_distance * reduction
            diverse_indices = np.where(np.min(distances, axis=1) > reduced_threshold)[0]
            if len(diverse_indices) > 0:
                break
    
    # If still no diverse candidates, return the most distant candidates
    if len(diverse_indices) == 0:
        max_min_distance_idx = np.argmax(np.min(distances, axis=1))
        return candidate_inputs[np.array([max_min_distance_idx])]
    
    return candidate_inputs[diverse_indices]


# Balance exploration and exploitation
def balance_exploration_exploitation(utility_scores, uncertainty_scores, novelty_scores, curiosity):
    """
    Adjust utility scores based on exploration-exploitation balance.
    
    Parameters:
    -----------
    utility_scores : numpy.ndarray
        Base utility scores
    uncertainty_scores : numpy.ndarray
        Uncertainty scores for each candidate
    novelty_scores : numpy.ndarray
        Novelty scores for each candidate
    curiosity : float
        Exploration vs exploitation parameter (-2 to +2)
        
    Returns:
    --------
    balanced_scores : numpy.ndarray
        Utility scores adjusted for exploration-exploitation balance
    """
    # Convert to numpy arrays and ensure correct shape
    utility_scores = np.array(utility_scores).flatten()
    uncertainty_scores = np.array(uncertainty_scores).flatten()
    novelty_scores = np.array(novelty_scores).flatten()
    
    # Normalize all scores to [0, 1]
    norm_utility = utility_scores / (np.max(utility_scores) + 1e-10)
    norm_uncertainty = uncertainty_scores / (np.max(uncertainty_scores) + 1e-10)
    norm_novelty = novelty_scores / (np.max(novelty_scores) + 1e-10)
    
    # Create exploration score as combination of uncertainty and novelty
    exploration_score = 0.7 * norm_uncertainty + 0.3 * norm_novelty
    
    # Map curiosity from [-2, 2] to [0, 1] for weighting
    # -2: pure exploitation, +2: pure exploration
    exploration_weight = (curiosity + 2) / 4.0
    
    # Calculate balanced score as weighted combination
    balanced_scores = (1 - exploration_weight) * norm_utility + exploration_weight * exploration_score
    
    return balanced_scores


# Generate diverse batch of candidates
def generate_diverse_batch(utility_scores, features, batch_size=5, diversity_weight=0.5):
    """
    Select a diverse batch of candidates with high utility.
    
    Parameters:
    -----------
    utility_scores : numpy.ndarray
        Utility scores for candidates
    features : numpy.ndarray
        Feature vectors for candidates
    batch_size : int
        Number of candidates to select
    diversity_weight : float
        Weight for diversity vs utility (0-1)
        
    Returns:
    --------
    selected_indices : numpy.ndarray
        Indices of selected candidates
    """
    if batch_size >= len(utility_scores):
        return np.arange(len(utility_scores))
    
    # Normalize utility scores
    norm_utility = utility_scores / (np.max(utility_scores) + 1e-10)
    
    selected_indices = []
    remaining_indices = np.arange(len(utility_scores))
    
    # Select first candidate based on highest utility
    first_idx = np.argmax(norm_utility)
    selected_indices.append(first_idx)
    remaining_indices = np.setdiff1d(remaining_indices, selected_indices)
    
    # Select remaining candidates
    for _ in range(batch_size - 1):
        if len(remaining_indices) == 0:
            break
            
        # Calculate distances to already selected candidates
        remaining_features = features[remaining_indices]
        selected_features = features[selected_indices]
        
        distances = distance_matrix(remaining_features, selected_features)
        min_distances = distances.min(axis=1)
        
        # Normalize distances
        norm_distances = min_distances / (np.max(min_distances) + 1e-10)
        
        # Calculate combined score (higher is better)
        combined_scores = (1 - diversity_weight) * norm_utility[remaining_indices] + \
                         diversity_weight * norm_distances
        
        # Select candidate with highest combined score
        next_local_idx = np.argmax(combined_scores)
        next_global_idx = remaining_indices[next_local_idx]
        
        selected_indices.append(next_global_idx)
        remaining_indices = np.setdiff1d(remaining_indices, [next_global_idx])
    
    return np.array(selected_indices)


# Pareto front identification
def identify_pareto_front(predictions: np.ndarray, max_or_min: list[str]) -> np.ndarray:
    """
    Identifies the Pareto front from a set of multi-objective predictions.

    A solution is on the Pareto front if it is not dominated by any other solution.
    A solution `A` dominates solution `B` if `A` is no worse than `B` in all
    objectives and strictly better in at least one objective.

    Args:
        predictions: A numpy array where rows are samples and columns are
                     objective values. Shape (n_samples, n_objectives).
        max_or_min: A list of strings, one for each objective, indicating
                    whether to 'max' (maximize) or 'min' (minimize) that objective.

    Returns:
        A numpy array containing the indices of the samples that are on
        the Pareto front.
    """
    n_samples = predictions.shape[0]
    n_objectives = predictions.shape[1]
    
    # Convert all objectives to maximization
    mod_predictions = predictions.copy()
    for i, direction in enumerate(max_or_min):
        if direction == "min":
            mod_predictions[:, i] = -mod_predictions[:, i]
    
    # Initialize domination count array
    is_dominated = np.zeros(n_samples, dtype=bool)
    
    # Compare each sample with all others
    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                # Check if j dominates i
                dominates = True
                
                for k in range(n_objectives):
                    # j doesn't dominate i if any objective is worse
                    if mod_predictions[j, k] < mod_predictions[i, k]:
                        dominates = False
                        break
                
                # If all objectives are at least as good and at least one is better
                if dominates and np.any(mod_predictions[j, :] > mod_predictions[i, :]):
                    is_dominated[i] = True
                    break
    
    # Return indices of non-dominated solutions
    return np.where(~is_dominated)[0]


def select_acquisition_function(curiosity: float, num_labeled_samples: int) -> str:
    """
    Dynamically selects an acquisition function based on the curiosity level
    and the number of available labeled samples.

    This implements a SLAMD-style strategy:
    - With few labeled samples (<10):
        - High curiosity (> 0.5) favors MaxEntropy (strong exploration).
        - Low curiosity (< -0.5) favors EI (strong exploitation).
        - Otherwise, UCB (balanced).
    - With more labeled samples (>=10):
        - PI is chosen if curiosity is high (> 1.0).
        - Otherwise, UCB.

    Args:
        curiosity: User-defined exploration-exploitation balance (-2 to +2).
        num_labeled_samples: Number of labeled samples available.

    Returns:
        The name of the selected acquisition function (e.g., "UCB", "EI", "PI", "MaxEntropy").
    """
    if num_labeled_samples < 10:
        if curiosity > 0.5:
            return "MaxEntropy"  # Strong exploration
        elif curiosity < -0.5:
            return "EI"          # Strong exploitation
        else:
            return "UCB"         # Balanced
    else:
        return "PI" if curiosity > 1.0 else "UCB"
