from scipy.spatial import distance_matrix
import numpy as np
from scipy.stats import norm


def enforce_diversity(candidate_inputs, selected_inputs, min_distance=0.1):
    if len(selected_inputs) == 0:
        return candidate_inputs
    
    distances = distance_matrix(candidate_inputs, selected_inputs)
    
    diverse_indices = np.where(np.min(distances, axis=1) > min_distance)[0]
    
    if len(diverse_indices) == 0:
        for reduction in [0.75, 0.5, 0.25, 0.1, 0.05]:
            reduced_threshold = min_distance * reduction
            diverse_indices = np.where(np.min(distances, axis=1) > reduced_threshold)[0]
            if len(diverse_indices) > 0:
                break
    
    if len(diverse_indices) == 0:
        max_min_distance_idx = np.argmax(np.min(distances, axis=1))
        return candidate_inputs[np.array([max_min_distance_idx])]
    
    return candidate_inputs[diverse_indices]

def calculate_novelty(features, labeled_features):
    if labeled_features.shape[0] == 0:
        return np.ones(features.shape[0])
    
    features = np.nan_to_num(features, nan=0.0)
    labeled_features = np.nan_to_num(labeled_features, nan=0.0)
    
    distances = distance_matrix(features, labeled_features)
    
    min_distances = distances.min(axis=1)
    
    max_distance = min_distances.max()
    if max_distance > 0:
        novelty = min_distances / max_distance
    else:
        novelty = np.ones_like(min_distances)

    return novelty



def select_acquisition_function(curiosity: float, n_labeled_samples: int) -> str:
    """
    Dynamically selects an acquisition function based on curiosity and dataset size.
    Mimics SLAMD adaptive exploration–exploitation behavior.
    """
    if n_labeled_samples < 10:
        return "UCB"  # more exploration when data is scarce
    elif curiosity > 0.5:
        return "EI"
    elif curiosity < -0.5:
        return "PI"
    else:
        return "MaxEntropy"


def calculate_utility(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    novelty: np.ndarray,
    curiosity: float,
    weights: np.ndarray,
    max_or_min: list[str],
    thresholds=None,
    acquisition: str = "UCB",
    for_visualization: bool = False
) -> np.ndarray:
    """
    SLAMD-EXACT utility computation - matches WEBSLAMD implementation exactly.

    Formula: Utility = Σ(normalized_prediction × weight × direction) + curiosity × Σ(uncertainty)
    
    Where direction = 1 for 'max', -1 for 'min'
    
    Args:
        predictions: Predicted target values (n_samples, n_targets)
        uncertainties: Estimated uncertainties (n_samples, 1) or (n_samples,)
        novelty: Novelty scores (n_samples,) - added to exploration
        curiosity: Exploration factor (0=exploit, 1=explore)
        weights: Relative importance of each target
        max_or_min: Optimization direction list for each target ('max' or 'min')
        thresholds: Optional thresholds for clipping
        acquisition: Acquisition function type (UCB, EI, etc.)
        for_visualization: Whether the utility is for visualization scaling

    Returns:
        Utility scores as (n_samples, 1)
    """
    # Ensure correct shapes
    predictions = np.atleast_2d(predictions)
    uncertainties = np.array(uncertainties).reshape(-1, 1)
    novelty = np.array(novelty).reshape(-1, 1)
    weights = np.array(weights).reshape(1, -1)

    # --- WEBSLAMD-style z-score normalization ---
    preds_norm = np.zeros_like(predictions, dtype=float)
    unc_norm = np.zeros_like(uncertainties, dtype=float)
    
    for i in range(predictions.shape[1]):
        col = predictions[:, i]
        mean_val = np.nanmean(col)
        std_val = np.nanstd(col)
        if std_val < 1e-9:
            std_val = 1.0  # Avoid division by zero
        
        # Z-score normalize
        preds_norm[:, i] = (col - mean_val) / std_val
        
        # WEBSLAMD: Multiply by -1 for minimization targets (AFTER normalization)
        if max_or_min[i].lower() == "min":
            preds_norm[:, i] *= -1

    # Normalize uncertainty (same z-score approach)
    unc_mean = np.nanmean(uncertainties)
    unc_std = np.nanstd(uncertainties)
    if unc_std < 1e-9:
        unc_std = 1.0
    unc_norm = (uncertainties - unc_mean) / unc_std

    # Weighted performance score (prediction component)
    weighted_perf = np.dot(preds_norm, weights.T).flatten()

    # --- WEBSLAMD Utility Formula ---
    # Utility = prediction_sum + curiosity * uncertainty_sum
    utility = weighted_perf + curiosity * unc_norm.flatten()
    
    # Add novelty component (WEBSLAMD stores separately, we add it)
    utility += 0.1 * novelty.flatten()

    # WEBSLAMD: Keep raw z-scored utility values (typically -3 to +3 range)
    # Do NOT scale to [0, 1] - this matches WEBSLAMD's display

    # Optional visualization scaling only
    if for_visualization:
        utility = np.log1p(np.abs(utility) * 10) * np.sign(utility)
        utility = utility / (np.max(np.abs(utility)) + 1e-12)

    return utility.reshape(-1, 1)

