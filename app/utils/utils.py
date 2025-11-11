from scipy.spatial import distance_matrix
import numpy as np


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

from scipy.stats import norm
import numpy as np

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
    SLAMD-inspired utility computation combining exploration, exploitation, and novelty.

    Args:
        predictions: Predicted target values (n_samples, n_targets)
        uncertainties: Estimated uncertainties (n_samples, 1) or (n_samples,)
        novelty: Novelty scores (n_samples,)
        curiosity: Exploration factor (-2 to +2)
        weights: Relative importance of each target
        max_or_min: Optimization direction list for each target ('max' or 'min')
        thresholds: Optional thresholds for normalization or safety limits
        acquisition: Type of acquisition function ('UCB', 'EI', 'PI', 'MaxEntropy')
        for_visualization: Whether the utility is for visualization (scaling adjustments)

    Returns:
        Utility scores as (n_samples, 1)
    """
    # Ensure correct shapes
    predictions = np.atleast_2d(predictions)
    uncertainties = np.array(uncertainties).reshape(-1, 1)
    novelty = np.array(novelty).reshape(-1, 1)
    weights = np.array(weights).reshape(1, -1)

    # --- Normalize predictions per target to [0,1] ---
    preds_norm = np.zeros_like(predictions, dtype=float)
    for i in range(predictions.shape[1]):
        col = predictions[:, i]
        min_val, max_val = np.nanmin(col), np.nanmax(col)
        if max_val - min_val > 1e-9:
            preds_norm[:, i] = (col - min_val) / (max_val - min_val)
        else:
            preds_norm[:, i] = 0.5  # fallback if constant

        if max_or_min[i].lower() == "min":
            preds_norm[:, i] = 1.0 - preds_norm[:, i]

    # Weighted performance score
    weighted_perf = np.dot(preds_norm, weights.T).flatten()

    # --- Acquisition-based adjustment ---
    if acquisition == "UCB":
        acquisition_term = weighted_perf + curiosity * np.squeeze(uncertainties)
    elif acquisition == "EI":
        acquisition_term = weighted_perf + np.log1p(curiosity * np.squeeze(uncertainties))
    elif acquisition == "PI":
        acquisition_term = weighted_perf - np.log1p(np.squeeze(uncertainties))
    elif acquisition == "MaxEntropy":
        acquisition_term = weighted_perf + np.sqrt(np.abs(curiosity)) * novelty.flatten()
    else:
        acquisition_term = weighted_perf

    # --- Final utility calculation ---
    utility = (
        (0.6 * weighted_perf)
        + (0.3 * acquisition_term)
        + (0.1 * novelty.flatten())
    )

    # Scale between 0 and 1
    utility = np.clip((utility - np.min(utility)) / (np.ptp(utility) + 1e-12), 0, 1)

    # Optional visualization scaling
    if for_visualization:
        utility = np.log1p(utility * 10)
        utility = utility / np.max(utility)

    return utility.reshape(-1, 1)
