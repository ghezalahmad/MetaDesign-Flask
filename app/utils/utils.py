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
