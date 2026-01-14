"""
Batch Selector Utility for Multi-Sample Selection

Implements diverse batch selection to avoid redundant sample picks.
Uses a greedy algorithm that balances utility with diversity.
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.preprocessing import StandardScaler


def select_batch(
    candidate_df: pd.DataFrame,
    n_samples: int = 1,
    input_columns: Optional[List[str]] = None,
    diversity_weight: float = 0.3
) -> pd.DataFrame:
    """
    Select top N samples with diversity consideration.
    
    Uses a greedy algorithm:
    1. Select highest utility sample
    2. For subsequent samples, balance utility vs. distance from already-selected
    
    Args:
        candidate_df: DataFrame with candidates, must have 'Utility' column
        n_samples: Number of samples to select
        input_columns: Feature columns for computing diversity (optional)
        diversity_weight: Weight for diversity term (0 = pure utility, 1 = pure diversity)
    
    Returns:
        Modified DataFrame with 'Selected for Testing' column updated
    """
    if candidate_df.empty:
        return candidate_df
    
    # Ensure we don't try to select more samples than available
    n_samples = min(n_samples, len(candidate_df))
    
    # If only selecting 1 sample, use simple max utility
    if n_samples == 1:
        candidate_df["Selected for Testing"] = False
        if "Utility" in candidate_df.columns and not candidate_df["Utility"].empty:
            max_utility_idx = candidate_df["Utility"].idxmax()
            candidate_df.loc[max_utility_idx, "Selected for Testing"] = True
        return candidate_df
    
    # Initialize selection column
    candidate_df = candidate_df.copy()
    candidate_df["Selected for Testing"] = False
    
    # Get utility values (normalized to 0-1)
    utility = candidate_df["Utility"].values.copy()
    utility_normalized = (utility - utility.min()) / (utility.max() - utility.min() + 1e-9)
    
    # Get feature matrix for diversity computation
    if input_columns and all(col in candidate_df.columns for col in input_columns):
        X = candidate_df[input_columns].values
        # Standardize for distance computation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        # Fallback: use all numeric columns except utility-related ones
        numeric_cols = candidate_df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['Utility', 'Uncertainty', 'Novelty', 'Row number', 'Semantic_Score', 'ML_Utility']
        feature_cols = [c for c in numeric_cols if c not in exclude_cols and not c.startswith('Uncertainty')]
        if feature_cols:
            X = candidate_df[feature_cols].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            # No features available, use pure utility ranking
            top_indices = candidate_df.nlargest(n_samples, 'Utility').index
            candidate_df.loc[top_indices, "Selected for Testing"] = True
            return candidate_df
    
    # Greedy diverse batch selection
    selected_indices = []
    selected_mask = np.zeros(len(candidate_df), dtype=bool)
    
    for i in range(n_samples):
        if i == 0:
            # First sample: pure highest utility
            scores = utility_normalized
        else:
            # Subsequent samples: utility + diversity bonus
            # Compute minimum distance to already-selected samples
            selected_features = X_scaled[selected_mask]
            distances = np.zeros(len(candidate_df))
            
            for j in range(len(candidate_df)):
                if selected_mask[j]:
                    distances[j] = -np.inf  # Already selected
                else:
                    # Minimum distance to any selected sample
                    dists = np.linalg.norm(X_scaled[j] - selected_features, axis=1)
                    distances[j] = dists.min()
            
            # Normalize distances to 0-1
            valid_distances = distances[distances > -np.inf]
            if len(valid_distances) > 0 and valid_distances.max() > valid_distances.min():
                dist_normalized = (distances - valid_distances.min()) / (valid_distances.max() - valid_distances.min() + 1e-9)
                dist_normalized[distances == -np.inf] = -np.inf
            else:
                dist_normalized = np.zeros_like(distances)
                dist_normalized[selected_mask] = -np.inf
            
            # Combined score: (1-λ)*utility + λ*diversity
            scores = (1 - diversity_weight) * utility_normalized + diversity_weight * dist_normalized
            scores[selected_mask] = -np.inf  # Exclude already selected
        
        # Select the sample with highest combined score
        best_idx = np.argmax(scores)
        original_idx = candidate_df.index[best_idx]
        selected_indices.append(original_idx)
        selected_mask[best_idx] = True
    
    # Mark selected samples
    candidate_df.loc[selected_indices, "Selected for Testing"] = True
    
    return candidate_df
