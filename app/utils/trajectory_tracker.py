"""
Trajectory Tracker for Active Learning experiments.

Tracks the exploration path through the feature space across iterations,
implementing the trajectory analysis from the LLM-AL paper.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from flask import has_request_context
from app.utils.session_store import get_session_id, get_session_trajectory_path


class TrajectoryTracker:
    """
    Tracks selected points across iterations to visualize exploration trajectory.
    
    Implements trajectory analysis from:
    "Training-Free Active Learning Framework in Materials Science with Large Language Models"
    """
    
    # Class-level storage, isolated by browser session.
    _trajectory_history_by_key = {}
    _feature_columns_by_key = {}
    _loaded_keys = set()
    _save_path = Path("data/trajectory_history.json")

    @classmethod
    def _state_key(cls):
        return get_session_id() if has_request_context() else "global"

    @classmethod
    def _path_for_key(cls, key):
        if has_request_context() and key != "global":
            return get_session_trajectory_path()
        return cls._save_path

    @classmethod
    def _ensure_loaded(cls, key=None):
        key = key or cls._state_key()
        if key not in cls._loaded_keys:
            cls.load_from_file(key=key)
        cls._trajectory_history_by_key.setdefault(key, [])
        cls._feature_columns_by_key.setdefault(key, [])
        return key

    @classmethod
    def _history(cls, key=None):
        key = cls._ensure_loaded(key)
        return cls._trajectory_history_by_key[key]

    @classmethod
    def _feature_columns(cls, key=None):
        key = cls._ensure_loaded(key)
        return cls._feature_columns_by_key[key]
    
    @classmethod
    def add_point(cls, iteration: int, selected_row: pd.Series, 
                  feature_columns: list, mode: str):
        """
        Add a selected point to the trajectory.
        
        Args:
            iteration: Current iteration number
            selected_row: The selected experiment row (pandas Series)
            feature_columns: List of feature column names for distance calculation
            mode: Active learning mode (ML_MODE, LLM_AGENT_MODE, HYBRID_MODE)
        """
        key = cls._ensure_loaded()
        cls._feature_columns_by_key[key] = feature_columns
        
        # Extract feature values
        feature_values = {}
        for col in feature_columns:
            if col in selected_row.index:
                val = selected_row[col]
                feature_values[col] = float(val) if pd.notna(val) else None
        
        point = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "row_index": int(selected_row.name) if hasattr(selected_row, 'name') else None,
            "features": feature_values,
            "utility": float(selected_row.get('Utility', 0)) if 'Utility' in selected_row else 0
        }
        
        cls._trajectory_history_by_key[key].append(point)
        logging.info(f"📍 Trajectory: Added point {iteration} ({mode})")
        
        # Auto-save for persistence
        cls.save_to_file(key=key)
        
        return point
    
    @classmethod
    def get_trajectory(cls) -> list:
        """Get the full trajectory history."""
        return cls._history().copy()
    
    @classmethod
    def get_iteration_count(cls) -> int:
        """Get the current iteration count."""
        return len(cls._history())
    
    @classmethod
    def calculate_cumulative_distance(cls) -> list:
        """
        Calculate cumulative Euclidean distance in standardized feature space.
        
        Returns:
            List of cumulative distances for each iteration
        """
        history = cls._history()
        feature_columns = cls._feature_columns()
        if len(history) < 2:
            return [0.0] * len(history)
        
        # Extract feature vectors
        feature_vectors = []
        for point in history:
            vec = []
            for col in feature_columns:
                val = point['features'].get(col)
                vec.append(val if val is not None else 0.0)
            feature_vectors.append(np.array(vec))
        
        if not feature_vectors:
            return [0.0] * len(history)
        
        # Standardize features (like the paper)
        all_features = np.array(feature_vectors)
        mean = np.mean(all_features, axis=0)
        std = np.std(all_features, axis=0) + 1e-9  # Avoid division by zero
        standardized = (all_features - mean) / std
        
        # Calculate cumulative distance
        cumulative_distances = [0.0]
        total_distance = 0.0
        
        for i in range(1, len(standardized)):
            distance = np.linalg.norm(standardized[i] - standardized[i-1])
            total_distance += distance
            cumulative_distances.append(total_distance)
        
        return cumulative_distances
    
    @classmethod
    def get_trajectory_summary(cls) -> dict:
        """Get summary statistics for the trajectory."""
        distances = cls.calculate_cumulative_distance()
        
        history = cls._history()
        return {
            "total_iterations": len(history),
            "total_distance": distances[-1] if distances else 0.0,
            "cumulative_distances": distances,
            "modes_used": list(set(p['mode'] for p in history)),
            "trajectory": history
        }
    
    @classmethod
    def clear(cls):
        """Clear the trajectory history."""
        key = cls._ensure_loaded()
        cls._trajectory_history_by_key[key] = []
        cls._feature_columns_by_key[key] = []
        logging.info("🗑️ Trajectory: Cleared history")
        
        # Remove saved file
        save_path = cls._path_for_key(key)
        if save_path.exists():
            save_path.unlink()
    
    @classmethod
    def save_to_file(cls, key=None):
        """Save trajectory to JSON file."""
        try:
            key = cls._ensure_loaded(key)
            save_path = cls._path_for_key(key)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "feature_columns": cls._feature_columns_by_key.get(key, []),
                "trajectory": cls._trajectory_history_by_key.get(key, [])
            }
            
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logging.warning(f"⚠️ Could not save trajectory: {e}")
    
    @classmethod
    def load_from_file(cls, key=None):
        """Load trajectory from JSON file."""
        key = key or cls._state_key()
        save_path = cls._path_for_key(key)
        try:
            if save_path.exists():
                with open(save_path, 'r') as f:
                    data = json.load(f)
                
                cls._feature_columns_by_key[key] = data.get('feature_columns', [])
                cls._trajectory_history_by_key[key] = data.get('trajectory', [])
                logging.info(f"📂 Trajectory: Loaded {len(cls._trajectory_history_by_key[key])} points")
            else:
                cls._feature_columns_by_key[key] = []
                cls._trajectory_history_by_key[key] = []
            cls._loaded_keys.add(key)
                
        except Exception as e:
            logging.warning(f"⚠️ Could not load trajectory: {e}")
            cls._trajectory_history_by_key[key] = []
            cls._feature_columns_by_key[key] = []
            cls._loaded_keys.add(key)


# Load existing trajectory on module import for persistence
TrajectoryTracker.load_from_file()
