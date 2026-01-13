"""
Experiment Tracking Module

Simple experiment logger that saves experiment configurations, results,
and metrics to JSON files. This provides basic MLOps capabilities without
requiring external dependencies like MLflow.

For production use, consider integrating with MLflow or Weights & Biases.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        return super().default(obj)


class ExperimentLogger:
    """
    Simple experiment tracking that saves to JSON files.
    
    Usage:
        logger = ExperimentLogger()
        
        # Start an experiment
        exp_id = logger.start_experiment(
            name="PINN_test",
            config={'model': 'pinn', 'epochs': 100}
        )
        
        # Log metrics
        logger.log_metrics(exp_id, {'train_loss': 0.05, 'physics_loss': 0.02})
        
        # End experiment with results
        logger.end_experiment(exp_id, results_df)
    """
    
    def __init__(self, log_dir: str = "experiments"):
        """
        Initialize the experiment logger.
        
        Args:
            log_dir: Directory to store experiment logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.active_experiments: Dict[str, dict] = {}
    
    def start_experiment(self, 
                         name: str,
                         config: Dict[str, Any],
                         tags: Optional[List[str]] = None) -> str:
        """
        Start a new experiment run.
        
        Args:
            name: Human-readable experiment name
            config: Experiment configuration dict
            tags: Optional tags for filtering
        
        Returns:
            Unique experiment ID
        """
        exp_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.active_experiments[exp_id] = {
            'id': exp_id,
            'name': name,
            'config': config,
            'tags': tags or [],
            'metrics': {},
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'status': 'running'
        }
        
        logger.info(f"Started experiment: {exp_id}")
        return exp_id
    
    def log_metrics(self, exp_id: str, metrics: Dict[str, float]):
        """
        Log metrics to an active experiment.
        
        Args:
            exp_id: Experiment ID
            metrics: Dict of metric name -> value
        """
        if exp_id not in self.active_experiments:
            logger.warning(f"Experiment {exp_id} not found")
            return
        
        self.active_experiments[exp_id]['metrics'].update(metrics)
    
    def log_artifact(self, exp_id: str, name: str, data: Any):
        """
        Log an artifact (e.g., DataFrame, model config).
        
        Args:
            exp_id: Experiment ID
            name: Artifact name
            data: Artifact data (must be JSON-serializable)
        """
        if exp_id not in self.active_experiments:
            logger.warning(f"Experiment {exp_id} not found")
            return
        
        if 'artifacts' not in self.active_experiments[exp_id]:
            self.active_experiments[exp_id]['artifacts'] = {}
        
        self.active_experiments[exp_id]['artifacts'][name] = data
    
    def end_experiment(self, 
                       exp_id: str, 
                       results: Optional[pd.DataFrame] = None,
                       status: str = 'completed'):
        """
        End an experiment and save to disk.
        
        Args:
            exp_id: Experiment ID
            results: Optional results DataFrame
            status: Final status ('completed', 'failed', 'cancelled')
        """
        if exp_id not in self.active_experiments:
            logger.warning(f"Experiment {exp_id} not found")
            return
        
        exp = self.active_experiments[exp_id]
        exp['end_time'] = datetime.now().isoformat()
        exp['status'] = status
        
        if results is not None:
            # Save summary statistics
            exp['results_summary'] = {
                'num_candidates': len(results),
                'utility_mean': float(results['Utility'].mean()) if 'Utility' in results else None,
                'utility_max': float(results['Utility'].max()) if 'Utility' in results else None,
                'selected_sample': results[results.get('Selected for Testing', False) == True].to_dict('records') if 'Selected for Testing' in results else []
            }
        
        # Save to file
        exp_file = self.log_dir / f"{exp_id}.json"
        with open(exp_file, 'w') as f:
            json.dump(exp, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Saved experiment: {exp_file}")
        del self.active_experiments[exp_id]
        
        return exp_file
    
    def list_experiments(self, 
                         name_filter: Optional[str] = None,
                         tag_filter: Optional[str] = None) -> List[Dict]:
        """
        List saved experiments.
        
        Args:
            name_filter: Filter by name substring
            tag_filter: Filter by tag
        
        Returns:
            List of experiment summaries
        """
        experiments = []
        
        for exp_file in self.log_dir.glob("*.json"):
            try:
                with open(exp_file) as f:
                    exp = json.load(f)
                
                # Apply filters
                if name_filter and name_filter not in exp.get('name', ''):
                    continue
                if tag_filter and tag_filter not in exp.get('tags', []):
                    continue
                
                experiments.append({
                    'id': exp.get('id'),
                    'name': exp.get('name'),
                    'status': exp.get('status'),
                    'start_time': exp.get('start_time'),
                    'metrics': exp.get('metrics', {})
                })
            except Exception as e:
                logger.warning(f"Error reading {exp_file}: {e}")
        
        return sorted(experiments, key=lambda x: x.get('start_time', ''), reverse=True)
    
    def get_experiment(self, exp_id: str) -> Optional[Dict]:
        """
        Load a specific experiment by ID.
        
        Args:
            exp_id: Experiment ID
        
        Returns:
            Experiment data dict or None
        """
        exp_file = self.log_dir / f"{exp_id}.json"
        if not exp_file.exists():
            return None
        
        with open(exp_file) as f:
            return json.load(f)
    
    def delete_experiment(self, exp_id: str) -> bool:
        """
        Delete a specific experiment by ID.
        
        Args:
            exp_id: Experiment ID
        
        Returns:
            True if deleted, False if not found
        """
        exp_file = self.log_dir / f"{exp_id}.json"
        if not exp_file.exists():
            logger.warning(f"Experiment file not found: {exp_id}")
            return False
        
        try:
            exp_file.unlink()
            logger.info(f"Deleted experiment: {exp_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete experiment {exp_id}: {e}")
            return False
    
    def compare_experiments(self, exp_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple experiments.
        
        Args:
            exp_ids: List of experiment IDs to compare
        
        Returns:
            DataFrame with experiments as rows, metrics as columns
        """
        rows = []
        for exp_id in exp_ids:
            exp = self.get_experiment(exp_id)
            if exp:
                row = {
                    'id': exp_id,
                    'name': exp.get('name'),
                    'status': exp.get('status'),
                    **exp.get('config', {}),
                    **exp.get('metrics', {})
                }
                rows.append(row)
        
        return pd.DataFrame(rows)


# Global logger instance
_experiment_logger: Optional[ExperimentLogger] = None


def get_experiment_logger(log_dir: str = "experiments") -> ExperimentLogger:
    """Get the global experiment logger instance."""
    global _experiment_logger
    if _experiment_logger is None:
        _experiment_logger = ExperimentLogger(log_dir)
    return _experiment_logger
