"""
MLflow Adapter for Experiment Tracking

This module provides optional MLflow integration for experiment tracking.
When MLflow is available and enabled, experiments are logged to MLflow
in addition to the local JSON files.

Features:
- Automatic experiment creation
- Parameter logging
- Metrics logging
- Optional model artifact logging
- Fallback to local logging if MLflow unavailable

Usage:
    from app.tracking.mlflow_adapter import get_mlflow_tracker
    
    tracker = get_mlflow_tracker()
    if tracker.is_enabled():
        with tracker.start_run(run_name="pinn_experiment") as run:
            tracker.log_params(config)
            tracker.log_metrics(metrics)
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Check if MLflow is available
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None


class MLflowTracker:
    """
    MLflow integration for experiment tracking.
    
    This class wraps MLflow functionality and provides a clean interface
    for logging experiments. If MLflow is not installed or disabled,
    all methods gracefully return without errors.
    """
    
    def __init__(self, 
                 experiment_name: str = "MetaDesign",
                 tracking_uri: Optional[str] = None,
                 enabled: bool = True):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: Optional MLflow tracking server URI
            enabled: Whether MLflow tracking is enabled
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._enabled = enabled and MLFLOW_AVAILABLE
        self._current_run = None
        
        if self._enabled:
            try:
                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(experiment_name)
                logger.info(f"MLflow initialized: experiment='{experiment_name}'")
            except Exception as e:
                logger.warning(f"MLflow initialization failed: {e}")
                self._enabled = False
    
    def is_enabled(self) -> bool:
        """Check if MLflow tracking is enabled and available."""
        return self._enabled
    
    def is_available(self) -> bool:
        """Check if MLflow library is installed."""
        return MLFLOW_AVAILABLE
    
    def start_run(self, run_name: Optional[str] = None, 
                  nested: bool = False) -> 'MLflowRunContext':
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            nested: Whether this is a nested run
            
        Returns:
            Context manager for the run
        """
        return MLflowRunContext(self, run_name, nested)
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to the current run.
        
        Args:
            params: Dictionary of parameters to log
        """
        if not self._enabled or not self._current_run:
            return
        
        try:
            # MLflow has limits on param value length, truncate if needed
            clean_params = {}
            for k, v in params.items():
                str_val = str(v)
                if len(str_val) > 250:
                    str_val = str_val[:247] + "..."
                clean_params[k] = str_val
            
            mlflow.log_params(clean_params)
        except Exception as e:
            logger.warning(f"MLflow log_params failed: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to the current run.
        
        Args:
            metrics: Dictionary of metrics (must be numeric)
            step: Optional step number
        """
        if not self._enabled or not self._current_run:
            return
        
        try:
            # Filter to only numeric values
            numeric_metrics = {
                k: float(v) for k, v in metrics.items()
                if isinstance(v, (int, float)) and v == v  # Filter out NaN
            }
            mlflow.log_metrics(numeric_metrics, step=step)
        except Exception as e:
            logger.warning(f"MLflow log_metrics failed: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact file.
        
        Args:
            local_path: Path to the local file
            artifact_path: Optional path within artifacts directory
        """
        if not self._enabled or not self._current_run:
            return
        
        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            logger.warning(f"MLflow log_artifact failed: {e}")
    
    def log_model(self, model, model_name: str, **kwargs):
        """
        Log a model artifact.
        
        Args:
            model: The model object to log
            model_name: Name for the model
            **kwargs: Additional arguments for the model logger
        """
        if not self._enabled or not self._current_run:
            return
        
        try:
            # Try to log as sklearn model first, then as pytorch
            try:
                mlflow.sklearn.log_model(model, model_name, **kwargs)
            except Exception:
                try:
                    mlflow.pytorch.log_model(model, model_name, **kwargs)
                except Exception:
                    # Just log as a generic artifact
                    pass
        except Exception as e:
            logger.warning(f"MLflow log_model failed: {e}")
    
    def set_tag(self, key: str, value: str):
        """Set a tag on the current run."""
        if not self._enabled or not self._current_run:
            return
        
        try:
            mlflow.set_tag(key, value)
        except Exception as e:
            logger.warning(f"MLflow set_tag failed: {e}")
    
    def get_run_id(self) -> Optional[str]:
        """Get the current run ID."""
        if self._current_run:
            return self._current_run.info.run_id
        return None
    
    def end_run(self, status: str = "FINISHED"):
        """
        End the current run.
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        if not self._enabled or not self._current_run:
            return
        
        try:
            mlflow.end_run(status)
            self._current_run = None
        except Exception as e:
            logger.warning(f"MLflow end_run failed: {e}")


class MLflowRunContext:
    """Context manager for MLflow runs."""
    
    def __init__(self, tracker: MLflowTracker, run_name: Optional[str], nested: bool):
        self.tracker = tracker
        self.run_name = run_name
        self.nested = nested
        self.run = None
    
    def __enter__(self):
        if self.tracker.is_enabled():
            try:
                self.run = mlflow.start_run(run_name=self.run_name, nested=self.nested)
                self.tracker._current_run = self.run
            except Exception as e:
                logger.warning(f"MLflow start_run failed: {e}")
        return self.tracker
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tracker.is_enabled() and self.run:
            status = "FINISHED" if exc_type is None else "FAILED"
            self.tracker.end_run(status)
        return False


# Global tracker instance
_mlflow_tracker: Optional[MLflowTracker] = None


def get_mlflow_tracker(experiment_name: str = "MetaDesign",
                       tracking_uri: Optional[str] = None,
                       enabled: bool = True) -> MLflowTracker:
    """
    Get the global MLflow tracker instance.
    
    Args:
        experiment_name: Name of the experiment
        tracking_uri: Optional tracking server URI
        enabled: Whether to enable MLflow
        
    Returns:
        MLflowTracker instance
    """
    global _mlflow_tracker
    
    if _mlflow_tracker is None:
        _mlflow_tracker = MLflowTracker(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            enabled=enabled
        )
    
    return _mlflow_tracker


def mlflow_available() -> bool:
    """Check if MLflow is installed."""
    return MLFLOW_AVAILABLE
