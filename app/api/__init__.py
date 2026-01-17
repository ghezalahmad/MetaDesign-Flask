"""
API Package

This package contains all API route modules organized by functionality.
Each module defines its own blueprint which is registered with the main app.

Modules:
- settings: User settings management
- trajectory: Experiment trajectory tracking
- experiments: Experiment logging and history
- design_space: Design space creation and management
- uploads: File upload handling
- run_experiment: Main experiment execution endpoint
- results: Lab results and cycle management
- scenarios: Project-based scenario management
"""

from app.api.settings import settings_bp
from app.api.trajectory import trajectory_bp
from app.api.experiments import experiments_bp
from app.api.design_space import design_space_bp
from app.api.uploads import uploads_bp
from app.api.run_experiment import run_experiment_bp
from app.api.results import results_bp
from app.api.scenarios import scenarios_bp

# List of all blueprints to register
all_blueprints = [
    settings_bp,
    trajectory_bp,
    experiments_bp,
    design_space_bp,
    uploads_bp,
    run_experiment_bp,
    results_bp,
    scenarios_bp
]

__all__ = [
    'settings_bp',
    'trajectory_bp', 
    'experiments_bp',
    'design_space_bp',
    'uploads_bp',
    'run_experiment_bp',
    'results_bp',
    'scenarios_bp',
    'all_blueprints'
]
