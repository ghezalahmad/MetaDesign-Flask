"""
Model Registry for MetaDesign

Centralized registry for all surrogate model implementations.
Enables dynamic model selection, consistent configuration, and
easy addition of new models.
"""

from typing import Dict, Any, Optional, Type, Callable
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Central registry for surrogate model configurations.
    
    Usage:
        # Register a model
        ModelRegistry.register(
            name='pinn',
            model_class=PINNModel,
            train_func=pinn_train,
            evaluate_func=evaluate_pinn,
            default_params={'epochs': 100, 'learning_rate': 0.001}
        )
        
        # Get model config
        config = ModelRegistry.get('pinn')
        model = config['class'](input_size, output_size)
        
        # List available models
        models = ModelRegistry.list_models()
    """
    
    _registry: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls,
                 name: str,
                 model_class: Type,
                 train_func: Optional[Callable] = None,
                 evaluate_func: Optional[Callable] = None,
                 default_params: Optional[Dict[str, Any]] = None,
                 description: str = "",
                 category: str = "general"):
        """
        Register a new model type.
        
        Args:
            name: Unique identifier for the model (e.g., 'pinn', 'maml')
            model_class: The model class to instantiate
            train_func: Optional training function (if not using model.train())
            evaluate_func: Evaluation function for utility scoring
            default_params: Default hyperparameters
            description: Human-readable description
            category: Model category ('meta-learning', 'bayesian', 'ensemble', etc.)
        """
        cls._registry[name.lower()] = {
            'class': model_class,
            'train': train_func,
            'evaluate': evaluate_func,
            'params': default_params or {},
            'description': description,
            'category': category
        }
        logger.info(f"Registered model: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[Dict[str, Any]]:
        """Get model configuration by name."""
        return cls._registry.get(name.lower())
    
    @classmethod
    def list_models(cls) -> list:
        """List all registered model names."""
        return list(cls._registry.keys())
    
    @classmethod
    def list_by_category(cls, category: str) -> list:
        """List models in a specific category."""
        return [
            name for name, config in cls._registry.items()
            if config.get('category') == category
        ]
    
    @classmethod
    def get_model_info(cls, name: str) -> dict:
        """Get human-readable model information."""
        config = cls.get(name)
        if config is None:
            return {}
        return {
            'name': name,
            'description': config.get('description', ''),
            'category': config.get('category', 'general'),
            'default_params': config.get('params', {})
        }
    
    @classmethod
    def instantiate(cls, name: str, *args, **kwargs):
        """
        Create a new instance of a registered model.
        
        Args:
            name: Model name
            *args, **kwargs: Arguments passed to model constructor
        
        Returns:
            Model instance
        """
        config = cls.get(name)
        if config is None:
            raise ValueError(f"Model '{name}' not registered. Available: {cls.list_models()}")
        
        model_class = config['class']
        return model_class(*args, **kwargs)
    
    @classmethod
    def clear(cls):
        """Clear all registrations (mainly for testing)."""
        cls._registry.clear()


# =====================================================================
# Register all available models
# =====================================================================

def _register_all_models():
    """
    Register all model implementations.
    
    Called at module import to populate the registry.
    """
    # Import model implementations
    from app.models.pinn_model import PINNModel, pinn_train, evaluate_pinn
    from app.models.models import MAMLModel, evaluate_maml, MAMLMultiTargetWrapper
    from app.models.reptile_model import ReptileModel, reptile_train, evaluate_reptile
    from app.models.gp_model import GPModel, train_gp_model as gp_train, evaluate_gp_model as evaluate_gp
    from app.models.rf_model import RFModel, train_rf_model as rf_train, evaluate_rf_model as evaluate_rf
    from app.models.dkl_surrogate_model import DKLModel, train_dkl_model, evaluate_dkl_model
    from app.models.protonet_model import ProtoNetModel as ProtoNet, protonet_train, evaluate_protonet
    from app.models.lolopy_model import LolopyRFModel, train_lolopy_model as lolopy_train, evaluate_lolopy_model
    from app.models.rl_model import RLModel, train_rl_model, evaluate_rl_model
    from app.models.ensemble import EnsembleSurrogate, weighted_uncertainty_ensemble
    
    # Register PINN
    ModelRegistry.register(
        name='pinn',
        model_class=PINNModel,
        train_func=pinn_train,
        evaluate_func=evaluate_pinn,
        default_params={'epochs': 100, 'learning_rate': 0.001, 'physics_loss_weight': 0.1},
        description='Physics-Informed Neural Network with domain constraints',
        category='physics-informed'
    )
    
    # Register MAML
    ModelRegistry.register(
        name='maml',
        model_class=MAMLMultiTargetWrapper,
        train_func=None,  # Uses internal meta_train
        evaluate_func=evaluate_maml,
        default_params={'epochs': 100, 'inner_lr': 0.01, 'outer_lr': 0.001},
        description='Model-Agnostic Meta-Learning with per-target training',
        category='meta-learning'
    )
    
    # Register Reptile
    ModelRegistry.register(
        name='reptile',
        model_class=ReptileModel,
        train_func=reptile_train,
        evaluate_func=evaluate_reptile,
        default_params={'epochs': 100, 'learning_rate': 0.001},
        description='Reptile meta-learning with per-target training',
        category='meta-learning'
    )
    
    # Register GP
    ModelRegistry.register(
        name='gp',
        model_class=GPModel,
        train_func=gp_train,
        evaluate_func=evaluate_gp,
        default_params={},
        description='Gaussian Process with RBF kernel',
        category='bayesian'
    )
    
    # Register RF
    ModelRegistry.register(
        name='rf',
        model_class=RFModel,
        train_func=rf_train,
        evaluate_func=evaluate_rf,
        default_params={'n_estimators': 100},
        description='Random Forest with uncertainty from tree variance',
        category='ensemble-tree'
    )
    
    # Register DKL
    ModelRegistry.register(
        name='dkl',
        model_class=DKLModel,
        train_func=train_dkl_model,
        evaluate_func=evaluate_dkl_model,
        default_params={'epochs': 50},
        description='Deep Kernel Learning (neural network + GP)',
        category='bayesian'
    )
    
    # Register ProtoNet
    ModelRegistry.register(
        name='protonet',
        model_class=ProtoNet,
        train_func=protonet_train,
        evaluate_func=evaluate_protonet,
        default_params={'epochs': 100, 'hidden_size': 64},
        description='Prototypical Networks for few-shot learning',
        category='meta-learning'
    )
    
    # Register Lolopy
    ModelRegistry.register(
        name='lolopy',
        model_class=LolopyRFModel,
        train_func=lolopy_train,
        evaluate_func=evaluate_lolopy_model,
        default_params={'n_estimators': 100},
        description='Lolopy Random Forest with native uncertainty',
        category='ensemble-tree'
    )
    
    # Register RL (Reinforcement Learning)
    ModelRegistry.register(
        name='rl',
        model_class=RLModel,
        train_func=train_rl_model,
        evaluate_func=evaluate_rl_model,
        default_params={'hidden_dim': 128, 'lr': 1e-3},
        description='Reinforcement Learning (PPO) for adaptive sample selection',
        category='reinforcement-learning'
    )
    
    # Register Ensemble
    ModelRegistry.register(
        name='ensemble',
        model_class=EnsembleSurrogate,
        train_func=None,  # Ensemble trains multiple models
        evaluate_func=weighted_uncertainty_ensemble,
        default_params={},
        description='Weighted ensemble of multiple models',
        category='ensemble'
    )


# Auto-register models when module is imported
# Wrapped in try-except to avoid import errors during development
try:
    _register_all_models()
except ImportError as e:
    logger.warning(f"Could not auto-register all models: {e}")
