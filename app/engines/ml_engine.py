"""
ML Engine - Refactored with Registry and Acquisition Functions

This module provides the core ML pipeline for active learning experiments.
Refactored to use:
- ModelRegistry for clean model selection
- AcquisitionFunction interface for pluggable utility calculation
- ExperimentLogger for tracking
"""

import pandas as pd
import numpy as np
import logging

from app.utils.experiment_preprocessor import ExperimentPreprocessor
from app.utils.experiment_data import ExperimentData
from app.utils.utils import calculate_novelty
from app.acquisition import get_acquisition_function, WEBSLAMD
from app.tracking import get_experiment_logger

# Import models (for fallback and ensemble)
from app.models.models import MAMLModel, evaluate_maml
from app.models.reptile_model import ReptileModel, evaluate_reptile, reptile_train
from app.models.gp_model import GPModel, train_gp_model, evaluate_gp_model
from app.models.protonet_model import ProtoNetModel, evaluate_protonet, protonet_train
from app.models.dkl_surrogate_model import DKLModel, train_dkl_model, evaluate_dkl_model
from app.models.rf_model import train_rf_model, evaluate_rf_model, RFModel
from app.models.pinn_model import PINNModel, pinn_train, evaluate_pinn
from app.models.lolopy_model import LolopyRFModel, train_lolopy_model, evaluate_lolopy_model
from app.models.rl_model import RLModel, train_rl_model, evaluate_rl_model
from app.models.ensemble import weighted_uncertainty_ensemble

logger = logging.getLogger(__name__)


# --- Model Configuration (will be replaced by registry) ---
MODEL_CONFIG = {
    'maml': {
        'model_class': MAMLModel,
        'evaluate_func': evaluate_maml,
    },
    'reptile': {
        'model_class': ReptileModel,
        'train_func': reptile_train,
        'evaluate_func': evaluate_reptile,
        'train_params': (50, 0.001, 5, 16),
    },
    'protonet': {
        'model_class': ProtoNetModel,
        'train_func': protonet_train,
        'evaluate_func': evaluate_protonet,
        'train_params': (50, 0.001, 5, 5, 5),
    },
    'rf': {
        'model_class': RFModel,
        'train_func': train_rf_model,
        'evaluate_func': evaluate_rf_model,
        'train_params': None,
    },
    'pinn': {
        'model_class': PINNModel,
        'train_func': pinn_train,
        'evaluate_func': evaluate_pinn,
        'train_params': (100, 0.001, 0.1, 32),
    },
    'gp': {
        'model_class': GPModel,
        'train_func': train_gp_model,
        'evaluate_func': evaluate_gp_model,
        'train_params': ({}),
    },
    'lolopy': {
        'model_class': LolopyRFModel,
        'train_func': train_lolopy_model,
        'evaluate_func': evaluate_lolopy_model,
        'train_params': None,
    },
    'dkl': {
        'model_class': DKLModel,
        'train_func': train_dkl_model,
        'evaluate_func': evaluate_dkl_model,
        'train_params': ({}),
    },
    'rl': {
        'model_class': RLModel,
        'train_func': train_rl_model,
        'evaluate_func': evaluate_rl_model,
        'train_params': ({}),
    },
}


class MLEngine:
    """
    Core ML engine for active learning experiments.
    
    Supports multiple surrogate models and acquisition functions.
    """
    
    @staticmethod
    def run_experiment(data, model_name, input_columns, target_columns_config, 
                       curiosity=0.5, apriori_config=None,
                       acquisition_function='webslamd', batch_size=1):
        """
        Executes the ML pipeline: Preprocessing -> Training -> Evaluation -> Utility Calculation
        
        Args:
            data: Full dataset (labeled + unlabeled)
            model_name: Name of surrogate model to use
            input_columns: List of feature column names
            target_columns_config: List of dicts with 'name', 'weight', 'optimization'
            curiosity: Exploration-exploitation trade-off (0-1)
            apriori_config: Optional a-priori property configuration
            acquisition_function: Name of acquisition function ('webslamd', 'ucb', 'ei', 'thompson')
        
        Returns:
            DataFrame with predictions, uncertainties, and utility scores
        """
        # --- Initialize experiment tracking ---
        exp_logger = get_experiment_logger()
        exp_id = exp_logger.start_experiment(
            name=f"{model_name}_experiment",
            config={
                'model': model_name,
                'acquisition': acquisition_function,
                'curiosity': curiosity,
                'num_features': len(input_columns),
                'num_targets': len(target_columns_config)
            }
        )
        
        try:
            result = MLEngine._run_pipeline(
                data, model_name, input_columns, target_columns_config,
                curiosity, apriori_config, acquisition_function, batch_size
            )
            
            # Log success metrics
            if not result.empty and 'Utility' in result.columns:
                metrics = {
                    'num_candidates': len(result),
                    'utility_max': float(result['Utility'].max()),
                    'utility_mean': float(result['Utility'].mean())
                }
                exp_logger.log_metrics(exp_id, metrics)
                
                # --- Optional MLflow logging ---
                try:
                    from app.utils.settings_manager import SettingsManager
                    from app.tracking import get_mlflow_tracker, mlflow_available
                    
                    if mlflow_available() and SettingsManager.get_setting('mlflow_enabled', False):
                        mlflow_tracker = get_mlflow_tracker(
                            experiment_name=SettingsManager.get_setting('mlflow_experiment_name', 'MetaDesign'),
                            tracking_uri=SettingsManager.get_setting('mlflow_tracking_uri', None),
                            enabled=True
                        )
                        
                        with mlflow_tracker.start_run(run_name=f"{model_name}_{exp_id}"):
                            mlflow_tracker.log_params({
                                'model': model_name,
                                'acquisition': acquisition_function,
                                'curiosity': curiosity,
                                'num_features': len(input_columns),
                                'num_targets': len(target_columns_config)
                            })
                            mlflow_tracker.log_metrics(metrics)
                            mlflow_tracker.set_tag('status', 'completed')
                            logger.info(f"✅ MLflow logged: {mlflow_tracker.get_run_id()}")
                except Exception as mlflow_err:
                    logger.warning(f"MLflow logging skipped: {mlflow_err}")
            
            exp_logger.end_experiment(exp_id, result, status='completed')
            return result
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            exp_logger.log_metrics(exp_id, {'error': str(e)})
            exp_logger.end_experiment(exp_id, status='failed')
            raise
    
    @staticmethod
    def _run_pipeline(data, model_name, input_columns, target_columns_config,
                      curiosity, apriori_config, acquisition_function, batch_size=1):
        """Internal pipeline execution."""
        
        # --- Create WEBSLAMD-style ExperimentData ---
        exp = ExperimentData(
            dataframe=data,
            model=model_name,
            curiosity=float(curiosity),
            feature_names=input_columns,
            target_names=[t['name'] for t in target_columns_config],
            target_weights=[float(t['weight']) for t in target_columns_config],
            target_thresholds=[t.get('threshold') for t in target_columns_config],
            target_max_or_min=[t['optimization'] for t in target_columns_config],
            apriori_names=[a['name'] for a in (apriori_config or [])],
            apriori_weights=[float(a.get('weight', 1.0)) for a in (apriori_config or [])],
            apriori_thresholds=[a.get('threshold') for a in (apriori_config or [])],
            apriori_max_or_min=[a.get('optimization', 'max') for a in (apriori_config or [])],
        )
        
        logger.info(f"📊 ExperimentData: {exp.model}, {len(exp.feature_names)} features, "
                   f"{len(exp.target_names)} targets, labelled={len(exp.index_all_labelled)}, "
                   f"to_predict={len(exp.index_predicted)}")
        
        # --- Preprocessing ---
        preprocess_config = {
            'model': model_name,
            'input_columns': input_columns,
            'target_columns': target_columns_config,
            'apriori_columns': apriori_config or [],
            'curiosity': curiosity
        }
        
        data, preprocess_result = ExperimentPreprocessor.preprocess(exp.dataframe, preprocess_config)
        exp.dataframe = data
        
        if not preprocess_result['valid']:
            logger.error(f"❌ Preprocessing failed: {preprocess_result['errors']}")
            error_df = pd.DataFrame()
            error_df.attrs['preprocessing_errors'] = preprocess_result['errors']
            return error_df
        
        for warning in preprocess_result.get('warnings', []):
            logger.warning(f"⚠️ Preprocessing: {warning}")
        
        if preprocess_result.get('dropped_columns'):
            exp.feature_names = [c for c in exp.feature_names if c not in preprocess_result['dropped_columns']]
        
        # Extract for compatibility
        target_columns = exp.target_names
        weights = np.array(exp.target_weights)
        max_or_min = exp.target_max_or_min
        input_columns = exp.feature_names
        
        results_df = pd.DataFrame()
        config_entry = MODEL_CONFIG.get(model_name)
        input_size = len(input_columns)
        output_size = len(target_columns)

        # --- Model Training and Evaluation ---
        if config_entry:
            if model_name in ['maml', 'reptile', 'protonet', 'pinn']:
                model = config_entry['model_class'](input_size=input_size, output_size=output_size)
            else:
                model = None

            if 'train_func' in config_entry:
                train_func = config_entry['train_func']
                train_params = config_entry.get('train_params')
                
                if model_name in ['dkl', 'gp', 'rl']:
                    model_params = train_params[0] if isinstance(train_params, tuple) else train_params or {}
                    model, _, _ = train_func(data, input_columns, target_columns, model_params)
                elif train_params is not None:
                    model, _, _ = train_func(model, data, input_columns, target_columns, *train_params)
                else:
                    model, _, _ = train_func(data, input_columns, target_columns)

            evaluate_func = config_entry['evaluate_func']
            
            if model_name in ['dkl', 'gp']:
                labeled_data = data.dropna(subset=target_columns)
                candidate_data = data[data[target_columns[0]].isnull()] if isinstance(target_columns, list) else data[data[target_columns].isnull()]
                candidate_inputs = candidate_data[input_columns]
                results_df = evaluate_func(model, labeled_data, candidate_inputs, input_columns, target_columns, weights, max_or_min, curiosity)
            elif model_name == 'rl':
                # RL uses same signature as lolopy
                results_df = evaluate_func(model, data, input_columns, target_columns, curiosity, weights, max_or_min)
            else:
                results_df = evaluate_func(model, data, input_columns, target_columns, curiosity, weights, max_or_min)

        elif model_name == 'ensemble':
            pinn_model = PINNModel(input_size=input_size, output_size=output_size)
            pinn_model, pinn_scaler_x, pinn_scaler_y = pinn_train(pinn_model, data, input_columns, target_columns, 100, 0.001, 0.1, 32)
            rf_model, rf_scaler_x, rf_scaler_y = train_rf_model(data, input_columns, target_columns)
            models = {'pinn': (pinn_model, pinn_scaler_x, pinn_scaler_y), 'rf': (rf_model, rf_scaler_x, rf_scaler_y)}
            results_df, _ = weighted_uncertainty_ensemble(models, data, input_columns, target_columns, curiosity, weights, max_or_min)

        if results_df.empty:
            return results_df

        # --- Recalculate Utility with Selected Acquisition Function ---
        # This allows switching acquisition functions without retraining
        if acquisition_function != 'webslamd' and 'Utility' in results_df.columns:
            results_df = MLEngine._recalculate_utility(
                results_df, data, target_columns, weights, max_or_min,
                curiosity, acquisition_function
            )

        # --- Fallback Utility Calculation ---
        if 'Utility' not in results_df.columns or results_df['Utility'].isnull().all():
            pred_col = 'prediction' if 'prediction' in results_df.columns else target_columns[0]
            unc_col = 'uncertainty' if 'uncertainty' in results_df.columns else 'Uncertainty'
            
            preds = results_df[pred_col].values.reshape(-1, 1) if pred_col in results_df.columns else np.zeros((len(results_df), 1))
            uncs = results_df[unc_col].values.reshape(-1, 1) if unc_col in results_df.columns else np.ones((len(results_df), 1)) * 0.1

            # Novelty
            is_labeled = ~data[target_columns].isnull().any(axis=1)
            labeled_features = data.loc[is_labeled, input_columns].values
            candidate_features = data.loc[results_df.index, input_columns].values if len(results_df.index) > 0 else np.array([])
            
            if len(candidate_features) > 0 and len(labeled_features) > 0:
                novelty_scores = calculate_novelty(candidate_features, labeled_features)
                results_df['Novelty'] = novelty_scores

            # Use selected acquisition function
            acq_func = get_acquisition_function(acquisition_function)
            labeled_data = data.dropna(subset=target_columns)
            
            utility = acq_func.compute(
                predictions=preds,
                uncertainties=uncs,
                labeled_data=labeled_data,
                target_columns=target_columns,
                max_or_min=max_or_min,
                weights=weights,
                curiosity=curiosity
            )
            results_df['Utility'] = utility.flatten()

        results_df['Utility'] = pd.to_numeric(results_df['Utility'], errors='coerce').fillna(0.0)

        # Standardize Uncertainty
        if 'uncertainty' in results_df.columns:
            results_df['Uncertainty'] = results_df['uncertainty']
        
        if 'Uncertainty' not in results_df.columns or results_df['Uncertainty'].max() < 1e-6:
            results_df['Uncertainty'] = results_df['Utility'].abs() * 0.2 + 0.01

        # --- Apply Batch Selection ---
        from app.utils.batch_selector import select_batch
        results_df = select_batch(
            results_df, 
            n_samples=batch_size, 
            input_columns=input_columns,
            diversity_weight=0.3
        )
        logger.info(f"✅ Batch selection: {batch_size} samples selected (diversity_weight=0.3)")

        return results_df
    
    @staticmethod
    def _recalculate_utility(results_df, data, target_columns, weights, max_or_min,
                             curiosity, acquisition_function):
        """
        Recalculate utility scores using a different acquisition function.
        
        This allows comparing different acquisition strategies without retraining.
        """
        labeled_data = data.dropna(subset=target_columns)
        
        # Extract predictions and uncertainties from results
        predictions = np.zeros((len(results_df), len(target_columns)))
        uncertainties = np.zeros((len(results_df), len(target_columns)))
        
        for i, col in enumerate(target_columns):
            if col in results_df.columns:
                predictions[:, i] = results_df[col].values
            unc_col = f"Uncertainty ({col})"
            if unc_col in results_df.columns:
                uncertainties[:, i] = results_df[unc_col].values
            elif 'Uncertainty' in results_df.columns:
                uncertainties[:, i] = results_df['Uncertainty'].values
        
        # Get acquisition function and compute
        acq_func = get_acquisition_function(acquisition_function)
        utility = acq_func.compute(
            predictions=predictions,
            uncertainties=uncertainties,
            labeled_data=labeled_data,
            target_columns=target_columns,
            max_or_min=max_or_min,
            weights=weights,
            curiosity=curiosity
        )
        
        results_df['Utility'] = utility
        return results_df
    
    @staticmethod
    def list_available_models():
        """List all available model names."""
        return list(MODEL_CONFIG.keys()) + ['ensemble']
    
    @staticmethod
    def list_available_acquisitions():
        """List all available acquisition functions."""
        from app.acquisition import ACQUISITION_FUNCTIONS
        return list(ACQUISITION_FUNCTIONS.keys())
