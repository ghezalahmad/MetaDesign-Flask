
import pandas as pd
import numpy as np
import logging

from app.models.models import MAMLModel, evaluate_maml
from app.models.reptile_model import ReptileModel, evaluate_reptile, reptile_train
from app.models.gp_model import GPModel, train_gp_model, evaluate_gp_model
from app.models.protonet_model import ProtoNetModel, evaluate_protonet, protonet_train
from app.models.dkl_surrogate_model import DKLModel, train_dkl_model, evaluate_dkl_model
from app.models.rf_model import train_rf_model, evaluate_rf_model, RFModel
from app.models.pinn_model import PINNModel, pinn_train, evaluate_pinn
from app.models.lolopy_model import LolopyRFModel, train_lolopy_model, evaluate_lolopy_model
from app.models.ensemble import weighted_uncertainty_ensemble
from app.utils.utils import calculate_utility, calculate_novelty
from app.utils.experiment_preprocessor import ExperimentPreprocessor
from app.utils.experiment_data import ExperimentData

# --- Model Configuration Dictionary ---
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
}

class MLEngine:
    @staticmethod
    def run_experiment(data, model_name, input_columns, target_columns_config, curiosity=0.5, apriori_config=None):
        """
        Executes the ML pipeline: Preprocessing -> Training -> Evaluation -> Utility Calculation
        
        Now uses WEBSLAMD-style ExperimentData object for configuration.
        """
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
        
        logging.info(f"📊 ExperimentData: {exp.model}, {len(exp.feature_names)} features, "
                    f"{len(exp.target_names)} targets, labelled={len(exp.index_all_labelled)}, "
                    f"to_predict={len(exp.index_predicted)}")
        
        # --- WEBSLAMD-style Preprocessing ---
        preprocess_config = {
            'model': model_name,
            'input_columns': input_columns,
            'target_columns': target_columns_config,
            'apriori_columns': apriori_config or [],
            'curiosity': curiosity
        }
        
        data, preprocess_result = ExperimentPreprocessor.preprocess(exp.dataframe, preprocess_config)
        exp.dataframe = data  # Update with preprocessed data
        
        # Check for preprocessing errors
        if not preprocess_result['valid']:
            logging.error(f"❌ Preprocessing failed: {preprocess_result['errors']}")
            error_df = pd.DataFrame()
            error_df.attrs['preprocessing_errors'] = preprocess_result['errors']
            return error_df
        
        # Log any warnings
        for warning in preprocess_result.get('warnings', []):
            logging.warning(f"⚠️ Preprocessing: {warning}")
        
        # Update feature_names if any were dropped
        if preprocess_result.get('dropped_columns'):
            exp.feature_names = [c for c in exp.feature_names if c not in preprocess_result['dropped_columns']]
        
        # Extract for compatibility with existing code
        target_columns = exp.target_names
        weights = np.array(exp.target_weights)
        max_or_min = exp.target_max_or_min
        input_columns = exp.feature_names
        
        results_df = pd.DataFrame()
        config_entry = MODEL_CONFIG.get(model_name)
        input_size = len(input_columns)
        output_size = len(target_columns)

        if config_entry:
            if model_name in ['maml', 'reptile', 'protonet', 'pinn']:
                model = config_entry['model_class'](input_size=input_size, output_size=output_size)
            else:
                model = None

            if 'train_func' in config_entry:
                train_func = config_entry['train_func']
                train_params = config_entry.get('train_params')
                
                if model_name in ['dkl', 'gp']:
                    if train_params is not None:
                        model_params = train_params[0] if isinstance(train_params, tuple) else train_params
                        model, _, _ = train_func(data, input_columns, target_columns, model_params)
                    else:
                        model, _, _ = train_func(data, input_columns, target_columns, {})
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

        # --- Utility Calculation (if missing) ---
        if 'Utility' not in results_df.columns or results_df['Utility'].isnull().all():
             pred_col = 'prediction' if 'prediction' in results_df.columns else target_columns[0]
             unc_col = 'uncertainty' if 'uncertainty' in results_df.columns else 'Uncertainty'
             
             if pred_col in results_df.columns:
                 preds = results_df[pred_col].values.reshape(-1, 1)
             else:
                 preds = np.zeros((len(results_df), 1))
                 
             if unc_col in results_df.columns:
                 uncs = results_df[unc_col].values.reshape(-1, 1)
             else:
                 uncs = np.ones((len(results_df), 1)) * 0.1

             # Novelty Calculation
             is_labeled = ~data[target_columns].isnull().any(axis=1)
             labeled_features = data.loc[is_labeled, input_columns].values
             candidate_features = data.loc[results_df.index, input_columns].values
             
             novelty_scores = calculate_novelty(candidate_features, labeled_features)
             results_df['Novelty'] = novelty_scores

             utility = calculate_utility(
                 predictions=preds,
                 uncertainties=uncs,
                 novelty=novelty_scores,
                 curiosity=curiosity,
                 weights=weights,
                 max_or_min=max_or_min,
                 acquisition="UCB"
             )
             results_df['Utility'] = utility.flatten()

        results_df['Utility'] = pd.to_numeric(results_df['Utility'], errors='coerce').fillna(0.0)

        # Standardize Uncertainty Column Name
        if 'uncertainty' in results_df.columns:
            results_df['Uncertainty'] = results_df['uncertainty']
        
        if 'Uncertainty' not in results_df.columns or results_df['Uncertainty'].max() < 1e-6:
             results_df['Uncertainty'] = results_df['Utility'].abs() * 0.2 + 0.01

        return results_df
