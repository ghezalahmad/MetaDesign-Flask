"""
Run Experiment API Routes

Handles the main experiment execution endpoint.
This is the core ML pipeline that trains models and returns predictions.
"""
import os
import sys
import logging
import json
import hashlib
import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify, session
from werkzeug.utils import secure_filename

from app.utils.plot_generator import PlotGenerator
from app.utils.settings_manager import SettingsManager
from app.utils.trajectory_tracker import TrajectoryTracker
from app.utils.decision_analysis import DecisionAnalyzer
from app.utils.session_store import resolve_dataset_path

run_experiment_bp = Blueprint('run_experiment', __name__)
logger = logging.getLogger(__name__)


def _build_tsne_cache_key(filepath, input_columns):
    """Build a t-SNE cache key that changes with dataset content and feature space."""
    if not filepath or not os.path.exists(filepath):
        return None

    mtime = os.path.getmtime(filepath)
    feature_signature = hashlib.sha256(
        json.dumps(input_columns or [], ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"{filepath}_{mtime}_tsne_v2_inputs_{feature_signature}"


def _build_tsne_graph_data(tsne_plot_df, input_columns, target_columns):
    """Return compact row data so the browser can redraw t-SNE parameter views."""
    if tsne_plot_df is None or tsne_plot_df.empty:
        return {
            'rows': [],
            'parameters': [],
            'numeric_parameters': [],
            'color_parameters': [],
            'overlay_parameters': []
        }

    df = tsne_plot_df.copy()

    if 'Row number' not in df.columns:
        df['Row number'] = df.index + 1

    df['TSNE_X'] = pd.to_numeric(df.get('tsne-2d-one', 0), errors='coerce').fillna(0.0)
    df['TSNE_Y'] = pd.to_numeric(df.get('tsne-2d-two', 0), errors='coerce').fillna(0.0)

    selected = df.get('Selected for Testing', pd.Series([False] * len(df), index=df.index))
    if selected.dtype == object:
        selected = selected.astype(str).str.lower().isin(['true', '1', 'yes'])
    else:
        selected = selected.fillna(False).astype(bool)

    is_train = df.get('is_train_data', pd.Series([False] * len(df), index=df.index))
    if is_train.dtype == object:
        is_train = is_train.astype(str).str.lower().isin(['true', '1', 'yes'])
    else:
        is_train = is_train.fillna(False).astype(bool)

    df['Population'] = np.where(
        selected,
        'Selected for Testing',
        np.where(is_train, 'Labelled', 'Predicted')
    )

    preferred_columns = [
        'Row number', 'TSNE_X', 'TSNE_Y', 'Population',
        'Utility', 'Uncertainty', 'ML_Utility', 'Semantic_Score',
        'Decision_Score', 'Pareto_Front', 'Constraint_Feasible', 'Trust_Score',
        'Trust_Flag', 'OOD_Risk', 'Decision_Action', 'Fidelity_Level',
        'Selected for Testing', 'is_train_data'
    ]
    preferred_columns.extend(input_columns or [])
    preferred_columns.extend(target_columns or [])

    export_columns = []
    for col in preferred_columns:
        if col in df.columns and col not in export_columns:
            export_columns.append(col)

    parameter_defs = []
    numeric_parameters = []
    categorical_parameters = []

    for col in export_columns:
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        unique_count = int(series.nunique(dropna=True))

        if is_numeric:
            numeric_parameters.append(col)
            param_type = 'numeric'
        elif unique_count <= 30:
            categorical_parameters.append(col)
            param_type = 'categorical'
        else:
            param_type = 'text'

        parameter_defs.append({
            'name': col,
            'label': col,
            'type': param_type,
            'unique_count': unique_count
        })

    overlay_parameters = ['None']
    overlay_parameters.extend([c for c in ['Population', 'Selected for Testing', 'is_train_data'] if c in export_columns])
    overlay_parameters.extend([c for c in categorical_parameters if c not in overlay_parameters])

    color_parameters = []
    color_parameters.extend([c for c in ['Decision_Score', 'Utility', 'Trust_Score', 'Uncertainty', 'Population'] if c in export_columns])
    color_parameters.extend([c for c in numeric_parameters if c not in color_parameters])
    color_parameters.extend([c for c in categorical_parameters if c not in color_parameters])

    export_df = df[export_columns].copy()
    rows = json.loads(export_df.to_json(orient='records'))

    return {
        'rows': rows,
        'parameters': parameter_defs,
        'numeric_parameters': numeric_parameters,
        'color_parameters': color_parameters,
        'overlay_parameters': overlay_parameters,
        'defaults': {
            'x': 'TSNE_X',
            'y': 'TSNE_Y',
            'color': 'Utility' if 'Utility' in export_columns else 'Population',
            'overlay': 'Population'
        }
    }


def _merge_result_columns_into_tsne(tsne_df, results_df, columns):
    """Merge result annotations into full t-SNE rows by row identity when possible."""
    if tsne_df is None or tsne_df.empty or results_df is None or results_df.empty:
        return tsne_df

    key_column = None
    for candidate in ['Row number', 'Idx_Sample', 'IDX_SAMPLE', 'idx_sample']:
        if candidate in tsne_df.columns and candidate in results_df.columns:
            key_column = candidate
            break

    if key_column:
        tsne_keys = tsne_df[key_column].astype(str)
        result_keys = results_df[key_column].astype(str)
        for col in columns:
            if col in results_df.columns:
                mapping = pd.Series(results_df[col].values, index=result_keys).to_dict()
                tsne_df[col] = tsne_keys.map(mapping)
        return tsne_df

    common_indices = tsne_df.index.intersection(results_df.index)
    for col in columns:
        if col in results_df.columns:
            tsne_df.loc[common_indices, col] = results_df.loc[common_indices, col]
    return tsne_df


@run_experiment_bp.route('/run-experiment', methods=['POST'])
def run_experiment():
    """Execute the active learning experiment pipeline."""
    try:
        # 1. Setup and Loading
        config = request.get_json()
        dataset_filename = config.get('dataset_filename')
        
        if dataset_filename:
            filepath = resolve_dataset_path(secure_filename(dataset_filename))
        else:
            filepath = session.get('filepath')

        if not filepath or not os.path.exists(filepath):
            logger.error(f"Filepath not found: {filepath}")
            return jsonify({'success': False, 'error': 'Please upload or select a dataset first.'})

        data = pd.read_csv(filepath)
        if 'Row number' not in data.columns:
            data['Row number'] = range(1, len(data) + 1)
        
        model_name = config.get('model')
        curiosity = float(config.get('curiosity', 0.5))
        input_columns = config.get('input_columns')
        target_columns_config = config.get('target_columns')
        
        target_columns = [t['name'] for t in target_columns_config]
        weights = np.array([float(t['weight']) for t in target_columns_config])
        max_or_min = [t['optimization'] for t in target_columns_config]
        
        # Parse a-priori configuration (with min/max support)
        apriori_config = config.get('apriori_columns', [])
        apriori_columns = [a['name'] for a in apriori_config] if apriori_config else []
        apriori_weights = np.array([float(a['weight']) for a in apriori_config]) if apriori_config else np.array([])
        apriori_max_or_min = [a['optimization'] for a in apriori_config] if apriori_config else []
        
        # Store a-priori config in request json for engine to use
        config['apriori_columns_names'] = apriori_columns
        config['apriori_weights'] = apriori_weights.tolist() if len(apriori_weights) > 0 else []
        config['apriori_max_or_min'] = apriori_max_or_min
        
        results_df = pd.DataFrame()

        # 2. Active Learning Engine Execution
        from app.engines.hybrid_engine import HybridEngine
        
        # Pass the request json as config
        results_df = HybridEngine.run_experiment(data, config)

        # 3. Safety Checks & Post-Processing
        if results_df.empty:
            # Check for preprocessing errors
            if hasattr(results_df, 'attrs') and 'preprocessing_errors' in results_df.attrs:
                errors = results_df.attrs['preprocessing_errors']
                error_msg = errors[0] if errors else 'Unknown preprocessing error'
                return jsonify({
                    'success': False, 
                    'error': error_msg,
                    'error_type': 'preprocessing',
                    'all_errors': errors
                })
            return jsonify({'success': False, 'error': 'Model execution failed to produce results.'})

        engine_attrs = dict(getattr(results_df, 'attrs', {}))
        results_df, decision_analysis = DecisionAnalyzer.apply(
            results_df=results_df,
            source_df=data,
            input_columns=input_columns,
            target_configs=target_columns_config,
            apriori_configs=apriori_config,
            config=config
        )
        results_df.attrs.update(engine_attrs)

        # Ensure Utility and Uncertainty exist
        results_df['Utility'] = pd.to_numeric(results_df.get('Utility', 0), errors='coerce').fillna(0.0)
        
        if 'Uncertainty' in results_df.columns:
            results_df['Uncertainty'] = pd.to_numeric(results_df['Uncertainty'], errors='coerce').fillna(0.01)
        else:
            results_df['Uncertainty'] = 0.01

        # 4. Generate Visualizations
        logger.info("Starting visualization generation...")
        
        tsne_df = data.copy()
        
        if 'Row number' not in tsne_df.columns:
            tsne_df['Row number'] = range(1, len(tsne_df) + 1)
        
        tsne_df["is_train_data"] = ~tsne_df[target_columns].isnull().any(axis=1)
        
        logger.debug(f"TSNE Preparation: {len(tsne_df)} samples, "
                     f"{tsne_df['is_train_data'].sum()} labelled, "
                     f"{(~tsne_df['is_train_data']).sum()} predicted")
        
        tsne_cache_key = _build_tsne_cache_key(filepath, input_columns)
        
        tsne_df = PlotGenerator._run_tsne(tsne_df, input_columns, cache_key=tsne_cache_key)
        
        cols_to_merge = [
            'Utility', 'Uncertainty', 'ML_Utility', 'Semantic_Score', 'Selected for Testing',
            'Decision_Score', 'Pareto_Front', 'Constraint_Feasible', 'Trust_Score',
            'Trust_Flag', 'OOD_Risk', 'Decision_Action', 'Fidelity_Level'
        ]
        tsne_df = _merge_result_columns_into_tsne(tsne_df, results_df, cols_to_merge)

        MAX_PLOT_POINTS = 2000
        if len(tsne_df) > MAX_PLOT_POINTS:
            logger.debug(f"Downsampling TSNE from {len(tsne_df)} to {MAX_PLOT_POINTS} points")
            train_mask = tsne_df['is_train_data']
            train_df = tsne_df[train_mask]
            pred_df = tsne_df[~train_mask]
            
            n_pred_sample = MAX_PLOT_POINTS - len(train_df)
            if n_pred_sample > 0 and len(pred_df) > n_pred_sample:
                pred_sample = pred_df.sample(n=n_pred_sample, random_state=42)
                tsne_plot_df = pd.concat([train_df, pred_sample])
            else:
                tsne_plot_df = tsne_df
        else:
            tsne_plot_df = tsne_df
        
        logger.debug(f"Generating TSNE plot with {len(tsne_plot_df)} points")
        current_mode = SettingsManager.get_setting("active_learning_mode", "ML_MODE")
        tsne_figure = PlotGenerator.create_tsne_input_space_plot(tsne_plot_df, input_columns, mode=current_mode)
        tsne_graph_data = _build_tsne_graph_data(tsne_plot_df, input_columns, target_columns)

        target_scatter_figure = PlotGenerator.create_target_scatter_plot(results_df, target_columns)
        uncertainty_plot = PlotGenerator.create_uncertainty_plot(results_df, target_columns)
        history_plot = PlotGenerator.create_optimization_history_plot(data, target_columns)
        
        SURFACE_MAX_POINTS = 500
        
        if len(results_df) > SURFACE_MAX_POINTS:
            logger.debug(f"Downsampling surface from {len(results_df)} to {SURFACE_MAX_POINTS} points")
            surface_df = results_df.sample(n=SURFACE_MAX_POINTS, random_state=42)
        else:
            surface_df = results_df
            
        if 'tsne-2d-one' in tsne_df.columns and 'tsne-2d-two' in tsne_df.columns:
            surface_df = surface_df.copy()
            surface_df['tsne-2d-one'] = tsne_df.loc[surface_df.index, 'tsne-2d-one']
            surface_df['tsne-2d-two'] = tsne_df.loc[surface_df.index, 'tsne-2d-two']
        
        utility_surface_plot = PlotGenerator.create_utility_surface_plot(surface_df, input_columns)

        prediction_error_plot = {'data': [], 'layout': {'title': 'Error Plot N/A'}}
        
        # Generate trajectory plot
        trajectory_summary = TrajectoryTracker.get_trajectory_summary()
        trajectory_plot = PlotGenerator.create_trajectory_plot(tsne_df, trajectory_summary, input_columns)
        distance_plot = PlotGenerator.create_distance_plot(trajectory_summary)

        # Ensure results are sorted by decision quality, then Utility descending
        if 'Decision_Score' in results_df.columns:
            results_df = results_df.sort_values(by='Decision_Score', ascending=False)
        elif 'Utility' in results_df.columns:
            results_df = results_df.sort_values(by='Utility', ascending=False)
            
        if len(results_df) > 500:
            table_df = results_df.head(500)
            table_html = table_df.to_html(classes="table table-striped", index=False)
            table_html += f'<p class="text-muted"><em>Showing top 500 of {len(results_df)} results by Utility. Download full results using the buttons above.</em></p>'
        else:
            table_html = results_df.to_html(classes="table table-striped", index=False)
        
        # Generate feature importance plot
        feature_importances = {}
        labeled_data_for_importance = data.dropna(subset=target_columns)
        if len(labeled_data_for_importance) > 2:
            for col in input_columns:
                if col in labeled_data_for_importance.columns:
                    try:
                        corr = labeled_data_for_importance[col].corr(labeled_data_for_importance[target_columns[0]])
                        feature_importances[col] = float(abs(corr)) if not np.isnan(corr) else 0.0
                    except:
                        feature_importances[col] = 0.0
        else:
            feature_importances = {col: 0.5 for col in input_columns}
        feature_importance_plot = PlotGenerator.create_feature_importance_plot(feature_importances, input_columns)
        
        # Generate prediction vs actual plot
        # For prediction vs actual, we need labeled samples with predictions
        # Use simple leave-one-out cross-validation style predictions
        labeled_data_for_plot = data.dropna(subset=target_columns).copy()
        
        if len(labeled_data_for_plot) > 2:
            from sklearn.neighbors import KNeighborsRegressor
            X_labeled = labeled_data_for_plot[input_columns].values
            y_labeled = labeled_data_for_plot[target_columns].values
            
            # Generate cross-val style predictions using KNN
            k = min(3, len(X_labeled) - 1)
            predictions_cv = np.zeros_like(y_labeled)
            
            for i in range(len(X_labeled)):
                # Leave-one-out: train on all except i, predict i
                X_train = np.delete(X_labeled, i, axis=0)
                y_train = np.delete(y_labeled, i, axis=0)
                
                knn = KNeighborsRegressor(n_neighbors=min(k, len(X_train)), algorithm='ball_tree')
                knn.fit(X_train, y_train)
                predictions_cv[i] = knn.predict(X_labeled[i:i+1])
            
            # Add predictions to the labeled data
            for j, col in enumerate(target_columns):
                labeled_data_for_plot[f'Predicted_{col}'] = predictions_cv[:, j].tolist()
        
        prediction_actual_plot = PlotGenerator.create_prediction_actual_plot(labeled_data_for_plot, target_columns)
        
        logger.info("Visualization generation complete")

        # Extract LLM trace if present (set by HybridEngine for LLM/Hybrid modes)
        llm_trace = engine_attrs.get('llm_trace') or results_df.attrs.get('llm_trace', None)
        run_warnings = []
        run_warnings.extend(engine_attrs.get('warnings', []) or [])
        run_warnings.extend(results_df.attrs.get('warnings', []) or [])
        run_warnings = list(dict.fromkeys(run_warnings))

        response_data = {
            "success": True,
            "warnings": run_warnings,
            "results_table": table_html,
            "tsne_figure": tsne_figure,
            "tsne_graph_data": tsne_graph_data,
            "target_scatter_figure": target_scatter_figure,
            "uncertainty_plot": uncertainty_plot,
            "history_plot": history_plot, 
            "utility_surface_plot": utility_surface_plot,
            "prediction_error_plot": prediction_error_plot,
            "trajectory_plot": trajectory_plot,
            "distance_plot": distance_plot,
            "trajectory_summary": trajectory_summary,
            "feature_importance_plot": feature_importance_plot,
            "prediction_actual_plot": prediction_actual_plot,
            "decision_analysis": decision_analysis,
            "llm_trace": llm_trace,
        }

        try:
            response = jsonify(response_data)
            logger.debug(f"Response created: {len(response.get_data()) / 1024:.1f} KB")
            return response
        except Exception as json_error:
            logger.exception("JSON serialization error in /run-experiment")
            
            return jsonify({
                'success': False, 
                'error': 'Failed to serialize response. Please try with a smaller dataset.'
            }), 500

    except ValueError as e:
        # Specific handling for validation errors (safe to show to users)
        logger.warning(f"Validation error in /run-experiment: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400
    except FileNotFoundError as e:
        logger.warning(f"File not found in /run-experiment: {e}")
        return jsonify({'success': False, 'error': 'Dataset file not found.'}), 404
    except Exception as e:
        # Log full traceback for debugging, but return sanitized message
        logger.exception("An error occurred in /run-experiment")
        return jsonify({'success': False, 'error': 'An internal error occurred. Please check your configuration and try again.'}), 500
