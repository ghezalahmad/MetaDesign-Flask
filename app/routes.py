import logging
import os
import itertools
import json
from datetime import datetime
import numpy as np
import pandas as pd
from flask import (
    Blueprint, render_template, request, jsonify,
    session, redirect, url_for, send_from_directory
)
from werkzeug.utils import secure_filename

from app.models.models import MAMLModel, evaluate_maml, meta_train
from app.models.reptile_model import ReptileModel, evaluate_reptile, reptile_train
from app.models.gp_model import GPModel, train_gp_model, evaluate_gp_model
from app.models.protonet_model import ProtoNetModel, evaluate_protonet, protonet_train
from app.models.dkl_surrogate_model import DKLModel, train_dkl_model, evaluate_dkl_model
from app.models.rf_model import train_rf_model, evaluate_rf_model, RFModel
from app.models.pinn_model import PINNModel, pinn_train, evaluate_pinn
from app.models.lolopy_model import LolopyRFModel, train_lolopy_model, evaluate_lolopy_model
from app.models.ensemble import weighted_uncertainty_ensemble
from app.models.ensemble import weighted_uncertainty_ensemble
# from app.models.bayesian_optimizer import BayesianOptimizer # Removed broken usage
from app.utils.utils import calculate_utility, calculate_novelty
from app.utils.plot_generator import PlotGenerator

logging.basicConfig(level=logging.DEBUG)
main_bp = Blueprint('main', __name__)

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
        'train_params': ({}),  # Empty dict for model_params
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
        'train_params': ({}),  # Empty dict for model_params
    },
}

# ======================================================
# UTILITY FUNCTIONS
# ======================================================

def generate_feature_values(feature):
    if feature['type'] == 'continuous':
        return np.arange(feature['min'], feature['max'] + feature['step'], feature['step'])
    return feature['values']

# ======================================================
# ROUTES
# ======================================================

@main_bp.route('/')
def index():
    return render_template('index.html')


@main_bp.route('/dashboard')
def dashboard():
    initial_data = {
        'data_columns': session.get('data_columns', []),
        'filename': session.get('filename', None)
    }
    return render_template('dashboard.html', initial_data=json.dumps(initial_data))


@main_bp.route("/scenario", methods=["GET", "POST"])
def scenario():
    scenario_file = os.path.join(os.path.dirname(__file__), "..", "data", "scenarios.csv")

    default_data = [
        ["Scenario 1", 10125, 240, 8.0, 15, 675, 2, 10, 5, 120, 4.5],
        ["Scenario 2", 10125, 240, 8.0, 15, 675, 2, 10, 5, 120, 4.5],
        ["Scenario 3", 10800, 240, 8.0, 16, 675, 2, 12, 4, 120, 4.8],
        ["Scenario 4", 10200, 240, 8.0, 24, 425, 2, 16, 8, 120, 7.3],
        ["Scenario 5 (Selected Scenario)", 10200, 224, 7.5, 34, 300, 8, 6, 4, 28, 10.3],
        ["Scenario ALL SAMPLES", 165000, 120, 4.0, 330, 500, 1, 330, 0, 120, 100.0]
    ]

    columns = [
        "Scenario", "Total Cost (EUR)", "Total Duration (days)", "Total Duration (months)",
        "Total Recipes Tested", "Cost per Recipe (EUR)", "No. of cycles",
        "No. of initial recipes", "No. of recipes per cycle",
        "Duration per Cycle (days)", "Coverage of material space (%)"
    ]

    if os.path.exists(scenario_file):
        data = pd.read_csv(scenario_file)
    else:
        data = pd.DataFrame(default_data, columns=columns)
        data.to_csv(scenario_file, index=False)

    if request.method == "POST":
        updated_data = request.get_json()
        pd.DataFrame(updated_data, columns=columns).to_csv(scenario_file, index=False)
        return jsonify({"success": True, "message": "Scenarios updated successfully."})

    return render_template("scenario.html", data=data.to_dict(orient="records"), columns=columns)


@main_bp.route('/design-space')
def design_space():
    design_space_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'designspaces')
    history = []
    if os.path.exists(design_space_dir):
        for filename in sorted(os.listdir(design_space_dir), reverse=True):
            if filename.endswith('.csv'):
                filepath = os.path.join(design_space_dir, filename)
                history.append({'name': filename, 'path': filepath})
    return render_template('design_space.html', history=history)


@main_bp.route('/generate-design-space', methods=['POST'])
def generate_design_space():
    data = request.form
    material_name = data.get('material_name')
    feature_names = data.getlist('feature_name')
    feature_types = data.getlist('feature_type')
    target_names = data.getlist('target_name')

    feature_definitions = []
    total_combinations = 1

    for i in range(len(feature_names)):
        feature = {'name': feature_names[i], 'type': feature_types[i]}
        if feature['type'] == 'continuous':
            try:
                min_val = float(data.getlist('min')[i])
                max_val = float(data.getlist('max')[i])
                step_val = float(data.getlist('step')[i])
                if min_val > max_val or step_val <= 0:
                    return "Invalid range or step for continuous feature.", 400
                feature['min'] = min_val
                feature['max'] = max_val
                feature['step'] = step_val
                total_combinations *= ((max_val - min_val) // step_val) + 1
            except (ValueError, IndexError):
                return "Invalid input for continuous feature.", 400
        else:
            values = [v.strip() for v in data.getlist('values')[i].split(',')]
            if not values:
                return "Empty values for discrete/categorical feature.", 400
            feature['values'] = values
            total_combinations *= len(values)
        feature_definitions.append(feature)

    if total_combinations > 100000 and 'confirm' not in data:
        return "Dataset is too large. Please confirm to proceed.", 400

    design_space_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'designspaces')
    product_iter = itertools.product(*[generate_feature_values(f) for f in feature_definitions])
    df = pd.DataFrame(list(product_iter), columns=feature_names)
    df.insert(0, 'Idx_Sample', range(1, len(df) + 1))

    for target in target_names:
        df[target] = np.nan

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"designspace_{secure_filename(material_name)}_{timestamp}.csv"
    filepath = os.path.join(design_space_dir, filename)
    df.to_csv(filepath, index=False)

    if data.get('action') == 'open':
        session['filepath'] = filepath
        session['filename'] = filename
        return redirect(url_for('main.dashboard', ds=filename))

    return redirect(url_for('main.design_space'))


@main_bp.route('/download-design-space/<filename>')
def download_design_space(filename):
    design_space_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'designspaces')
    return send_from_directory(design_space_dir, filename, as_attachment=True)


@main_bp.route('/delete-design-space/<filename>', methods=['DELETE'])
def delete_design_space(filename):
    try:
        design_space_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'designspaces')
        filepath = os.path.join(design_space_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@main_bp.route('/set-filepath-from-url', methods=['POST'])
def set_filepath_from_url():
    filename = request.args.get('filename')
    if filename:
        design_space_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'designspaces')
        filepath = os.path.join(design_space_dir, filename)
        if os.path.exists(filepath):
            session['filepath'] = filepath
            try:
                data = pd.read_csv(filepath)
                session['data_columns'] = data.columns.tolist()
                session['filename'] = filename
                
                response = {'success': True, 'columns': data.columns.tolist(), 'filename': filename}
                return jsonify(response)
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
    return jsonify({'success': False, 'error': 'File not found.'})


@main_bp.route('/upload', methods=['POST'])
def upload_data():
    if 'dataset' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})
    file = request.files['dataset']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})

    filename = secure_filename(file.filename)
    upload_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    os.makedirs(upload_folder, exist_ok=True)

    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)
    session['filepath'] = filepath

    try:
        data = pd.read_csv(filepath)
        session['data_columns'] = data.columns.tolist()
        session['filename'] = filename 
        return jsonify({'success': True, 'columns': data.columns.tolist(), 'filename': filename})
    except Exception as e:
        return jsonify({'success': False, 'error': f"Failed to read CSV: {str(e)}"})


@main_bp.route('/run-experiment', methods=['POST'])
def run_experiment():
    try:
        # 1. Setup and Loading
        config = request.get_json()
        dataset_filename = config.get('dataset_filename')
        
        if dataset_filename:
             upload_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
             filepath = os.path.join(upload_folder, secure_filename(dataset_filename))
             if not os.path.exists(filepath):
                 design_space_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'designspaces')
                 filepath = os.path.join(design_space_dir, secure_filename(dataset_filename))
        else:
             filepath = session.get('filepath')

        if not filepath or not os.path.exists(filepath):
            logging.error(f"Filepath not found: {filepath}")
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
        
        results_df = pd.DataFrame()

        # 2. Model Execution
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
                
                # DKL and GP have different signatures - they don't take model as first arg
                if model_name in ['dkl', 'gp']:
                    if train_params is not None:
                        # train_params is a tuple with a single dict, extract it
                        model_params = train_params[0] if isinstance(train_params, tuple) else train_params
                        model, _, _ = train_func(data, input_columns, target_columns, model_params)
                    else:
                        model, _, _ = train_func(data, input_columns, target_columns, {})
                elif train_params is not None:
                    model, _, _ = train_func(model, data, input_columns, target_columns, *train_params)
                else:
                    model, _, _ = train_func(data, input_columns, target_columns)

            evaluate_func = config_entry['evaluate_func']
            
            # DKL and GP have different evaluate signatures
            if model_name in ['dkl', 'gp']:
                # These expect: (model, labeled_data, candidate_inputs, input_columns, target_columns, weights, max_or_min, curiosity)
                labeled_data = data.dropna(subset=target_columns)
                candidate_data = data[data[target_columns[0]].isnull()] if isinstance(target_columns, list) else data[data[target_columns].isnull()]
                candidate_inputs = candidate_data[input_columns]
                results_df = evaluate_func(model, labeled_data, candidate_inputs, input_columns, target_columns, weights, max_or_min, curiosity)
            else:
                # Standard signature: (model, data, input_columns, target_columns, curiosity, weights, max_or_min)
                results_df = evaluate_func(model, data, input_columns, target_columns, curiosity, weights, max_or_min)

        elif model_name == 'ensemble':
            pinn_model = PINNModel(input_size=input_size, output_size=output_size)
            pinn_model, pinn_scaler_x, pinn_scaler_y = pinn_train(pinn_model, data, input_columns, target_columns, 100, 0.001, 0.1, 32)
            rf_model, rf_scaler_x, rf_scaler_y = train_rf_model(data, input_columns, target_columns)
            models = {'pinn': (pinn_model, pinn_scaler_x, pinn_scaler_y), 'rf': (rf_model, rf_scaler_x, rf_scaler_y)}
            results_df, _ = weighted_uncertainty_ensemble(models, data, input_columns, target_columns, curiosity, weights, max_or_min)

        # 3. Safety Checks
        if results_df.empty:
             return jsonify({'success': False, 'error': 'Model execution failed to produce results.'})

        if 'Utility' not in results_df.columns or results_df['Utility'].isnull().all():
             # optimizer = BayesianOptimizer(data[input_columns].values, data[target_columns].values, target_columns_config)
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

             # --- SLAMD-like Bayesian Optimization Logic ---
             # 1. Calculate Novelty
             # 'results_df' contains predictions for unlabeled/candidate samples only
             # We need to calculate novelty for these candidates relative to labeled samples
             
             # Identify labeled data (rows where ALL targets are present)
             is_labeled = ~data[target_columns].isnull().any(axis=1)
             labeled_features = data.loc[is_labeled, input_columns].values
             
             # Get features for the samples in results_df
             # results_df should have the same index as the unlabeled rows in data
             candidate_features = data.loc[results_df.index, input_columns].values
             
             # Calculate novelty for candidate samples relative to labeled set
             novelty_scores = calculate_novelty(candidate_features, labeled_features)
             results_df['Novelty'] = novelty_scores

             # 2. Calculate Utility
             # predictions: (n_samples, 1) - we use the primary prediction column
             # uncertainties: (n_samples, 1)
             # novelty: (n_samples, 1)
             
             utility = calculate_utility(
                 predictions=preds,
                 uncertainties=uncs,
                 novelty=novelty_scores,
                 curiosity=curiosity,
                 weights=weights,
                 max_or_min=max_or_min,
                 acquisition="UCB" # Default to UCB-like logic inside calculate_utility
             )
             results_df['Utility'] = utility.flatten()

        results_df['Utility'] = pd.to_numeric(results_df['Utility'], errors='coerce').fillna(0.0)

        if 'uncertainty' in results_df.columns:
            results_df['Uncertainty'] = results_df['uncertainty']
        
        if 'Uncertainty' not in results_df.columns or results_df['Uncertainty'].max() < 1e-6:
             results_df['Uncertainty'] = results_df['Utility'].abs() * 0.2 + 0.01

        # SLAMD APPROACH: Utility is already normalized by the model
        # But we need to ensure it's in a reasonable range for visualization
        # SLAMD uses z-score normalization (mean=0, std=1), then clips to reasonable bounds
        
        # Normalize utility using z-score if values are too large
        utility_mean = results_df['Utility'].mean()
        utility_std = results_df['Utility'].std()
        
        if utility_std > 0 and (results_df['Utility'].abs().max() > 10):
            # If utility values are outside typical range, normalize them
            results_df['Utility'] = (results_df['Utility'] - utility_mean) / utility_std
            print(f"✅ Utility z-score normalized (mean={utility_mean:.2f}, std={utility_std:.2f})")
        
        # Round to 6 decimal places like SLAMD
        results_df['Utility'] = results_df['Utility'].round(6)

        # 4. Generate Visualizations (OPTIMIZED)
        print("📊 Starting visualization generation...")
        
        tsne_df = data.copy()
        
        if 'Row number' not in tsne_df.columns:
            tsne_df['Row number'] = range(1, len(tsne_df) + 1)
        
        tsne_df["is_train_data"] = ~tsne_df[target_columns].isnull().any(axis=1)
        
        print(f"📊 TSNE Preparation:")
        print(f"   Total samples: {len(tsne_df)}")
        print(f"   Training samples (labelled): {tsne_df['is_train_data'].sum()}")
        print(f"   Prediction samples (predicted): {(~tsne_df['is_train_data']).sum()}")
        
        
        tsne_cache_key = None
        if filepath and os.path.exists(filepath):
             mtime = os.path.getmtime(filepath)
             tsne_cache_key = f"{filepath}_{mtime}"
        
        tsne_df = PlotGenerator._run_tsne(tsne_df, input_columns, cache_key=tsne_cache_key)
        
        cols_to_merge = ['Utility', 'Uncertainty']
        common_indices = tsne_df.index.intersection(results_df.index)
        
        for col in cols_to_merge:
            if col in results_df.columns:
                tsne_df.loc[common_indices, col] = results_df.loc[common_indices, col]

        MAX_PLOT_POINTS = 2000
        if len(tsne_df) > MAX_PLOT_POINTS:
            print(f"⚡ Downsampling TSNE DISPLAY from {len(tsne_df)} to {MAX_PLOT_POINTS} points...")
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
        
        print(f"📈 Generating TSNE plot with {len(tsne_plot_df)} points...")
        tsne_figure = PlotGenerator.create_tsne_input_space_plot(tsne_plot_df, input_columns)
        print("✅ TSNE plot generated")

        print(f"📈 Generating target scatter plot...")
        target_scatter_figure = PlotGenerator.create_target_scatter_plot(results_df, target_columns)
        print("✅ Target scatter plot generated")

        print(f"📈 Generating uncertainty plot...")
        uncertainty_plot = PlotGenerator.create_uncertainty_plot(results_df, target_columns)
        print("✅ Uncertainty plot generated")

        print(f"📈 Generating optimization history...")
        history_plot = PlotGenerator.create_optimization_history_plot(data, target_columns)
        print("✅ History plot generated")
        
        print(f"📈 Generating utility surface...")
        SURFACE_MAX_POINTS = 500
        
        if len(results_df) > SURFACE_MAX_POINTS:
            print(f"⚡ Downsampling surface plot from {len(results_df)} to {SURFACE_MAX_POINTS} points...")
            surface_df = results_df.sample(n=SURFACE_MAX_POINTS, random_state=42)
        else:
            surface_df = results_df
            
        if 'tsne-2d-one' in tsne_df.columns and 'tsne-2d-two' in tsne_df.columns:
            surface_df = surface_df.copy()
            surface_df['tsne-2d-one'] = tsne_df.loc[surface_df.index, 'tsne-2d-one']
            surface_df['tsne-2d-two'] = tsne_df.loc[surface_df.index, 'tsne-2d-two']
        
        utility_surface_plot = PlotGenerator.create_utility_surface_plot(surface_df, input_columns)
        print("✅ Utility surface plot generated")

        prediction_error_plot = {'data': [], 'layout': {'title': 'Error Plot N/A'}}

        print(f"📊 Preparing results table...")
        
        # Ensure results are sorted by Utility descending (SLAMD style)
        if 'Utility' in results_df.columns:
            results_df = results_df.sort_values(by='Utility', ascending=False)
            
        if len(results_df) > 500:
            table_df = results_df.head(500)
            table_html = table_df.to_html(classes="table table-striped", index=False)
            table_html += f'<p class="text-muted"><em>Showing top 500 of {len(results_df)} results by Utility. Download full results using the buttons above.</em></p>'
        else:
            table_html = results_df.to_html(classes="table table-striped", index=False)
        
        print("✅ Results table prepared")
        print("🎉 All visualizations complete!")

        import sys
        print(f"📊 Response object sizes:")
        print(f"   TSNE figure: {sys.getsizeof(tsne_figure) / 1024:.1f} KB")
        print(f"   Scatter figure: {sys.getsizeof(target_scatter_figure) / 1024:.1f} KB")
        print(f"   Table HTML: {len(table_html) / 1024:.1f} KB")

        response_data = {
            "success": True,
            "results_table": table_html,
            "tsne_figure": tsne_figure,
            "target_scatter_figure": target_scatter_figure,
            "uncertainty_plot": uncertainty_plot,
            "history_plot": history_plot, 
            "utility_surface_plot": utility_surface_plot,
            "prediction_error_plot": prediction_error_plot
        }

        print("📤 Sending response to client...")
        
        try:
            response = jsonify(response_data)
            print(f"✅ JSON response created successfully (size: {len(response.get_data()) / 1024:.1f} KB)")
            return response
        except Exception as json_error:
            print(f"❌ JSON serialization error: {json_error}")
            import traceback
            traceback.print_exc()
            
            return jsonify({
                'success': False, 
                'error': f'Failed to serialize response: {str(json_error)}'
            }), 500

    except Exception as e:
        logging.exception("An error occurred in /run-experiment")
        return jsonify({'success': False, 'error': str(e)}), 500