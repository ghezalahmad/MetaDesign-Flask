import logging
import os
import itertools
from datetime import datetime
import numpy as np
import pandas as pd
from flask import (
    Blueprint, render_template, request, jsonify,
    session, redirect, url_for, send_from_directory
)
from werkzeug.utils import secure_filename
from sklearn.manifold import TSNE

# ---- Model Imports ----
from app.models.models import MAMLModel, evaluate_maml
from app.models.reptile_model import ReptileModel, evaluate_reptile, reptile_train
from app.models.protonet_model import ProtoNetModel, evaluate_protonet, protonet_train
from app.models.rf_model import train_rf_model, evaluate_rf_model, RFModel
from app.models.pinn_model import PINNModel, pinn_train, evaluate_pinn
from app.models.lolopy_model import LolopyRFModel, train_lolopy_model, evaluate_lolopy_model
from app.models.ensemble import weighted_uncertainty_ensemble
from app.utils.plot_generator import PlotGenerator

logging.basicConfig(level=logging.DEBUG)
main_bp = Blueprint('main', __name__)

# ======================================================
# ROUTES
# ======================================================

@main_bp.route('/')
def index():
    return render_template('index.html')


@main_bp.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


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
        return redirect(url_for('main.dashboard', ds=filename))

    return redirect(url_for('main.design_space'))


def generate_feature_values(feature):
    if feature['type'] == 'continuous':
        return np.arange(feature['min'], feature['max'] + feature['step'], feature['step'])
    return feature['values']


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

    data = pd.read_csv(filepath)
    session['data_columns'] = data.columns.tolist()
    return jsonify({'success': True, 'columns': data.columns.tolist(), 'filename': filename})


@main_bp.route('/run-experiment', methods=['POST'])
def run_experiment():
    try:
        filepath = session.get('filepath')
        if not filepath or not os.path.exists(filepath):
            logging.error("Filepath not found in session or file does not exist.")
            return jsonify({'success': False, 'error': 'Please upload a dataset first.'})

        logging.info(f"Loading data from: {filepath}")
        data = pd.read_csv(filepath)
        data['Row number'] = data.index
        logging.info(f"Original data shape: {data.shape}")

        config = request.get_json()
        logging.info(f"Received experiment config: {config}")

        model_name = config.get('model')
        curiosity = float(config.get('curiosity', 0.5))
        input_columns = config.get('input_columns')
        target_columns_config = config.get('target_columns')

        target_columns = [t['name'] for t in target_columns_config]
        weights = np.array([float(t['weight']) for t in target_columns_config])
        max_or_min = [t['optimization'] for t in target_columns_config]

        # ---- Model Execution ----
        if model_name == 'maml':
            model = MAMLModel(input_size=len(input_columns), output_size=len(target_columns))
            results_df = evaluate_maml(model, data, input_columns, target_columns, curiosity, weights, max_or_min)
        elif model_name == 'reptile':
            model = ReptileModel(input_size=len(input_columns), output_size=len(target_columns))
            model, _, _ = reptile_train(model, data, input_columns, target_columns, 50, 0.001, 5, 16)
            results_df = evaluate_reptile(model, data, input_columns, target_columns, curiosity, weights, max_or_min)
        elif model_name == 'protonet':
            model = ProtoNetModel(input_size=len(input_columns), output_size=len(target_columns))
            model, _, _ = protonet_train(model, data, input_columns, target_columns, 50, 0.001, 5, 5, 5)
            results_df = evaluate_protonet(model, data, input_columns, target_columns, curiosity, weights, max_or_min)
        elif model_name == 'rf':
            model, _, _ = train_rf_model(data, input_columns, target_columns)
            results_df = evaluate_rf_model(model, data, input_columns, target_columns, curiosity, weights, max_or_min)
        elif model_name == 'pinn':
            model = PINNModel(input_size=len(input_columns), output_size=len(target_columns))
            model, _, _ = pinn_train(model, data, input_columns, target_columns, 100, 0.001, 0.1, 32)
            results_df = evaluate_pinn(model, data, input_columns, target_columns, curiosity, weights, max_or_min)
        elif model_name == 'lolopy':
            model, _, _ = train_lolopy_model(data, input_columns, target_columns)
            results_df = evaluate_lolopy_model(model, data, input_columns, target_columns, curiosity, weights, max_or_min)
        elif model_name == 'ensemble':
            pinn_model = PINNModel(input_size=len(input_columns), output_size=len(target_columns))
            pinn_model, pinn_scaler_x, pinn_scaler_y = pinn_train(pinn_model, data, input_columns, target_columns, 100, 0.001, 0.1, 32)
            rf_model, rf_scaler_x, rf_scaler_y = train_rf_model(data, input_columns, target_columns)
            models = {'pinn': (pinn_model, pinn_scaler_x, pinn_scaler_y), 'rf': (rf_model, rf_scaler_x, rf_scaler_y)}
            results_df, _ = weighted_uncertainty_ensemble(models, data, input_columns, target_columns, curiosity, weights, max_or_min)
        else:
            results_df = pd.DataFrame()

        logging.info(f"Results DataFrame shape after model evaluation: {results_df.shape}")

        # ---- Build Visualization ----
        # Prepare data for t-SNE plot
        tsne_df = data.copy()
        tsne_df["is_train_data"] = ~tsne_df[target_columns].isnull().any(axis=1)

        # Merge results to get Utility scores
        if "Utility" in results_df.columns:
            tsne_df = tsne_df.merge(results_df[['Utility']], left_index=True, right_index=True, how='left')
            tsne_df["Utility"] = tsne_df["Utility"].fillna(0)
        else:
            tsne_df["Utility"] = 0.0

        logging.info(f"t-SNE DataFrame shape: {tsne_df.shape}")

        tsne_figure = PlotGenerator.create_tsne_input_space_plot(tsne_df, input_columns)

        # Prepare data for scatter plot
        scatter_df = results_df.copy()
        scatter_df["is_train_data"] = False  # All results are suggestions

        logging.info(f"Scatter DataFrame shape: {scatter_df.shape}")

        target_scatter_figure = PlotGenerator.create_target_scatter_plot(scatter_df, target_columns)

        response_data = {
            "success": True,
            "results_table": results_df.to_html(classes="table table-striped", index=False),
            "tsne_figure": tsne_figure,
            "target_scatter_figure": target_scatter_figure
        }

        # ---- Return to Frontend ----
        logging.info("Experiment completed successfully")
        return jsonify(response_data)

    except Exception as e:
        logging.exception("An error occurred in /run-experiment")
        return jsonify({'success': False, 'error': str(e)}), 500