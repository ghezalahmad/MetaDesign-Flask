import logging
logging.debug("Importing flask")
from flask import Blueprint, render_template, request, jsonify, session
logging.debug("Importing os")
import os
logging.debug("Importing secure_filename")
from werkzeug.utils import secure_filename
logging.debug("Importing pandas")
import pandas as pd
logging.debug("Importing MAMLModel")
from app.models.models import MAMLModel, evaluate_maml
logging.debug("Importing ReptileModel")
from app.models.reptile_model import ReptileModel, evaluate_reptile, reptile_train
logging.debug("Importing ProtoNetModel")
from app.models.protonet_model import ProtoNetModel, evaluate_protonet, protonet_train
logging.debug("Importing RFModel")
from app.models.rf_model import RFModel, train_rf_model, evaluate_rf_model
logging.debug("Importing PINNModel")
from app.models.pinn_model import PINNModel, pinn_train, evaluate_pinn
logging.debug("Importing LolopyRFModel")
from app.models.lolopy_model import LolopyRFModel, train_lolopy_model, evaluate_lolopy_model
logging.debug("Importing weighted_uncertainty_ensemble")
from app.models.ensemble import weighted_uncertainty_ensemble
logging.debug("Importing numpy")
import numpy as np
logging.debug("Importing TSNE")
from sklearn.manifold import TSNE
logging.debug("Finished imports")

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('dashboard.html')

@main_bp.route('/upload', methods=['POST'])
def upload_data():
    if 'dataset' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})
    file = request.files['dataset']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})
    if file:
        filename = secure_filename(file.filename)
        upload_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        session['filepath'] = filepath

        data = pd.read_csv(filepath)
        session['data_columns'] = data.columns.tolist()
        return jsonify({'success': True, 'columns': data.columns.tolist(), 'filename': filename})

@main_bp.route('/run-experiment', methods=['POST'])
def run_experiment():
    filepath = session.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'Please upload a dataset first.'})

    data = pd.read_csv(filepath)
    config = request.get_json()
    model_name = config.get('model')
    curiosity = float(config.get('curiosity', 0.5))
    input_columns = config.get('input_columns')
    target_columns_config = config.get('target_columns')

    target_columns = [t['name'] for t in target_columns_config]
    weights = np.array([float(t['weight']) for t in target_columns_config])
    max_or_min = [t['optimization'] for t in target_columns_config]

    if model_name == 'maml':
        model = MAMLModel(input_size=len(input_columns), output_size=len(target_columns))
        results_df = evaluate_maml(
            meta_model=model, data=data, input_columns=input_columns,
            target_columns=target_columns, curiosity=curiosity,
            weights=weights, max_or_min=max_or_min
        )
    elif model_name == 'reptile':
        model = ReptileModel(input_size=len(input_columns), output_size=len(target_columns))
        model, _, _ = reptile_train(model, data, input_columns, target_columns, epochs=50, learning_rate=0.001, num_tasks=5, batch_size=16)
        results_df = evaluate_reptile(
            model=model, data=data, input_columns=input_columns,
            target_columns=target_columns, curiosity=curiosity,
            weights=weights, max_or_min=max_or_min
        )
    elif model_name == 'protonet':
        model = ProtoNetModel(input_size=len(input_columns), output_size=len(target_columns))
        model, _, _ = protonet_train(model, data, input_columns, target_columns, epochs=50, learning_rate=0.001, num_tasks=5, num_shot=5, num_query=5)
        results_df = evaluate_protonet(
            model=model, data=data, input_columns=input_columns,
            target_columns=target_columns, curiosity=curiosity,
            weights=weights, max_or_min=max_or_min
        )
    elif model_name == 'rf':
        model, _, _ = train_rf_model(data, input_columns, target_columns)
        results_df = evaluate_rf_model(
            rf_model=model, data=data, input_columns=input_columns,
            target_columns=target_columns, curiosity=curiosity,
            weights=weights, max_or_min=max_or_min
        )
    elif model_name == 'pinn':
        model = PINNModel(input_size=len(input_columns), output_size=len(target_columns))
        model, _, _ = pinn_train(model, data, input_columns, target_columns, epochs=100, learning_rate=0.001, physics_loss_weight=0.1, batch_size=32)
        results_df = evaluate_pinn(
            model=model, data=data, input_columns=input_columns,
            target_columns=target_columns, curiosity=curiosity,
            weights=weights, max_or_min=max_or_min
        )
    elif model_name == 'lolopy':
        model, _, _ = train_lolopy_model(data, input_columns, target_columns)
        results_df = evaluate_lolopy_model(
            model=model, data=data, input_columns=input_columns,
            target_columns=target_columns, curiosity=curiosity,
            weights_targets=weights, max_or_min_targets=max_or_min
        )
    elif model_name == 'ensemble':
        # Train PINN model for the ensemble
        pinn_model = PINNModel(input_size=len(input_columns), output_size=len(target_columns))
        pinn_model, pinn_scaler_x, pinn_scaler_y = pinn_train(
            pinn_model, data, input_columns, target_columns, epochs=100, learning_rate=0.001, physics_loss_weight=0.1, batch_size=32
        )

        # Train RF model for the ensemble
        rf_model, rf_scaler_x, rf_scaler_y = train_rf_model(data, input_columns, target_columns)

        models = {
            'pinn': (pinn_model, pinn_scaler_x, pinn_scaler_y),
            'rf': (rf_model, rf_scaler_x, rf_scaler_y)
        }

        results_df, _ = weighted_uncertainty_ensemble(
            models=models, data=data, input_columns=input_columns,
            target_columns=target_columns, curiosity=curiosity,
            weights=weights, max_or_min_objectives=max_or_min
        )
    else:
        results_df = pd.DataFrame()

    # Generate visualization data
    tsne_data = None
    if not data[input_columns].empty:
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(data[input_columns])
        tsne_data = {
            'x': tsne_results[:, 0].tolist(),
            'y': tsne_results[:, 1].tolist(),
        }

    scatter_data = None
    if not results_df.empty and len(target_columns) > 0:
        scatter_data = {
            'x': results_df[target_columns[0]].tolist(),
            'y': results_df['Uncertainty'].tolist() if 'Uncertainty' in results_df else [],
            'labels': results_df.index.tolist()
        }

    parallel_coordinates_data = None
    if not results_df.empty:
        dimensions = []
        for col in input_columns + target_columns:
            dimensions.append({
                'label': col,
                'values': results_df[col].tolist()
            })
        if 'Uncertainty' in results_df.columns:
            dimensions.append({
                'label': 'Uncertainty',
                'values': results_df['Uncertainty'].tolist()
            })

        parallel_coordinates_data = {
            'type': 'parcoords',
            'line': {
                'color': 'blue'
            },
            'dimensions': dimensions
        }

    correlation_heatmap_data = None
    if not data.empty:
        # Compute correlation only on numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            corr = numeric_data.corr()
            correlation_heatmap_data = {
                'z': corr.values.tolist(),
                'x': corr.columns.tolist(),
                'y': corr.index.tolist()
            }
        else:
            correlation_heatmap_data = None


    prediction_error_data = None
    if not results_df.empty and 'predictions' in results_df.columns and not data.empty:
        labeled_data = data.dropna(subset=target_columns)
        if not labeled_data.empty and len(labeled_data) == len(results_df):
            prediction_error_data = {
                'actual': labeled_data[target_columns[0]].tolist(),
                'predicted': results_df['predictions'].tolist()
            }

    return jsonify({
        'success': True,
        'results_table': results_df.to_html(classes='table table-striped', index=False),
        'tsne_data': tsne_data,
        'scatter_data': scatter_data,
        'parallel_coordinates_data': parallel_coordinates_data,
        'correlation_heatmap_data': correlation_heatmap_data,
        'prediction_error_data': prediction_error_data
    })

