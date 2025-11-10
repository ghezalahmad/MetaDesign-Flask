from flask import Blueprint, render_template, request, jsonify, session
import os
from werkzeug.utils import secure_filename
import pandas as pd
from app.models.maml_model import MAMLModel, evaluate_maml
from app.models.reptile_model import ReptileModel, evaluate_reptile, reptile_train
from app.models.protonet_model import ProtoNetModel, evaluate_protonet, protonet_train
from app.models.rf_model import RFModel, train_rf_model, evaluate_rf_model
from app.models.pinn_model import PINNModel, pinn_train, evaluate_pinn
from app.models.lolopy_model import LolopyRFModel, train_lolopy_model, evaluate_lolopy_model
from app.models.ensemble import weighted_uncertainty_ensemble
import numpy as np

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/data-setup', methods=['GET', 'POST'])
def data_setup():
    if request.method == 'POST':
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
            return jsonify({'success': True, 'columns': data.columns.tolist()})

    return render_template('data_setup.html')

@main_bp.route('/experimentation')
def experimentation():
    data_columns = session.get('data_columns', [])
    return render_template('experimentation.html', columns=data_columns)

@main_bp.route('/run-experiment', methods=['POST'])
def run_experiment():
    filepath = session.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return render_template('experimentation.html', error="Please upload a dataset first.")

    data = pd.read_csv(filepath)
    model_name = request.form.get('model')
    curiosity = float(request.form.get('curiosity', 0.5))
    input_columns = request.form.getlist('input_columns')
    target_columns = request.form.getlist('target_columns')

    weights = np.array([float(w) for w in request.form.getlist('weights')])
    max_or_min = request.form.getlist('max_or_min')


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
        # You'll need to define how to create the ensemble of models
        models = {} # This needs to be populated with trained models
        results_df, _ = weighted_uncertainty_ensemble(
            models=models, data=data, input_columns=input_columns,
            target_columns=target_columns, curiosity=curiosity,
            weights=weights, max_or_min_objectives=max_or_min
        )
    else:
        results_df = None

    results_html = results_df.to_html(classes='table table-striped', index=False) if results_df is not None else "<p>No results to display.</p>"
    data_columns = session.get('data_columns', [])

    return render_template('experimentation.html', results=results_html, columns=data_columns)

