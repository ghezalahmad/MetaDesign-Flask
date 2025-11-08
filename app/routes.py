from flask import Blueprint, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import pandas as pd
from app.models.models import MAMLModel, evaluate_maml
import numpy as np

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/data-setup', methods=['GET', 'POST'])
def data_setup():
    if request.method == 'POST':
        print("Received POST request on /data-setup")
        if 'dataset' not in request.files:
            return jsonify({'success': False, 'error': 'No file part'})
        file = request.files['dataset']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'})
        if file:
            filename = secure_filename(file.filename)
            # This is not a good way to get the project root, but it will do for now
            upload_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)
            # Store the filepath in the session
            from flask import session
            session['filepath'] = filepath
            return jsonify({'success': True, 'message': f'{filename} uploaded successfully!'})
    return render_template('data_setup.html')

@main_bp.route('/experimentation')
def experimentation():
    return render_template('experimentation.html')

@main_bp.route('/run-experiment', methods=['POST'])
def run_experiment():
    print("Received POST request on /run-experiment")
    from flask import session
    filepath = session.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return render_template('experimentation.html', error="Please upload a dataset first.")

    data = pd.read_csv(filepath)
    model_name = request.form.get('model')
    curiosity = float(request.form.get('curiosity', 0.5))

    # For now, we'll just use MAML as an example
    input_columns = [col for col in data.columns if col.startswith('c_')]
    target_columns = [col for col in data.columns if col.startswith('target_')]

    model = MAMLModel(input_size=len(input_columns), output_size=len(target_columns))

    results_df = evaluate_maml(
        meta_model=model,
        data=data,
        input_columns=input_columns,
        target_columns=target_columns,
        curiosity=curiosity,
        weights=np.array([1.0] * len(target_columns)),
        max_or_min=['max'] * len(target_columns)
    )

    print(results_df)
    results_html = results_df.to_html(classes='table table-striped', index=False) if results_df is not None else "<p>No results to display.</p>"

    return render_template('experimentation.html', results=results_html)
