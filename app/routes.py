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

from app.utils.plot_generator import PlotGenerator
from app.utils.settings_manager import SettingsManager
from app.utils.trajectory_tracker import TrajectoryTracker

logging.basicConfig(level=logging.DEBUG)
main_bp = Blueprint('main', __name__)

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


@main_bp.route('/api/settings', methods=['GET'])
def get_settings():
    """Return current settings for the dashboard."""
    settings = SettingsManager.load_settings()
    return jsonify({'success': True, 'settings': settings})


@main_bp.route('/api/settings', methods=['POST'])
def save_settings():
    """Save settings from the dashboard UI."""
    try:
        new_settings = request.get_json()
        success = SettingsManager.save_settings(new_settings)
        if success:
            return jsonify({'success': True, 'message': 'Settings saved.'})
        else:
            return jsonify({'success': False, 'error': 'Failed to save settings.'}), 500
    except Exception as e:
        logging.error(f"Error saving settings: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@main_bp.route('/api/trajectory', methods=['GET'])
def get_trajectory():
    """Return current trajectory data for visualization."""
    try:
        summary = TrajectoryTracker.get_trajectory_summary()
        return jsonify({'success': True, 'trajectory': summary})
    except Exception as e:
        logging.error(f"Error getting trajectory: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@main_bp.route('/api/trajectory', methods=['DELETE'])
def clear_trajectory():
    """Clear trajectory history for new experiment run."""
    try:
        TrajectoryTracker.clear()
        return jsonify({'success': True, 'message': 'Trajectory cleared.'})
    except Exception as e:
        logging.error(f"Error clearing trajectory: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


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
        columns = data.columns.tolist()
        session['data_columns'] = columns
        session['filename'] = filename
        
        # Persist dataset info to settings for cross-page navigation
        SettingsManager.save_settings({
            'current_dataset': filename,
            'current_dataset_columns': columns
        })
        
        return jsonify({'success': True, 'columns': columns, 'filename': filename})
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
        
        # Parse a-priori configuration (new: with min/max support)
        apriori_config = config.get('apriori_columns', [])
        apriori_columns = [a['name'] for a in apriori_config] if apriori_config else []
        apriori_weights = np.array([float(a['weight']) for a in apriori_config]) if apriori_config else np.array([])
        apriori_max_or_min = [a['optimization'] for a in apriori_config] if apriori_config else []
        
        # Store a-priori config in request json for engine to use
        config['apriori_columns_names'] = apriori_columns
        config['apriori_weights'] = apriori_weights.tolist() if len(apriori_weights) > 0 else []
        config['apriori_max_or_min'] = apriori_max_or_min
        
        results_df = pd.DataFrame()

        # 2. Model Execution
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

        # Ensure Utility and Uncertainty exist (Engine should handle this, but double check)
        results_df['Utility'] = pd.to_numeric(results_df.get('Utility', 0), errors='coerce').fillna(0.0)
        
        if 'Uncertainty' in results_df.columns:
             results_df['Uncertainty'] = pd.to_numeric(results_df['Uncertainty'], errors='coerce').fillna(0.01)
        else:
             results_df['Uncertainty'] = 0.01

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
        
        cols_to_merge = ['Utility', 'Uncertainty', 'ML_Utility', 'Semantic_Score'] # Added debug cols
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
        current_mode = SettingsManager.get_setting("active_learning_mode", "ML_MODE")
        tsne_figure = PlotGenerator.create_tsne_input_space_plot(tsne_plot_df, input_columns, mode=current_mode)
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
        
        # Generate trajectory plot
        print(f"📈 Generating trajectory plot...")
        trajectory_summary = TrajectoryTracker.get_trajectory_summary()
        trajectory_plot = PlotGenerator.create_trajectory_plot(tsne_df, trajectory_summary, input_columns)
        distance_plot = PlotGenerator.create_distance_plot(trajectory_summary)
        print(f"✅ Trajectory plot generated ({trajectory_summary['total_iterations']} iterations)")

        print(f"📊 Preparing results table...")
        
        # Ensure results are sorted by Utility descending (SLAMD style)
        if 'Utility' in results_df.columns:
            results_df = results_df.sort_values(by='Utility', ascending=False)
            print(f"📊 After sort - First 5 Utility values: {results_df['Utility'].head(5).tolist()}")
            
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
            "prediction_error_plot": prediction_error_plot,
            "trajectory_plot": trajectory_plot,
            "distance_plot": distance_plot,
            "trajectory_summary": trajectory_summary
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