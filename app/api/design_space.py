"""
Design Space API Routes

Handles design space creation, viewing, and management.
"""
import os
import logging
import itertools
from datetime import datetime
import numpy as np
import pandas as pd
from flask import (
    Blueprint, render_template, request, jsonify,
    session, redirect, url_for, send_from_directory
)
from werkzeug.utils import secure_filename

design_space_bp = Blueprint('design_space', __name__)
logger = logging.getLogger(__name__)


def generate_feature_values(feature):
    """Generate values for a feature based on its type."""
    if feature['type'] == 'continuous':
        if 'n_points' in feature and feature['n_points'] is not None:
            # Use linspace when n_points is specified
            return np.linspace(feature['min'], feature['max'], feature['n_points'])
        else:
            # Use arange when step is specified
            return np.arange(feature['min'], feature['max'] + feature['step'], feature['step'])
    return feature['values']


@design_space_bp.route("/scenario", methods=["GET"])
def scenario():
    """Render the scenario management page."""
    return render_template("scenario.html")


@design_space_bp.route("/api/scenario-to-design-space", methods=["GET"])
def scenario_to_design_space():
    """
    Analyze scenario data and generate feature definitions for Design Space Builder.
    
    Returns JSON with:
    - features: list of feature definitions (name, type, min, max, suggested step)
    - targets: suggested target columns (usually optimization objectives)
    """
    scenario_file = os.path.join(os.path.dirname(__file__), "..", "..", "data", "scenarios.csv")
    
    if not os.path.exists(scenario_file):
        return jsonify({"success": False, "error": "No scenario data found"}), 404
    
    try:
        df = pd.read_csv(scenario_file)
        
        features = []
        targets = []
        
        for col in df.columns:
            # Skip the Scenario name column
            if col.lower() == "scenario" or "scenario" in col.lower():
                continue
            
            # Try to convert to numeric
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            
            if not numeric_col.isna().all():
                # It's a numeric column - can be a feature
                min_val = float(numeric_col.min())
                max_val = float(numeric_col.max())
                
                # Calculate a reasonable step (divide range into ~10 steps)
                range_val = max_val - min_val
                if range_val > 0:
                    step = round(range_val / 10, 2)
                    if step == 0:
                        step = 0.1
                else:
                    step = 1.0
                
                feature = {
                    "name": col,
                    "type": "continuous",
                    "min": min_val,
                    "max": max_val,
                    "step": step,
                    "values": sorted(numeric_col.dropna().unique().tolist())
                }
                
                # Classify as target if it looks like an objective
                if any(keyword in col.lower() for keyword in ['cost', 'coverage', 'performance', 'efficiency']):
                    targets.append(col)
                else:
                    features.append(feature)
        
        return jsonify({
            "success": True,
            "features": features,
            "suggested_targets": targets,
            "total_scenarios": len(df),
            "material_name": "Scenario_Design"
        })
        
    except FileNotFoundError:
        return jsonify({"success": False, "error": "Scenario file not found."}), 404
    except Exception as e:
        logger.exception("Error analyzing scenario")
        return jsonify({"success": False, "error": "Failed to analyze scenario data."}), 500


@design_space_bp.route('/design-space')
def design_space():
    """Design space management page."""
    design_space_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'designspaces')
    history = []
    if os.path.exists(design_space_dir):
        for filename in sorted(os.listdir(design_space_dir), reverse=True):
            if filename.endswith('.csv'):
                filepath = os.path.join(design_space_dir, filename)
                history.append({'name': filename, 'path': filepath})
    return render_template('design_space.html', history=history)


@design_space_bp.route('/generate-design-space', methods=['POST'])
def generate_design_space():
    """Generate a new design space from form data."""
    data = request.form
    material_name = data.get('material_name')
    feature_names = data.getlist('feature_name')
    feature_types = data.getlist('feature_type')
    target_names = data.getlist('target_name')

    feature_definitions = []
    total_combinations = 1
    
    # Track separate indices for different field types
    continuous_idx = 0
    discrete_idx = 0

    for i in range(len(feature_names)):
        feature = {'name': feature_names[i], 'type': feature_types[i]}
        if feature['type'] == 'continuous':
            try:
                min_list = data.getlist('min')
                max_list = data.getlist('max')
                n_points_list = data.getlist('n_points')
                step_list = data.getlist('step')
                
                min_val = float(min_list[continuous_idx])
                max_val = float(max_list[continuous_idx])
                if min_val > max_val:
                    return "Invalid range for continuous feature.", 400
                feature['min'] = min_val
                feature['max'] = max_val
                
                # Check for n_points mode first (only set if not empty)
                n_points_str = n_points_list[continuous_idx] if continuous_idx < len(n_points_list) else ''
                step_str = step_list[continuous_idx] if continuous_idx < len(step_list) else ''
                
                if n_points_str and n_points_str.strip():
                    # n_points mode
                    n_points = int(n_points_str)
                    if n_points < 2:
                        return "n_points must be at least 2 for continuous feature.", 400
                    feature['n_points'] = n_points
                    feature['step'] = None
                    total_combinations *= n_points
                elif step_str and step_str.strip():
                    # step mode
                    step_val = float(step_str)
                    if step_val <= 0:
                        return "Step must be positive for continuous feature.", 400
                    feature['step'] = step_val
                    feature['n_points'] = None
                    total_combinations *= int((max_val - min_val) // step_val) + 1
                else:
                    return "Either step or n_points must be specified for continuous feature.", 400
                
                continuous_idx += 1
            except (ValueError, IndexError) as e:
                logger.error(f"Error parsing continuous feature: {e}")
                return "Invalid input for continuous feature.", 400
        else:
            # Discrete or categorical
            try:
                values_list = data.getlist('values')
                values_str = values_list[discrete_idx] if discrete_idx < len(values_list) else ''
                values = [v.strip() for v in values_str.split(',') if v.strip()]
                if not values:
                    return "Empty values for discrete/categorical feature.", 400
                feature['values'] = values
                total_combinations *= len(values)
                discrete_idx += 1
            except (ValueError, IndexError) as e:
                logger.error(f"Error parsing discrete/categorical feature: {e}")
                return "Invalid input for discrete/categorical feature.", 400
        feature_definitions.append(feature)

    if total_combinations > 100000 and 'confirm' not in data:
        return "Dataset is too large. Please confirm to proceed.", 400

    design_space_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'designspaces')
    os.makedirs(design_space_dir, exist_ok=True)
    
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

    return redirect(url_for('design_space.design_space'))


@design_space_bp.route('/download-design-space/<filename>')
def download_design_space(filename):
    """Download a design space file."""
    filename = secure_filename(filename)
    design_space_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'designspaces')
    filepath = os.path.join(design_space_dir, filename)
    
    # Verify path is within allowed directory
    if not os.path.abspath(filepath).startswith(os.path.abspath(design_space_dir)):
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    return send_from_directory(design_space_dir, filename, as_attachment=True)


@design_space_bp.route('/delete-design-space/<filename>', methods=['DELETE'])
def delete_design_space(filename):
    """Delete a design space file."""
    try:
        filename = secure_filename(filename)
        design_space_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'designspaces')
        filepath = os.path.join(design_space_dir, filename)
        
        # Verify path is within allowed directory
        if not os.path.abspath(filepath).startswith(os.path.abspath(design_space_dir)):
            return jsonify({'success': False, 'error': 'Access denied'}), 403
        
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@design_space_bp.route('/set-filepath-from-url', methods=['POST'])
def set_filepath_from_url():
    """Set session filepath from URL parameter."""
    filename = request.args.get('filename')
    if filename:
        design_space_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'designspaces')
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


@design_space_bp.route('/api/design-space-info', methods=['GET'])
def get_design_space_info():
    """Get info about a design space file (columns, etc.) for dashboard auto-loading."""
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'success': False, 'error': 'No filename provided.'})
    
    design_space_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'designspaces')
    filepath = os.path.join(design_space_dir, filename)
    
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'File not found.'})
    
    try:
        df = pd.read_csv(filepath)
        
        # Also set session variables for compatibility
        session['filepath'] = filepath
        session['filename'] = filename
        session['data_columns'] = df.columns.tolist()
        
        return jsonify({
            'success': True,
            'filename': filename,
            'columns': df.columns.tolist(),
            'row_count': len(df)
        })
    except Exception as e:
        logger.error(f"Error reading design space file: {e}")
        return jsonify({'success': False, 'error': str(e)})


@design_space_bp.route('/api/design-space-data/<filename>', methods=['GET'])
def get_design_space_data(filename):
    """
    Get design space data for editing in the UI.
    Returns columns with metadata (including which have NaN values) and all data rows.
    """
    filename = secure_filename(filename)
    design_space_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'designspaces')
    filepath = os.path.join(design_space_dir, filename)
    
    # Security check
    if not os.path.abspath(filepath).startswith(os.path.abspath(design_space_dir)):
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'File not found.'}), 404
    
    try:
        df = pd.read_csv(filepath)
        
        # Build column metadata
        columns_meta = []
        for col in df.columns:
            nan_count = int(df[col].isna().sum())
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            columns_meta.append({
                'name': col,
                'nan_count': nan_count,
                'has_nan': nan_count > 0,
                'is_numeric': is_numeric,
                'is_target': nan_count == len(df)  # All NaN = likely a target column
            })
        
        # Convert DataFrame to list of dicts, handling NaN values properly
        # Use pandas to_json with orient='records' which properly converts NaN to null
        import json
        data = json.loads(df.to_json(orient='records'))
        
        return jsonify({
            'success': True,
            'filename': filename,
            'columns': columns_meta,
            'data': data,
            'row_count': len(df)
        })
    except Exception as e:
        logger.error(f"Error reading design space file for editing: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@design_space_bp.route('/api/design-space-data/<filename>', methods=['POST'])
def save_design_space_data(filename):
    """
    Save updated column values to the design space CSV.
    Expects JSON body: { column: "column_name", updates: { row_index: value, ... } }
    """
    filename = secure_filename(filename)
    design_space_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'designspaces')
    filepath = os.path.join(design_space_dir, filename)
    
    # Security check
    if not os.path.abspath(filepath).startswith(os.path.abspath(design_space_dir)):
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'File not found.'}), 404
    
    try:
        payload = request.get_json()
        column = payload.get('column')
        updates = payload.get('updates', {})
        
        if not column:
            return jsonify({'success': False, 'error': 'No column specified.'}), 400
        
        df = pd.read_csv(filepath)
        
        if column not in df.columns:
            return jsonify({'success': False, 'error': f'Column "{column}" not found.'}), 400
        
        # Apply updates
        updated_count = 0
        for row_idx, value in updates.items():
            row_idx = int(row_idx)
            if 0 <= row_idx < len(df):
                # Convert to appropriate type
                if value is None or value == '' or value == 'null':
                    df.at[row_idx, column] = np.nan
                else:
                    try:
                        df.at[row_idx, column] = float(value)
                    except (ValueError, TypeError):
                        df.at[row_idx, column] = value
                updated_count += 1
        
        # Save back to CSV
        df.to_csv(filepath, index=False)
        
        logger.info(f"Updated {updated_count} values in column '{column}' of {filename}")
        
        return jsonify({
            'success': True,
            'message': f'Updated {updated_count} values in column "{column}".',
            'updated_count': updated_count
        })
    except Exception as e:
        logger.error(f"Error saving design space data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

