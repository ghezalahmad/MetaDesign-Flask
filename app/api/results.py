"""
Results API Routes

Handles project management, cycle creation, sample updates, and CSV synchronization
for the active learning lab results workflow.
"""

import os
import json
import hashlib
import logging
import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify, abort
from werkzeug.utils import secure_filename

from app.database import db, Project, Cycle, Sample
from app.utils.plot_generator import PlotGenerator
from app.utils.session_store import get_session_id, resolve_dataset_path

results_bp = Blueprint('results', __name__, url_prefix='/api/results')
logger = logging.getLogger(__name__)

IDENTITY_COLUMNS = ['Idx_Sample', 'IDX_SAMPLE', 'idx_sample', 'IdxSample', 'Row number', 'row_number', 'Index', 'index']
TSNE_EXCLUDE_HINTS = {
    'utility', 'novelty', 'uncertainty', 'decision_score', 'trust_score',
    'ood_risk', 'pareto_rank', 'constraint_count', 'cost_penalty',
    'selected_for_lab', 'cycle_number', 'result_sample_id', 'cost'
}
TSNE_TARGET_HINTS = ['target', 'strength', 'slump', 'mpa', 'measured_', 'prediction_error_']


def _get_session_project_or_404(project_id):
    project = Project.query.filter_by(id=project_id, session_id=get_session_id()).first()
    if not project:
        abort(404)
    return project


def _get_session_sample_or_404(sample_id):
    sample = Sample.query.get_or_404(sample_id)
    cycle = db.session.get(Cycle, sample.cycle_id)
    project = db.session.get(Project, cycle.project_id) if cycle else None
    if not project or project.session_id != get_session_id():
        abort(404)
    return sample


def _value_matches(series, value):
    if value is None or value == '':
        return pd.Series([False] * len(series), index=series.index)

    numeric_series = pd.to_numeric(series, errors='coerce')
    numeric_value = pd.to_numeric(pd.Series([value]), errors='coerce').iloc[0]
    if pd.notna(numeric_value):
        return numeric_series == numeric_value

    return series.astype(str) == str(value)


def _find_sample_row(df, sample):
    """Find the original dataset row for a tracked sample."""
    row_data = sample.get_row_data()

    for col in IDENTITY_COLUMNS:
        if col not in df.columns:
            continue

        candidate_values = []
        if col in row_data:
            candidate_values.append(row_data.get(col))
        candidate_values.append(sample.idx_sample)

        for value in candidate_values:
            mask = _value_matches(df[col], value)
            if mask.any():
                return mask, col

    row_number = row_data.get('Row number') or sample.idx_sample
    try:
        position = int(float(row_number)) - 1
    except (TypeError, ValueError):
        position = None

    if position is not None and 0 <= position < len(df):
        mask = pd.Series(False, index=df.index)
        mask.iloc[position] = True
        return mask, 'row position'

    return None, None


def _sync_sample_lab_results(sample):
    """Write one sample's lab results back into its project CSV."""
    cycle = db.session.get(Cycle, sample.cycle_id)
    project = db.session.get(Project, cycle.project_id) if cycle else None

    if not project or not project.dataset_path or not os.path.exists(project.dataset_path):
        return {
            'success': False,
            'error': 'Dataset file not found',
            'sample_id': sample.id
        }

    lab_results = sample.get_lab_results()
    if not lab_results or not any(v is not None and v != '' for v in lab_results.values()):
        return {
            'success': False,
            'error': 'No lab results to sync',
            'sample_id': sample.id
        }

    df = pd.read_csv(project.dataset_path)
    row_mask, match_column = _find_sample_row(df, sample)
    if row_mask is None or not row_mask.any():
        return {
            'success': False,
            'error': f'Sample {sample.idx_sample} not found in dataset',
            'sample_id': sample.id
        }

    updated_columns = []
    for col, value in lab_results.items():
        if value is None or value == '':
            continue
        if col not in df.columns:
            df[col] = pd.NA
        df.loc[row_mask, col] = value
        updated_columns.append(col)

    if not updated_columns:
        return {
            'success': False,
            'error': 'No non-empty lab result values to sync',
            'sample_id': sample.id
        }

    df.to_csv(project.dataset_path, index=False)
    sample.status = 'completed'

    return {
        'success': True,
        'sample_id': sample.id,
        'dataset_path': project.dataset_path,
        'matched_by': match_column,
        'updated_columns': updated_columns
    }


def _request_float(name, default, min_value=None, max_value=None):
    value = request.args.get(name, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _request_int(name, default, min_value=None, max_value=None):
    return int(_request_float(name, default, min_value, max_value))


def _get_requested_feature_columns(df):
    raw_values = request.args.getlist('input_columns')
    if not raw_values and request.args.get('input_columns'):
        raw_values = request.args.get('input_columns', '').split(',')
    requested = [c.strip() for c in raw_values if c and c.strip()]
    if requested:
        return [c for c in requested if c in df.columns]
    return _infer_tsne_feature_columns(df)


def _infer_tsne_feature_columns(df):
    """Pick likely material/design-space features for t-SNE."""
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    features = []
    for col in numeric_cols:
        normalized = col.lower()
        if pd.api.types.is_bool_dtype(df[col]):
            continue
        if col in IDENTITY_COLUMNS or normalized in TSNE_EXCLUDE_HINTS:
            continue
        if any(hint in normalized for hint in TSNE_TARGET_HINTS):
            continue
        if normalized.startswith('predicted_') or 'uncertainty' in normalized:
            continue
        if normalized.startswith('lab_') or normalized.startswith('measured_') or normalized.startswith('prediction_error_'):
            continue
        features.append(col)
    return features[:20]


def _collect_lab_result_columns(project):
    columns = []
    for cycle in project.cycles:
        for col in cycle.get_lab_result_columns():
            if col and col not in columns:
                columns.append(col)
    return columns


def _annotate_results_tsne_rows(df, project, lab_result_columns):
    df = df.copy()
    if 'Row number' not in df.columns:
        df['Row number'] = range(1, len(df) + 1)

    df['Cycle_Number'] = pd.NA
    df['Cycle_Label'] = 'Not selected'
    df['Lab_Status'] = 'Untested'
    df['Result_Sample_ID'] = pd.NA
    df['Selected_For_Lab'] = False

    for col in lab_result_columns:
        measured_col = f'Measured_{col}'
        if measured_col not in df.columns:
            df[measured_col] = pd.NA

    for cycle in sorted(project.cycles, key=lambda c: c.cycle_number):
        for sample in cycle.samples:
            row_mask, _ = _find_sample_row(df, sample)
            if row_mask is None or not row_mask.any():
                continue

            df.loc[row_mask, 'Cycle_Number'] = cycle.cycle_number
            df.loc[row_mask, 'Cycle_Label'] = f'Cycle {cycle.cycle_number}'
            df.loc[row_mask, 'Lab_Status'] = sample.status
            df.loc[row_mask, 'Result_Sample_ID'] = sample.id
            df.loc[row_mask, 'Selected_For_Lab'] = True

            lab_results = sample.get_lab_results()
            predictions = sample.get_predictions()
            for col, value in lab_results.items():
                if value is None or value == '':
                    continue
                if col not in df.columns:
                    df[col] = pd.NA
                df.loc[row_mask, col] = value
                measured_col = f'Measured_{col}'
                df.loc[row_mask, measured_col] = value

                predicted_value = predictions.get(f'Predicted_{col}')
                if predicted_value is not None and predicted_value != '':
                    try:
                        error_col = f'Prediction_Error_{col}'
                        df[error_col] = pd.to_numeric(df.get(error_col, pd.NA), errors='coerce')
                        df.loc[row_mask, error_col] = float(value) - float(predicted_value)
                    except (TypeError, ValueError):
                        logger.debug("Skipping non-numeric prediction error for %s", col)

            for key, value in predictions.items():
                if key not in df.columns and value is not None and value != '':
                    df[key] = pd.NA
                if key in df.columns:
                    df.loc[row_mask, key] = value

    return df


def _build_tsne_quality_warnings(df, feature_columns):
    warnings = []
    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        warnings.append(f"Missing t-SNE feature columns were ignored: {', '.join(missing_features)}")

    valid_features = [c for c in feature_columns if c in df.columns]
    if not valid_features:
        warnings.append("No numeric feature columns are available for t-SNE.")
        return warnings

    feature_df = df[valid_features].apply(pd.to_numeric, errors='coerce')
    constant_cols = [c for c in valid_features if feature_df[c].nunique(dropna=True) <= 1]
    if constant_cols:
        warnings.append(f"Constant columns do not help t-SNE: {', '.join(constant_cols[:8])}")

    sparse_cols = [c for c in valid_features if feature_df[c].isna().mean() > 0.3]
    if sparse_cols:
        warnings.append(f"Columns with more than 30% missing values may add noise: {', '.join(sparse_cols[:8])}")

    duplicate_count = int(feature_df.fillna(0).duplicated().sum())
    if duplicate_count:
        warnings.append(f"{duplicate_count} rows have duplicate t-SNE feature values.")

    if len(df) > 2000:
        warnings.append("Large datasets can make t-SNE slow. Consider a representative subset if interaction feels heavy.")

    return warnings


def _build_results_tsne_cache_key(project, feature_columns, options):
    if not project.dataset_path or not os.path.exists(project.dataset_path):
        return None

    signature = {
        'path': project.dataset_path,
        'mtime': os.path.getmtime(project.dataset_path),
        'features': feature_columns,
        'options': options,
        'version': 'results-tsne-v1'
    }
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=True).encode('utf-8')).hexdigest()[:16]
    return f"{project.dataset_path}_results_tsne_{digest}"


def _downsample_results_tsne_df(df, max_points, random_seed):
    if len(df) <= max_points:
        return df, None

    selected = df[df.get('Selected_For_Lab', False) == True]
    remaining = df.drop(index=selected.index, errors='ignore')
    sample_size = max(0, int(max_points) - len(selected))
    if sample_size > 0 and len(remaining) > sample_size:
        remaining = remaining.sample(n=sample_size, random_state=int(random_seed))

    sampled = pd.concat([selected, remaining]).sort_index()
    warning = (
        f"t-SNE was computed on {len(sampled)} of {len(df)} rows for responsiveness. "
        "All selected lab samples were kept."
    )
    return sampled, warning


def _build_results_tsne_payload(df, feature_columns, lab_result_columns, warnings, options):
    df = df.copy()
    df['TSNE_X'] = pd.to_numeric(df.get('tsne-2d-one', 0), errors='coerce').fillna(0.0)
    df['TSNE_Y'] = pd.to_numeric(df.get('tsne-2d-two', 0), errors='coerce').fillna(0.0)

    preferred = [
        'Row number', 'Idx_Sample', 'TSNE_X', 'TSNE_Y', 'Cycle_Number', 'Cycle_Label',
        'Lab_Status', 'Selected_For_Lab', 'Result_Sample_ID'
    ]
    for col in lab_result_columns:
        preferred.extend([col, f'Measured_{col}', f'Predicted_{col}', f'Prediction_Error_{col}'])
    preferred.extend(feature_columns)

    export_columns = []
    for col in preferred:
        if col in df.columns and col not in export_columns:
            export_columns.append(col)

    numeric_parameters = []
    categorical_parameters = []
    for col in export_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_parameters.append(col)
        elif df[col].nunique(dropna=True) <= 40:
            categorical_parameters.append(col)

    color_parameters = []
    for col in numeric_parameters + categorical_parameters:
        if col not in color_parameters and col not in {'TSNE_X', 'TSNE_Y'}:
            color_parameters.append(col)

    overlay_parameters = ['None']
    for col in ['Lab_Status', 'Cycle_Label', 'Selected_For_Lab']:
        if col in export_columns:
            overlay_parameters.append(col)

    default_color = None
    for col in lab_result_columns:
        if f'Prediction_Error_{col}' in export_columns:
            default_color = f'Prediction_Error_{col}'
            break
        if f'Measured_{col}' in export_columns:
            default_color = f'Measured_{col}'
            break
        if col in export_columns:
            default_color = col
            break
    if default_color is None:
        default_color = 'Cycle_Number' if 'Cycle_Number' in export_columns else color_parameters[0] if color_parameters else 'Lab_Status'

    rows = json.loads(df[export_columns].to_json(orient='records'))
    return {
        'rows': rows,
        'feature_columns': feature_columns,
        'feature_candidates': _infer_tsne_feature_columns(df),
        'lab_result_columns': lab_result_columns,
        'numeric_parameters': numeric_parameters,
        'color_parameters': color_parameters,
        'overlay_parameters': overlay_parameters,
        'warnings': warnings,
        'options': options,
        'defaults': {
            'x': 'TSNE_X',
            'y': 'TSNE_Y',
            'color': default_color,
            'overlay': 'Lab_Status'
        }
    }


# ============================================================
# Project Endpoints
# ============================================================

@results_bp.route('/projects', methods=['GET'])
def get_projects():
    """List all projects."""
    projects = Project.query.filter_by(session_id=get_session_id()).order_by(Project.created_at.desc()).all()
    return jsonify({
        'success': True,
        'projects': [p.to_dict() for p in projects]
    })


@results_bp.route('/projects', methods=['POST'])
def create_project():
    """Create a new project."""
    data = request.get_json()
    
    name = data.get('name')
    dataset_path = resolve_dataset_path(data.get('dataset_path'), must_exist=False)
    
    if not name or not dataset_path:
        return jsonify({'success': False, 'error': 'Name and dataset_path are required'}), 400
    
    # Check if project with same name exists
    existing = Project.query.filter_by(name=name, session_id=get_session_id()).first()
    if existing:
        return jsonify({'success': False, 'error': f'Project "{name}" already exists'}), 400
    
    project = Project(name=name, dataset_path=dataset_path, session_id=get_session_id())
    db.session.add(project)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'project': project.to_dict()
    }), 201


@results_bp.route('/projects/<int:project_id>', methods=['GET'])
def get_project(project_id):
    """Get a project with its cycles."""
    project = _get_session_project_or_404(project_id)
    
    data = project.to_dict()
    data['cycles'] = [c.to_dict(include_samples=True) for c in project.cycles]
    
    return jsonify({
        'success': True,
        'project': data
    })


@results_bp.route('/projects/<int:project_id>/tsne', methods=['GET'])
def get_project_tsne(project_id):
    """Return a project-level t-SNE map with cycle and lab-result overlays."""
    project = _get_session_project_or_404(project_id)

    if not project.dataset_path or not os.path.exists(project.dataset_path):
        return jsonify({'success': False, 'error': 'Dataset file not found for this project.'}), 404

    try:
        df = pd.read_csv(project.dataset_path)
    except Exception:
        logger.exception("Could not read project dataset for results t-SNE")
        return jsonify({'success': False, 'error': 'Could not read the project dataset.'}), 400

    lab_result_columns = _collect_lab_result_columns(project)
    df = _annotate_results_tsne_rows(df, project, lab_result_columns)
    feature_columns = _get_requested_feature_columns(df)

    options = {
        'perplexity': _request_float('perplexity', min(20, max(2, len(df) - 1)), 2, max(2, len(df) - 1)),
        'iterations': _request_int('iterations', 350, 300, 3000),
        'learning_rate': _request_float('learning_rate', 100, 2, 2000),
        'random_seed': _request_int('random_seed', 42, 0, 999999),
        'scaling': request.args.get('scaling', 'standard'),
        'max_points': _request_int('max_points', 3000, 250, 20000)
    }
    if options['scaling'] not in {'standard', 'robust', 'none'}:
        options['scaling'] = 'standard'

    warnings = _build_tsne_quality_warnings(df, feature_columns)
    tsne_input_df, sampling_warning = _downsample_results_tsne_df(df, options['max_points'], options['random_seed'])
    if sampling_warning:
        warnings.append(sampling_warning)
    cache_key = None if request.args.get('refresh') else _build_results_tsne_cache_key(project, feature_columns, options)
    tsne_df = PlotGenerator._run_tsne(
        tsne_input_df,
        feature_columns,
        cache_key=cache_key,
        perplexity=options['perplexity'],
        max_iter=options['iterations'],
        learning_rate=options['learning_rate'],
        random_state=options['random_seed'],
        scaling=options['scaling'],
    )

    payload = _build_results_tsne_payload(tsne_df, feature_columns, lab_result_columns, warnings, options)
    payload['success'] = True
    payload['project'] = project.to_dict()
    return jsonify(payload)


@results_bp.route('/projects/<int:project_id>', methods=['DELETE'])
def delete_project(project_id):
    """Delete a project and all its cycles/samples."""
    project = _get_session_project_or_404(project_id)
    
    db.session.delete(project)
    db.session.commit()
    
    return jsonify({'success': True})


# ============================================================
# Cycle Endpoints
# ============================================================

@results_bp.route('/projects/<int:project_id>/cycles', methods=['GET'])
def get_cycles(project_id):
    """List all cycles for a project."""
    project = _get_session_project_or_404(project_id)
    
    cycles = Cycle.query.filter_by(project_id=project_id)\
                        .order_by(Cycle.cycle_number.desc()).all()
    
    return jsonify({
        'success': True,
        'project': project.to_dict(),
        'cycles': [c.to_dict(include_samples=True) for c in cycles]
    })


@results_bp.route('/cycles', methods=['POST'])
def create_cycle():
    """Create a new cycle with selected samples."""
    data = request.get_json()
    
    project_id = data.get('project_id')
    samples_data = data.get('samples', [])
    
    if not project_id:
        return jsonify({'success': False, 'error': 'project_id is required'}), 400
    
    if not samples_data:
        return jsonify({'success': False, 'error': 'At least one sample is required'}), 400
    
    project = _get_session_project_or_404(project_id)
    
    # Auto-set dataset_path if not already set
    dataset_path = data.get('dataset_path')
    if dataset_path and (not project.dataset_path or project.dataset_path == ''):
        project.dataset_path = resolve_dataset_path(dataset_path, must_exist=False) or project.dataset_path
        db.session.commit()
    
    # Determine the next cycle number
    max_cycle = db.session.query(db.func.max(Cycle.cycle_number))\
                          .filter_by(project_id=project_id).scalar() or 0
    next_cycle_number = max_cycle + 1
    
    # Create the cycle
    cycle = Cycle(
        project_id=project_id,
        cycle_number=next_cycle_number,
        notes=data.get('notes', '')
    )
    
    # Store lab result columns (target + a-priori columns from experiment)
    lab_result_columns = data.get('lab_result_columns', [])
    if lab_result_columns:
        cycle.set_lab_result_columns(lab_result_columns)
    
    db.session.add(cycle)
    db.session.flush()  # Get the cycle ID
    
    # Add samples
    for sample_data in samples_data:
        sample = Sample(
            cycle_id=cycle.id,
            idx_sample=sample_data.get('idx_sample', 0)
        )
        sample.set_row_data(sample_data.get('row_data', {}))
        sample.set_predictions(sample_data.get('predictions', {}))
        sample.status = 'pending'
        db.session.add(sample)
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'cycle': cycle.to_dict(include_samples=True)
    }), 201


# ============================================================
# Sample Endpoints
# ============================================================

@results_bp.route('/samples/<int:sample_id>', methods=['PUT'])
def update_sample(sample_id):
    """Update a sample with lab results."""
    sample = _get_session_sample_or_404(sample_id)
    data = request.get_json()
    
    if 'lab_results' in data:
        sample.set_lab_results(data['lab_results'])
    
    if 'status' in data:
        sample.status = data['status']
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'sample': sample.to_dict()
    })


@results_bp.route('/samples/<int:sample_id>', methods=['DELETE'])
def delete_sample(sample_id):
    """Delete a sample from a cycle."""
    sample = _get_session_sample_or_404(sample_id)
    
    db.session.delete(sample)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': f'Sample {sample_id} deleted'
    })


@results_bp.route('/samples/<int:sample_id>/sync', methods=['POST'])
def sync_sample_to_csv(sample_id):
    """Sync lab results back to the original CSV using IDX_SAMPLE."""
    sample = _get_session_sample_or_404(sample_id)

    try:
        sync_result = _sync_sample_lab_results(sample)
        if not sync_result['success']:
            status_code = 404 if sync_result['error'] == 'Dataset file not found' else 400
            return jsonify(sync_result), status_code

        db.session.commit()

        return jsonify({
            'success': True,
            'message': f"Lab results synced to {sync_result['dataset_path']}",
            'matched_by': sync_result['matched_by'],
            'updated_columns': sync_result['updated_columns']
        })

    except FileNotFoundError:
        return jsonify({'success': False, 'error': 'Dataset file not found.'}), 404
    except Exception as e:
        logger.exception("Error syncing sample to CSV")
        return jsonify({'success': False, 'error': 'Failed to sync results. Please check the file format.'}), 500


@results_bp.route('/samples/sync-batch', methods=['POST'])
def sync_batch_to_csv():
    """Sync multiple samples' lab results to CSV."""
    data = request.get_json()
    sample_ids = data.get('sample_ids', [])
    
    if not sample_ids:
        return jsonify({'success': False, 'error': 'No sample IDs provided'}), 400
    
    synced = 0
    errors = []
    
    for sample_id in sample_ids:
        sample = Sample.query.get(sample_id)
        if not sample:
            errors.append(f"Sample {sample_id} not found")
            continue
        cycle = db.session.get(Cycle, sample.cycle_id)
        project = db.session.get(Project, cycle.project_id) if cycle else None
        if not project or project.session_id != get_session_id():
            errors.append(f"Sample {sample_id} not found")
            continue

        try:
            sync_result = _sync_sample_lab_results(sample)
            if sync_result['success']:
                synced += 1
            elif sync_result['error'] != 'No lab results to sync':
                errors.append(f"Sample {sample_id}: {sync_result['error']}")
        except Exception as e:
            errors.append(f"Error syncing sample {sample_id}: {str(e)}")
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'synced_count': synced,
        'errors': errors if errors else None
    })


# ============================================================
# Plotting Data Endpoint
# ============================================================

@results_bp.route('/plot-data', methods=['GET'])
def get_plot_data():
    """Get data for plotting across cycles."""
    project_id = request.args.get('project_id', type=int)
    columns = request.args.getlist('columns')
    
    if not project_id:
        return jsonify({'success': False, 'error': 'project_id is required'}), 400
    
    project = _get_session_project_or_404(project_id)
    
    plot_data = {
        'cycles': [],
        'columns': columns
    }
    
    for cycle in project.cycles:
        cycle_data = {
            'cycle_number': cycle.cycle_number,
            'samples': []
        }
        
        for sample in cycle.samples:
            sample_values = {}
            
            # Merge row_data, predictions, and lab_results
            all_data = {
                **sample.get_row_data(),
                **sample.get_predictions(),
                **sample.get_lab_results()
            }
            
            # Extract requested columns
            for col in columns:
                if col in all_data:
                    sample_values[col] = all_data[col]
            
            sample_values['idx_sample'] = sample.idx_sample
            sample_values['status'] = sample.status
            cycle_data['samples'].append(sample_values)
        
        plot_data['cycles'].append(cycle_data)
    
    return jsonify({
        'success': True,
        'data': plot_data
    })
