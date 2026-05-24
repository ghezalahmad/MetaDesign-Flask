"""
Results API Routes

Handles project management, cycle creation, sample updates, and CSV synchronization
for the active learning lab results workflow.
"""

import os
import json
import logging
import pandas as pd
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from app.database import db, Project, Cycle, Sample

results_bp = Blueprint('results', __name__, url_prefix='/api/results')
logger = logging.getLogger(__name__)

IDENTITY_COLUMNS = ['Idx_Sample', 'IDX_SAMPLE', 'idx_sample', 'IdxSample', 'Row number', 'row_number', 'Index', 'index']


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


# ============================================================
# Project Endpoints
# ============================================================

@results_bp.route('/projects', methods=['GET'])
def get_projects():
    """List all projects."""
    projects = Project.query.order_by(Project.created_at.desc()).all()
    return jsonify({
        'success': True,
        'projects': [p.to_dict() for p in projects]
    })


@results_bp.route('/projects', methods=['POST'])
def create_project():
    """Create a new project."""
    data = request.get_json()
    
    name = data.get('name')
    dataset_path = data.get('dataset_path')
    
    if not name or not dataset_path:
        return jsonify({'success': False, 'error': 'Name and dataset_path are required'}), 400
    
    # Check if project with same name exists
    existing = Project.query.filter_by(name=name).first()
    if existing:
        return jsonify({'success': False, 'error': f'Project "{name}" already exists'}), 400
    
    project = Project(name=name, dataset_path=dataset_path)
    db.session.add(project)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'project': project.to_dict()
    }), 201


@results_bp.route('/projects/<int:project_id>', methods=['GET'])
def get_project(project_id):
    """Get a project with its cycles."""
    project = Project.query.get_or_404(project_id)
    
    data = project.to_dict()
    data['cycles'] = [c.to_dict(include_samples=True) for c in project.cycles]
    
    return jsonify({
        'success': True,
        'project': data
    })


@results_bp.route('/projects/<int:project_id>', methods=['DELETE'])
def delete_project(project_id):
    """Delete a project and all its cycles/samples."""
    project = Project.query.get_or_404(project_id)
    
    db.session.delete(project)
    db.session.commit()
    
    return jsonify({'success': True})


# ============================================================
# Cycle Endpoints
# ============================================================

@results_bp.route('/projects/<int:project_id>/cycles', methods=['GET'])
def get_cycles(project_id):
    """List all cycles for a project."""
    project = Project.query.get_or_404(project_id)
    
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
    
    project = Project.query.get_or_404(project_id)
    
    # Auto-set dataset_path if not already set
    dataset_path = data.get('dataset_path')
    if dataset_path and (not project.dataset_path or project.dataset_path == ''):
        project.dataset_path = dataset_path
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
    sample = Sample.query.get_or_404(sample_id)
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
    sample = Sample.query.get_or_404(sample_id)
    
    db.session.delete(sample)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': f'Sample {sample_id} deleted'
    })


@results_bp.route('/samples/<int:sample_id>/sync', methods=['POST'])
def sync_sample_to_csv(sample_id):
    """Sync lab results back to the original CSV using IDX_SAMPLE."""
    sample = Sample.query.get_or_404(sample_id)

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
    
    project = Project.query.get_or_404(project_id)
    
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
