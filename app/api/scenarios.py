"""
Scenarios API Routes

Handles project management with scenarios for experimental planning.
"""

import os
import glob
from flask import Blueprint, request, jsonify
from ..database import db, Project, Scenario, Cycle

scenarios_bp = Blueprint('scenarios', __name__, url_prefix='/api/scenarios')


# ============================================================
# Dataset Discovery Endpoint
# ============================================================

@scenarios_bp.route('/datasets', methods=['GET'])
def list_available_datasets():
    """List all available datasets from design spaces and uploaded files."""
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    designspaces_dir = os.path.join(data_dir, 'designspaces')
    
    datasets = []
    
    # 1. Get design space files
    if os.path.exists(designspaces_dir):
        for filepath in glob.glob(os.path.join(designspaces_dir, '*.csv')):
            filename = os.path.basename(filepath)
            # Get file size
            size_bytes = os.path.getsize(filepath)
            size_kb = round(size_bytes / 1024, 1)
            
            datasets.append({
                'name': filename,
                'path': os.path.abspath(filepath),
                'source': 'Design Space',
                'size_kb': size_kb,
                'icon': 'bi-grid-3x3-gap'
            })
    
    # 2. Get uploaded datasets from data/ directory (excluding design spaces)
    if os.path.exists(data_dir):
        for ext in ['*.csv', '*.xlsx', '*.xls']:
            for filepath in glob.glob(os.path.join(data_dir, ext)):
                filename = os.path.basename(filepath)
                # Skip non-dataset files
                if filename in ['scenarios.csv', 'trajectory_history.json']:
                    continue
                
                size_bytes = os.path.getsize(filepath)
                size_kb = round(size_bytes / 1024, 1)
                
                datasets.append({
                    'name': filename,
                    'path': os.path.abspath(filepath),
                    'source': 'Uploaded',
                    'size_kb': size_kb,
                    'icon': 'bi-file-earmark-spreadsheet'
                })
    
    # Sort by source (Design Space first), then by name
    datasets.sort(key=lambda x: (0 if x['source'] == 'Design Space' else 1, x['name']))
    
    return jsonify({
        'success': True,
        'datasets': datasets
    })


@scenarios_bp.route('/dataset-stats', methods=['POST'])
def get_dataset_stats():
    """Get statistics about a dataset for coverage calculation.
    
    Returns:
    - total_rows: Total number of rows in the dataset
    - labeled_count: Number of rows with at least one target column filled
    - target_columns: List of detected target columns (columns with >50% NaN)
    """
    import pandas as pd
    
    data = request.get_json()
    dataset_path = data.get('dataset_path', '')
    
    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({
            'success': False, 
            'error': 'Dataset not found'
        }), 404
    
    try:
        df = pd.read_csv(dataset_path)
        total_rows = len(df)
        
        # Detect target columns (numeric columns with >50% NaN values)
        target_columns = []
        labeled_count = 0
        
        for col in df.columns:
            # Skip index columns
            if col.lower() in ['idx_sample', 'index', 'row number', 'id']:
                continue
            
            # Check if column is numeric
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            nan_ratio = df[col].isna().sum() / total_rows
            
            # If more than 50% NaN, it's likely a target column
            if nan_ratio > 0.5:
                filled_count = total_rows - df[col].isna().sum()
                target_columns.append({
                    'name': col,
                    'nan_count': int(df[col].isna().sum()),
                    'filled_count': int(filled_count),
                    'nan_ratio': round(nan_ratio * 100, 1)
                })
        
        # Calculate labeled count as the max filled count across target columns
        if target_columns:
            labeled_count = max(tc['filled_count'] for tc in target_columns)
        else:
            # If no target columns detected, assume all rows are unlabeled
            labeled_count = 0
        
        return jsonify({
            'success': True,
            'total_rows': total_rows,
            'labeled_count': labeled_count,
            'target_columns': target_columns,
            'unlabeled_count': total_rows - labeled_count
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error reading dataset: {str(e)}'
        }), 500


# ============================================================
# Project Endpoints (with Scenarios)
# ============================================================

@scenarios_bp.route('/projects', methods=['GET'])
def get_projects():
    """List all projects with their scenarios."""
    projects = Project.query.order_by(Project.created_at.desc()).all()
    
    result = []
    for p in projects:
        data = p.to_dict(include_scenarios=True)
        data['progress'] = p.get_progress()
        result.append(data)
    
    return jsonify({
        'success': True,
        'projects': result
    })


@scenarios_bp.route('/projects', methods=['POST'])
def create_project():
    """Create a new project."""
    data = request.get_json()
    name = data.get('name', '').strip()
    dataset_path = data.get('dataset_path', '').strip()
    
    if not name:
        return jsonify({'success': False, 'error': 'Project name is required'}), 400
    
    # Check if project with same name exists
    existing = Project.query.filter_by(name=name).first()
    if existing:
        return jsonify({'success': False, 'error': 'Project with this name already exists'}), 400
    
    project = Project(name=name, dataset_path=dataset_path)
    db.session.add(project)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'project': project.to_dict(include_scenarios=True)
    })


@scenarios_bp.route('/projects/<int:project_id>', methods=['GET'])
def get_project(project_id):
    """Get a project with its scenarios and progress."""
    project = Project.query.get_or_404(project_id)
    
    data = project.to_dict(include_scenarios=True)
    data['progress'] = project.get_progress()
    data['cycles'] = [c.to_dict() for c in project.cycles]
    
    return jsonify({
        'success': True,
        'project': data
    })


@scenarios_bp.route('/projects/<int:project_id>', methods=['PUT'])
def update_project(project_id):
    """Update a project."""
    project = Project.query.get_or_404(project_id)
    data = request.get_json()
    
    if 'name' in data:
        project.name = data['name'].strip()
    if 'dataset_path' in data:
        project.dataset_path = data['dataset_path'].strip()
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'project': project.to_dict(include_scenarios=True)
    })


@scenarios_bp.route('/projects/<int:project_id>', methods=['DELETE'])
def delete_project(project_id):
    """Delete a project and all its scenarios/cycles."""
    project = Project.query.get_or_404(project_id)
    
    db.session.delete(project)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Project deleted'})


# ============================================================
# Scenario Endpoints
# ============================================================

@scenarios_bp.route('/projects/<int:project_id>/scenarios', methods=['GET'])
def get_scenarios(project_id):
    """Get all scenarios for a project."""
    project = Project.query.get_or_404(project_id)
    
    scenarios = Scenario.query.filter_by(project_id=project_id).all()
    
    return jsonify({
        'success': True,
        'project': project.to_dict(),
        'scenarios': [s.to_dict() for s in scenarios],
        'active_scenario_id': project.active_scenario_id
    })


@scenarios_bp.route('/projects/<int:project_id>/scenarios', methods=['POST'])
def create_scenario(project_id):
    """Create a new scenario for a project."""
    project = Project.query.get_or_404(project_id)
    data = request.get_json()
    
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'success': False, 'error': 'Scenario name is required'}), 400
    
    scenario = Scenario(
        project_id=project_id,
        name=name,
        planned_cycles=data.get('planned_cycles', 2),
        samples_per_cycle=data.get('samples_per_cycle', 5),
        initial_samples=data.get('initial_samples', 10),
        duration_per_cycle_days=data.get('duration_per_cycle_days', 30),
        cost_per_sample=data.get('cost_per_sample', 100.0),
        target_coverage=data.get('target_coverage', 10.0),
        notes=data.get('notes', '')
    )
    
    db.session.add(scenario)
    db.session.commit()
    
    # If this is the first scenario, make it active
    if project.active_scenario_id is None:
        project.active_scenario_id = scenario.id
        db.session.commit()
    
    return jsonify({
        'success': True,
        'scenario': scenario.to_dict()
    })


@scenarios_bp.route('/<int:scenario_id>', methods=['GET'])
def get_scenario(scenario_id):
    """Get a single scenario."""
    scenario = Scenario.query.get_or_404(scenario_id)
    
    return jsonify({
        'success': True,
        'scenario': scenario.to_dict()
    })


@scenarios_bp.route('/<int:scenario_id>', methods=['PUT'])
def update_scenario(scenario_id):
    """Update a scenario."""
    scenario = Scenario.query.get_or_404(scenario_id)
    data = request.get_json()
    
    if 'name' in data:
        scenario.name = data['name'].strip()
    if 'planned_cycles' in data:
        scenario.planned_cycles = int(data['planned_cycles'])
    if 'samples_per_cycle' in data:
        scenario.samples_per_cycle = int(data['samples_per_cycle'])
    if 'initial_samples' in data:
        scenario.initial_samples = int(data['initial_samples'])
    if 'duration_per_cycle_days' in data:
        scenario.duration_per_cycle_days = int(data['duration_per_cycle_days'])
    if 'cost_per_sample' in data:
        scenario.cost_per_sample = float(data['cost_per_sample'])
    if 'target_coverage' in data:
        scenario.target_coverage = float(data['target_coverage'])
    if 'notes' in data:
        scenario.notes = data['notes']
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'scenario': scenario.to_dict()
    })


@scenarios_bp.route('/<int:scenario_id>', methods=['DELETE'])
def delete_scenario(scenario_id):
    """Delete a scenario."""
    scenario = Scenario.query.get_or_404(scenario_id)
    project = Project.query.get(scenario.project_id)
    
    # If this was the active scenario, clear it
    if project and project.active_scenario_id == scenario_id:
        project.active_scenario_id = None
    
    db.session.delete(scenario)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Scenario deleted'})


@scenarios_bp.route('/<int:scenario_id>/activate', methods=['POST'])
def activate_scenario(scenario_id):
    """Set a scenario as the active scenario for its project."""
    scenario = Scenario.query.get_or_404(scenario_id)
    project = Project.query.get_or_404(scenario.project_id)
    
    project.active_scenario_id = scenario_id
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': f'Scenario "{scenario.name}" is now active',
        'project': project.to_dict(include_scenarios=True)
    })


@scenarios_bp.route('/<int:scenario_id>/deactivate', methods=['POST'])
def deactivate_scenario(scenario_id):
    """Deactivate a scenario."""
    scenario = Scenario.query.get_or_404(scenario_id)
    project = Project.query.get_or_404(scenario.project_id)
    
    if project.active_scenario_id == scenario_id:
        project.active_scenario_id = None
        db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'Scenario deactivated',
        'project': project.to_dict(include_scenarios=True)
    })


# ============================================================
# Progress Endpoint
# ============================================================

@scenarios_bp.route('/projects/<int:project_id>/progress', methods=['GET'])
def get_project_progress(project_id):
    """Get real-time progress for a project against its active scenario."""
    project = Project.query.get_or_404(project_id)
    
    progress = project.get_progress()
    
    if not progress:
        return jsonify({
            'success': True,
            'has_active_scenario': False,
            'message': 'No active scenario selected for this project'
        })
    
    active_scenario = Scenario.query.get(project.active_scenario_id)
    
    return jsonify({
        'success': True,
        'has_active_scenario': True,
        'scenario': active_scenario.to_dict() if active_scenario else None,
        'progress': progress
    })
