"""
Uploads API Routes

Handles file upload functionality.
"""
import os
import logging
import shutil
import pandas as pd
from flask import Blueprint, request, jsonify, session
from werkzeug.utils import secure_filename
from app.utils.settings_manager import SettingsManager
from app.utils import session_store
from app.utils.session_store import get_session_upload_dir

uploads_bp = Blueprint('uploads', __name__)
logger = logging.getLogger(__name__)

# Allowed file extensions for dataset uploads
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
DEFAULT_DEMO_DATASET = 'metadesign_demo_cement.csv'


def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _examples_dir():
    return session_store.DATA_DIR / 'examples'


def _read_dataset_columns(filepath, filename):
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == 'csv':
        data = pd.read_csv(filepath)
    else:
        data = pd.read_excel(filepath)
    return data.columns.tolist()


def _activate_dataset(filepath, filename, columns):
    session['filepath'] = filepath
    session['data_columns'] = columns
    session['filename'] = filename
    SettingsManager.save_settings({
        'current_dataset': filename,
        'current_dataset_columns': columns
    })


@uploads_bp.route('/upload', methods=['POST'])
def upload_data():
    """Upload a dataset file."""
    if 'dataset' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})
    file = request.files['dataset']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})

    # Validate file type
    if not allowed_file(file.filename):
        return jsonify({
            'success': False, 
            'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        })

    filename = secure_filename(file.filename)
    upload_folder = get_session_upload_dir(create=True)

    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)
    session['filepath'] = filepath

    try:
        columns = _read_dataset_columns(filepath, filename)
        _activate_dataset(filepath, filename, columns)
        
        return jsonify({'success': True, 'columns': columns, 'filename': filename})
    except Exception as e:
        # Clean up invalid file
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'success': False, 'error': f"Failed to read file: {str(e)}"})


@uploads_bp.route('/api/demo-datasets/load', methods=['POST'])
def load_demo_dataset():
    """Copy a curated demo dataset into the current session."""
    payload = request.get_json(silent=True) or {}
    filename = secure_filename(payload.get('filename') or DEFAULT_DEMO_DATASET)
    if not allowed_file(filename):
        return jsonify({'success': False, 'error': 'Invalid demo dataset type.'}), 400

    example_path = (_examples_dir() / filename).resolve()
    examples_root = _examples_dir().resolve()
    try:
        example_path.relative_to(examples_root)
    except ValueError:
        return jsonify({'success': False, 'error': 'Demo dataset not found.'}), 404
    if not example_path.exists():
        return jsonify({'success': False, 'error': 'Demo dataset not found.'}), 404

    upload_folder = get_session_upload_dir(create=True)
    session_path = os.path.join(upload_folder, filename)
    shutil.copyfile(example_path, session_path)

    try:
        columns = _read_dataset_columns(session_path, filename)
        _activate_dataset(session_path, filename, columns)
        return jsonify({
            'success': True,
            'filename': filename,
            'columns': columns,
            'message': 'Demo dataset loaded into this session.'
        })
    except Exception as e:
        if os.path.exists(session_path):
            os.remove(session_path)
        return jsonify({'success': False, 'error': f"Failed to load demo dataset: {str(e)}"}), 500
