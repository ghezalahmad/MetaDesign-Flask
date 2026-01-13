"""
Uploads API Routes

Handles file upload functionality.
"""
import os
import logging
import pandas as pd
from flask import Blueprint, request, jsonify, session
from werkzeug.utils import secure_filename
from app.utils.settings_manager import SettingsManager

uploads_bp = Blueprint('uploads', __name__)
logger = logging.getLogger(__name__)

# Allowed file extensions for dataset uploads
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}


def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
    upload_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    os.makedirs(upload_folder, exist_ok=True)

    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)
    session['filepath'] = filepath

    try:
        # Read file based on extension
        ext = filename.rsplit('.', 1)[1].lower()
        if ext == 'csv':
            data = pd.read_csv(filepath)
        else:  # xlsx or xls
            data = pd.read_excel(filepath)
        
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
        # Clean up invalid file
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'success': False, 'error': f"Failed to read file: {str(e)}"})

