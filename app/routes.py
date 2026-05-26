"""
Core Routes

This module contains only the core view routes (index, dashboard).
All API endpoints have been moved to the app/api/ package.
"""
import os
import json
import logging
from flask import Blueprint, render_template, session, send_from_directory, abort
from werkzeug.utils import secure_filename
from app.utils.session_store import resolve_dataset_path

logging.basicConfig(level=logging.DEBUG)

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@main_bp.route('/dashboard')
def dashboard():
    """Dashboard page for active learning experiments."""
    initial_data = {
        'data_columns': session.get('data_columns', []),
        'filename': session.get('filename', None)
    }
    return render_template('dashboard.html', initial_data=json.dumps(initial_data))


@main_bp.route('/results')
def results():
    """Results page for tracking lab testing cycles."""
    return render_template('results.html')


@main_bp.route('/data/<filename>')
def download_data_file(filename):
    """Download a dataset file from the data directory."""
    filename = secure_filename(filename)
    filepath = resolve_dataset_path(filename)
    if not filepath:
        abort(404)
    data_dir = os.path.dirname(filepath)
    
    # Verify path is within allowed directory
    if not os.path.abspath(filepath).startswith(os.path.abspath(data_dir)):
        abort(403)
    
    return send_from_directory(data_dir, filename, as_attachment=True)
