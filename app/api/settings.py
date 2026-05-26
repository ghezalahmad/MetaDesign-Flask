"""
Settings API Routes

Handles user settings management.
"""
import logging
from flask import Blueprint, request, jsonify
from app.utils.settings_manager import SettingsManager

settings_bp = Blueprint('settings', __name__)
logger = logging.getLogger(__name__)


@settings_bp.route('/api/settings', methods=['GET'])
def get_settings():
    """Return current settings for the dashboard."""
    settings = SettingsManager.strip_sensitive(SettingsManager.load_settings())
    return jsonify({'success': True, 'settings': settings})


@settings_bp.route('/api/settings', methods=['POST'])
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
        logger.error(f"Error saving settings: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
