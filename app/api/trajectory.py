"""
Trajectory API Routes

Handles experiment trajectory tracking for visualization.
"""
import logging
from flask import Blueprint, jsonify
from app.utils.trajectory_tracker import TrajectoryTracker

trajectory_bp = Blueprint('trajectory', __name__)
logger = logging.getLogger(__name__)


@trajectory_bp.route('/api/trajectory', methods=['GET'])
def get_trajectory():
    """Return current trajectory data for visualization."""
    try:
        summary = TrajectoryTracker.get_trajectory_summary()
        return jsonify({'success': True, 'trajectory': summary})
    except Exception as e:
        logger.error(f"Error getting trajectory: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@trajectory_bp.route('/api/trajectory', methods=['DELETE'])
def clear_trajectory():
    """Clear trajectory history for new experiment run."""
    try:
        TrajectoryTracker.clear()
        return jsonify({'success': True, 'message': 'Trajectory cleared.'})
    except Exception as e:
        logger.error(f"Error clearing trajectory: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
