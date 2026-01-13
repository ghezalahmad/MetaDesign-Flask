"""
Experiments API Routes

Handles experiment logging and history retrieval.
"""
import logging
from flask import Blueprint, jsonify

experiments_bp = Blueprint('experiments', __name__)
logger = logging.getLogger(__name__)


@experiments_bp.route('/api/experiments', methods=['GET'])
def get_experiments():
    """Return list of all logged experiments."""
    try:
        from app.tracking import get_experiment_logger
        exp_logger = get_experiment_logger()
        experiments = exp_logger.list_experiments()
        return jsonify({'success': True, 'experiments': experiments})
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@experiments_bp.route('/api/experiments/<exp_id>', methods=['GET'])
def get_experiment_detail(exp_id):
    """Return details of a specific experiment."""
    try:
        from app.tracking import get_experiment_logger
        exp_logger = get_experiment_logger()
        experiment = exp_logger.get_experiment(exp_id)
        if experiment:
            return jsonify({'success': True, 'experiment': experiment})
        else:
            return jsonify({'success': False, 'error': 'Experiment not found'}), 404
    except Exception as e:
        logger.error(f"Error getting experiment {exp_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@experiments_bp.route('/api/experiments/<exp_id>', methods=['DELETE'])
def delete_experiment(exp_id):
    """Delete a specific experiment."""
    try:
        from app.tracking import get_experiment_logger
        exp_logger = get_experiment_logger()
        success = exp_logger.delete_experiment(exp_id)
        if success:
            logger.info(f"Deleted experiment: {exp_id}")
            return jsonify({'success': True, 'message': f'Experiment {exp_id} deleted'})
        else:
            return jsonify({'success': False, 'error': 'Experiment not found'}), 404
    except Exception as e:
        logger.error(f"Error deleting experiment {exp_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
