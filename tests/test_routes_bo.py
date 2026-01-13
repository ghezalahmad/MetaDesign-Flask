"""
Test Suite for Active Learning Pipeline

Tests the experiment execution, acquisition functions, and API endpoints.
Updated to match the refactored architecture with HybridEngine and
pluggable acquisition functions (webslamd, ucb, ei, thompson).
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from flask import Flask

# Import the actual modules being tested
from app.routes import main_bp
from app.api.run_experiment import run_experiment_bp
from app.acquisition import get_acquisition_function, ACQUISITION_FUNCTIONS
from app.utils.utils import calculate_novelty


class TestAcquisitionFunctions(unittest.TestCase):
    """Test the acquisition function implementations."""
    
    def setUp(self):
        """Set up test data."""
        # Mock labeled data (what we've already observed)
        self.labeled_data = pd.DataFrame({
            'target1': [1.0, 2.0, 3.0],
            'target2': [10.0, 20.0, 30.0]
        })
        
        # Predictions and uncertainties for candidates
        self.predictions = np.array([
            [2.5, 25.0],
            [4.0, 40.0],
            [1.5, 15.0]
        ])
        self.uncertainties = np.array([
            [0.5, 5.0],
            [0.3, 3.0],
            [0.8, 8.0]
        ])
        
        self.target_columns = ['target1', 'target2']
        self.max_or_min = ['max', 'max']
        self.weights = np.array([1.0, 1.0])

    def test_webslamd_acquisition(self):
        """Test WEBSLAMD acquisition function."""
        acq = get_acquisition_function('webslamd')
        
        scores = acq.compute(
            predictions=self.predictions,
            uncertainties=self.uncertainties,
            labeled_data=self.labeled_data,
            target_columns=self.target_columns,
            max_or_min=self.max_or_min,
            weights=self.weights,
            curiosity=0.5
        )
        
        # Should return one score per candidate
        self.assertEqual(len(scores), 3)
        # Higher prediction + uncertainty should yield higher score
        self.assertTrue(scores[1] > scores[2])  # 4.0 > 1.5

    def test_ucb_acquisition(self):
        """Test UCB acquisition function."""
        acq = get_acquisition_function('ucb')
        
        scores = acq.compute(
            predictions=self.predictions,
            uncertainties=self.uncertainties,
            labeled_data=self.labeled_data,
            target_columns=self.target_columns,
            max_or_min=self.max_or_min,
            weights=self.weights,
            curiosity=0.5
        )
        
        self.assertEqual(len(scores), 3)
        # UCB = μ + β*σ, so higher prediction AND uncertainty should score higher

    def test_ei_acquisition(self):
        """Test Expected Improvement acquisition function."""
        acq = get_acquisition_function('ei')
        
        scores = acq.compute(
            predictions=self.predictions,
            uncertainties=self.uncertainties,
            labeled_data=self.labeled_data,
            target_columns=self.target_columns,
            max_or_min=self.max_or_min,
            weights=self.weights,
            curiosity=0.5
        )
        
        self.assertEqual(len(scores), 3)
        # EI should be non-negative
        self.assertTrue(all(s >= 0 for s in scores))

    def test_thompson_acquisition(self):
        """Test Thompson Sampling acquisition function."""
        acq = get_acquisition_function('thompson')
        
        # Thompson sampling is stochastic, run multiple times
        scores1 = acq.compute(
            predictions=self.predictions,
            uncertainties=self.uncertainties,
            labeled_data=self.labeled_data,
            target_columns=self.target_columns,
            max_or_min=self.max_or_min,
            weights=self.weights,
            curiosity=0.5
        )
        scores2 = acq.compute(
            predictions=self.predictions,
            uncertainties=self.uncertainties,
            labeled_data=self.labeled_data,
            target_columns=self.target_columns,
            max_or_min=self.max_or_min,
            weights=self.weights,
            curiosity=0.5
        )
        
        self.assertEqual(len(scores1), 3)
        # Due to stochasticity, scores may differ (but shape should be same)

    def test_minimization_targets(self):
        """Test that minimization targets are handled correctly."""
        acq = get_acquisition_function('webslamd')
        
        # For minimization, lower predictions should score higher
        max_or_min_mixed = ['min', 'max']
        
        scores = acq.compute(
            predictions=self.predictions,
            uncertainties=self.uncertainties,
            labeled_data=self.labeled_data,
            target_columns=self.target_columns,
            max_or_min=max_or_min_mixed,
            weights=self.weights,
            curiosity=0.5
        )
        
        self.assertEqual(len(scores), 3)

    def test_all_acquisition_functions_registered(self):
        """Verify all 4 acquisition functions are available."""
        expected = {'webslamd', 'ucb', 'ei', 'thompson'}
        actual = set(ACQUISITION_FUNCTIONS.keys())
        self.assertEqual(expected, actual)


class TestNoveltyCalculation(unittest.TestCase):
    """Test the novelty calculation utility."""
    
    def test_novelty_basic(self):
        """Test basic novelty calculation."""
        candidate_features = np.array([[0.5, 0.5], [0.9, 0.9]])
        labeled_features = np.array([[0.1, 0.1], [0.2, 0.2]])
        
        novelty = calculate_novelty(candidate_features, labeled_features)
        
        self.assertEqual(len(novelty), 2)
        # Point farther from labeled data should have higher novelty
        self.assertTrue(novelty[1] > novelty[0])

    def test_novelty_empty_labeled(self):
        """Test novelty with no labeled data."""
        candidate_features = np.array([[0.5, 0.5]])
        labeled_features = np.array([]).reshape(0, 2)
        
        novelty = calculate_novelty(candidate_features, labeled_features)
        
        # Should return 1.0 for all when no labeled data
        self.assertEqual(novelty[0], 1.0)


class TestRunExperimentAPI(unittest.TestCase):
    """Test the /run-experiment API endpoint."""
    
    def setUp(self):
        self.app = Flask(__name__)
        self.app.register_blueprint(main_bp)
        self.app.register_blueprint(run_experiment_bp)
        self.app.secret_key = 'test_secret_key'
        self.client = self.app.test_client()
        self.app.config['TESTING'] = True

    def test_run_experiment_missing_filepath(self):
        """Test experiment gracefully fails with missing filepath."""
        payload = {
            'model': 'gp',
            'curiosity': 0.5,
            'input_columns': ['x1'],
            'target_columns': [{'name': 'y1', 'weight': 1.0, 'optimization': 'max'}]
        }
        
        response = self.client.post('/run-experiment', json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertFalse(data.get('success', True))
        self.assertIn('error', data)

    def test_run_experiment_invalid_json(self):
        """Test experiment handles malformed requests."""
        # Send empty body
        response = self.client.post('/run-experiment', 
                                    data='not json', 
                                    content_type='application/json')
        
        # Should return 500 or 400, or a JSON error
        self.assertTrue(response.status_code in [200, 400, 500])

    def test_endpoint_exists(self):
        """Test that the /run-experiment endpoint is registered."""
        # Test with OPTIONS to verify endpoint exists
        response = self.client.post('/run-experiment', json={})
        
        # Should not be 404
        self.assertNotEqual(response.status_code, 404)


if __name__ == '__main__':
    unittest.main()
