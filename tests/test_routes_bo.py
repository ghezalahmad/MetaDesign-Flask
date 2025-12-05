
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from app.routes import main_bp
from flask import Flask

class TestRoutesBO(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.register_blueprint(main_bp)
        self.app.secret_key = 'test_secret_key' # Required for session
        self.client = self.app.test_client()
        self.app.config['TESTING'] = True

    @patch('app.routes.pd.read_csv')
    @patch('app.routes.os.path.exists')
    def test_run_experiment_utility_calculation(self, mock_exists, mock_read_csv):
        # Mock session and file existence
        mock_exists.return_value = True

        # Mock Data: 5 samples, 2 labeled (train), 3 unlabeled (candidates)
        data = pd.DataFrame({
            'x1': [0.1, 0.2, 0.8, 0.9, 0.5],
            'y1': [1.0, 2.0, np.nan, np.nan, np.nan], # Targets
            'prediction': [1.0, 2.0, 3.0, 4.0, 2.5], # Mock predictions
            'uncertainty': [0.1, 0.1, 0.5, 0.6, 0.2] # Mock uncertainties
        })
        mock_read_csv.return_value = data

        # Mock Request Payload
        payload = {
            'model': 'rf', # Use RF so it triggers the fallback or standard flow
            'curiosity': 1.0,
            'input_columns': ['x1'],
            'target_columns': [{'name': 'y1', 'weight': 1.0, 'optimization': 'max'}]
        }
        
        # Set session
        with self.client.session_transaction() as sess:
            sess['filepath'] = '/fake/path.csv'
            
        with patch('app.routes.train_rf_model') as mock_train, \
             patch('app.routes.evaluate_rf_model') as mock_eval:
             
             mock_train.return_value = (MagicMock(), MagicMock(), MagicMock())
             
             # Return data as results_df, but ensure no Utility column
             results_df = data.copy()
             if 'Utility' in results_df.columns:
                 del results_df['Utility']
             mock_eval.return_value = results_df

             response = self.client.post('/run-experiment', json=payload)
             
             self.assertEqual(response.status_code, 200)
             self.assertTrue(response.is_json)

    @patch('app.routes.calculate_utility')
    @patch('app.routes.calculate_novelty')
    @patch('app.routes.pd.read_csv')
    @patch('app.routes.os.path.exists')
    def test_run_experiment_calls_utils(self, mock_exists, mock_read_csv, mock_novelty, mock_utility):
        mock_exists.return_value = True
        
        data = pd.DataFrame({
            'x1': [0.1, 0.2, 0.8],
            'y1': [1.0, np.nan, np.nan],
            'prediction': [1.0, 2.0, 3.0],
            'uncertainty': [0.1, 0.5, 0.6]
        })
        mock_read_csv.return_value = data
        
        # Mock novelty and utility returns
        mock_novelty.return_value = np.array([0.1, 0.5, 0.9])
        mock_utility.return_value = np.array([[0.1], [0.5], [0.9]])

        payload = {
            'model': 'rf',
            'curiosity': 0.5,
            'input_columns': ['x1'],
            'target_columns': [{'name': 'y1', 'weight': 1.0, 'optimization': 'max'}]
        }

        with self.client.session_transaction() as sess:
            sess['filepath'] = '/fake/path.csv'

        with patch('app.routes.train_rf_model') as mock_train, \
             patch('app.routes.evaluate_rf_model') as mock_eval, \
             patch('app.routes.PlotGenerator'): # Mock plotting to avoid errors
            
            mock_train.return_value = (MagicMock(), MagicMock(), MagicMock())
            mock_eval.return_value = data.copy() # Return DF without Utility

            self.client.post('/run-experiment', json=payload)

            # Assert Novelty was calculated
            mock_novelty.assert_called()
            
            # Assert Utility was calculated with correct curiosity
            mock_utility.assert_called()
            call_args = mock_utility.call_args
            self.assertEqual(call_args.kwargs['curiosity'], 0.5)
            self.assertTrue('novelty' in call_args.kwargs)

if __name__ == '__main__':
    unittest.main()
