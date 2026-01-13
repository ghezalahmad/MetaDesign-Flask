
import unittest
import numpy as np
import pandas as pd

from app.models.bayesian_optimizer import BayesianOptimizer, multi_objective_bayesian_optimization
from app.models.rf_model import RFModel

class TestBayesianOptimizer(unittest.TestCase):

    def setUp(self):
        self.X_train = pd.DataFrame(np.random.rand(10, 2), columns=['x1', 'x2'])
        self.y_train = np.random.rand(10, 1)
        self.bounds = {'x1': (0, 1), 'x2': (0, 1)}

    def test_bayesian_optimizer_fit_predict(self):
        optimizer = BayesianOptimizer(bounds=self.bounds)
        optimizer.fit(self.X_train, self.y_train)
        X_test = np.array([[0.5, 0.5]])
        mu, sigma, _ = optimizer._get_surrogate_prediction(X_test)
        self.assertEqual(mu.shape, (1, 1))  # Now returns (N, 1) shaped arrays
        self.assertEqual(sigma.shape, (1, 1))

    def test_multi_objective_bayesian_optimization_with_surrogate(self):
        model = RFModel(n_estimators=10)
        model.train(pd.concat([self.X_train, pd.DataFrame(self.y_train, columns=['y1'])], axis=1),
                    input_columns=['x1', 'x2'],
                    target_columns=['y1'])

        candidate_inputs = pd.DataFrame(np.random.rand(5, 2), columns=['x1', 'x2'])

        scores = multi_objective_bayesian_optimization(
            train_inputs=self.X_train,
            train_targets=self.y_train,
            candidate_inputs=candidate_inputs,
            weights=np.array([1.0]),
            max_or_min=['max'],
            surrogate_model=model,
            input_columns=['x1', 'x2']
        )
        self.assertEqual(scores.shape, (5,))

if __name__ == '__main__':
    unittest.main()
