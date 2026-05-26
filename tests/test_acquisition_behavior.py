import numpy as np
import pandas as pd
from unittest.mock import patch

from app.acquisition import UCB, WEBSLAMD
from app.engines.hybrid_engine import HybridEngine
from app.engines.ml_engine import MLEngine
from app.models.gp_model import evaluate_gp_model


def test_exploit_mode_uses_prediction_without_uncertainty_bonus():
    labeled = pd.DataFrame({"target": [0.0, 1.0]})
    predictions = np.array([[2.0], [1.5]])
    uncertainties = np.array([[0.0], [100.0]])

    scores = UCB().compute(
        predictions=predictions,
        uncertainties=uncertainties,
        labeled_data=labeled,
        target_columns=["target"],
        max_or_min=["max"],
        weights=np.array([1.0]),
        curiosity=0.0,
    )

    assert scores[0] > scores[1]


def test_explore_mode_can_prioritize_high_uncertainty_candidates():
    labeled = pd.DataFrame({"target": [0.0, 1.0]})
    predictions = np.array([[2.0], [1.5]])
    uncertainties = np.array([[0.0], [100.0]])

    scores = UCB().compute(
        predictions=predictions,
        uncertainties=uncertainties,
        labeled_data=labeled,
        target_columns=["target"],
        max_or_min=["max"],
        weights=np.array([1.0]),
        curiosity=1.0,
    )

    assert scores[1] > scores[0]


def test_negative_curiosity_is_treated_as_pure_exploitation():
    labeled = pd.DataFrame({"target": [0.0, 1.0]})
    predictions = np.array([[2.0], [1.5]])
    uncertainties = np.array([[0.0], [100.0]])

    exploit = WEBSLAMD().compute(
        predictions=predictions,
        uncertainties=uncertainties,
        labeled_data=labeled,
        target_columns=["target"],
        max_or_min=["max"],
        weights=np.array([1.0]),
        curiosity=0.0,
    )
    negative = WEBSLAMD().compute(
        predictions=predictions,
        uncertainties=uncertainties,
        labeled_data=labeled,
        target_columns=["target"],
        max_or_min=["max"],
        weights=np.array([1.0]),
        curiosity=-2.0,
    )

    np.testing.assert_allclose(negative, exploit)


class _FakeGP:
    is_trained = True

    def predict_with_uncertainty(self, candidate_inputs):
        predictions = np.array([[10.0], [20.0]])
        uncertainties = np.array([[0.1], [0.1]])
        return predictions, uncertainties, None


def test_gp_evaluation_preserves_design_space_identity_after_sorting():
    labeled = pd.DataFrame({
        "x": [0.0, 1.0],
        "target": [0.0, 1.0],
    })
    candidate_inputs = pd.DataFrame({"x": [2.0, 3.0]}, index=[5, 9])
    candidate_inputs.attrs["identity_columns"] = pd.DataFrame(
        {"Row number": [6, 10], "Idx_Sample": [100, 200]},
        index=[5, 9],
    )

    result = evaluate_gp_model(
        _FakeGP(),
        labeled,
        candidate_inputs,
        input_columns=["x"],
        target_columns=["target"],
        weights=np.array([1.0]),
        max_or_min=["max"],
        curiosity=0.0,
    )

    assert result.iloc[0]["Row number"] == 10
    assert result.iloc[0]["Idx_Sample"] == 200


def test_acquisition_recalculation_uses_predicted_target_fallback():
    labeled = pd.DataFrame({"target": [0.0, 1.0]})
    results = pd.DataFrame({
        "target": [np.nan, np.nan],
        "Predicted_target": [1.5, 2.0],
        "Uncertainty (target)": [0.1, 0.1],
    })

    recalculated = MLEngine._recalculate_utility(
        results,
        labeled,
        target_columns=["target"],
        weights=np.array([1.0]),
        max_or_min=["max"],
        curiosity=0.0,
        acquisition_function="ucb",
    )

    assert recalculated.loc[1, "Utility"] > recalculated.loc[0, "Utility"]


class _FakeLLMAgent:
    def _build_system_prompt(self, prompt_style, llm_strategy):
        return "system"

    def _build_user_prompt(self, context, history_df, params_config, prompt_style, target_config=None, strategy=None):
        return "user"

    def propose_next_experiment(self, context, history_df, params_config, prompt_style, target_config=None, strategy=None):
        return "oxide: high"


def test_hybrid_mode_passes_selected_acquisition_to_ml_engine():
    data = pd.DataFrame({
        "oxide": [0.0, 1.0, 2.0, 3.0],
        "target": [0.0, 1.0, np.nan, np.nan],
        "prior": [1.0, 1.0, 1.0, 1.0],
    })
    ml_results = pd.DataFrame({
        "oxide": [2.0, 3.0],
        "Utility": [0.2, 0.8],
        "Uncertainty": [0.1, 0.2],
    })
    config = {
        "model": "gp",
        "curiosity": 1.2,
        "acquisition_function": "ucb",
        "batch_size": 1,
        "target_columns": [{"name": "target", "weight": 1.0, "optimization": "max"}],
        "apriori_columns": [{"name": "prior", "weight": 0.2, "optimization": "min"}],
    }

    with patch.object(MLEngine, "run_experiment", return_value=ml_results) as run_mock, \
            patch.object(HybridEngine, "_get_llm_agent", return_value=_FakeLLMAgent()), \
            patch.object(HybridEngine, "_record_trajectory"):
        HybridEngine._run_hybrid_mode(data, config, ["oxide"], ["target"])

    _, kwargs = run_mock.call_args
    assert kwargs["curiosity"] == 1.2
    assert kwargs["acquisition_function"] == "ucb"
    assert kwargs["apriori_config"] == config["apriori_columns"]
