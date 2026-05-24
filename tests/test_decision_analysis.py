import json

import numpy as np
import pandas as pd
from flask import Flask
from unittest.mock import patch

from app.api.run_experiment import run_experiment_bp
from app.utils.decision_analysis import DecisionAnalyzer


def _source_data():
    return pd.DataFrame({
        "Row number": [1, 2, 3, 4, 5, 6],
        "x1": [0.0, 1.0, 2.0, 8.0, 9.0, 10.0],
        "x2": [0.0, 0.5, 1.0, 8.0, 9.0, 10.0],
        "strength": [10.0, 12.0, np.nan, np.nan, np.nan, np.nan],
        "porosity": [4.0, 3.0, np.nan, np.nan, np.nan, np.nan],
        "cost": [2.0, 4.0, 6.0, 12.0, 3.0, 5.0],
        "fidelity": ["pilot", "pilot", "screen", "screen", "pilot", "screen"],
    })


def _result_data():
    return pd.DataFrame({
        "Row number": [1, 2, 3, 4],
        "x1": [0.0, 1.0, 2.0, 8.0],
        "x2": [0.0, 0.5, 1.0, 8.0],
        "strength": [10.0, 12.0, 8.0, 15.0],
        "porosity": [4.0, 3.0, 2.0, 1.0],
        "cost": [2.0, 4.0, 6.0, 12.0],
        "fidelity": ["pilot", "pilot", "screen", "screen"],
        "Utility": [0.2, 0.7, 0.4, 0.9],
        "Uncertainty": [0.05, 0.1, 0.6, 0.2],
        "Selected for Testing": [False, True, False, False],
        "is_train_data": [False, False, False, False],
    })


def _target_configs():
    return [
        {"name": "strength", "weight": 1.0, "optimization": "max", "threshold": 9.0},
        {"name": "porosity", "weight": 1.0, "optimization": "min", "threshold": 3.0},
    ]


def test_decision_analyzer_adds_model_agnostic_columns_and_plots():
    results, analysis = DecisionAnalyzer.apply(
        results_df=_result_data(),
        source_df=_source_data(),
        input_columns=["x1", "x2"],
        target_configs=_target_configs(),
        apriori_configs=[{"name": "cost", "weight": 1.0, "optimization": "min", "threshold": 8.0}],
        config={
            "model": "lolopy",
            "active_learning_mode": "ML_MODE",
            "batch_size": 2,
            "decision_settings": {
                "cost_column": "cost",
                "fidelity_column": "fidelity",
                "prefer_feasible": True,
            },
        },
    )

    expected_columns = {
        "Constraint_Feasible",
        "Constraint_Violations",
        "Pareto_Front",
        "Pareto_Rank",
        "OOD_Risk",
        "Trust_Score",
        "Trust_Flag",
        "Experiment_Cost",
        "Fidelity_Level",
        "Cost_Penalty",
        "Decision_Score",
        "Cost_Adjusted_Utility",
        "Decision_Action",
    }
    assert expected_columns.issubset(results.columns)
    assert int(results["Selected for Testing"].sum()) == 2
    assert analysis["summary"]["selected_count"] == 2
    assert set(analysis["plots"].keys()) == {"pareto", "trust", "batch", "fidelity"}
    assert analysis["fidelity"]["cost_column"] == "cost"
    json.dumps(analysis)


def test_decision_analyzer_applies_force_and_reject_after_any_engine_mode():
    results, analysis = DecisionAnalyzer.apply(
        results_df=_result_data(),
        source_df=_source_data(),
        input_columns=["x1", "x2"],
        target_configs=_target_configs(),
        apriori_configs=[],
        config={
            "model": "llm-agent",
            "active_learning_mode": "LLM_AGENT_MODE",
            "batch_size": 2,
            "decision_settings": {
                "force_rows": "4",
                "reject_rows": "2",
                "oversight_notes": "Manual lab override",
                "prefer_feasible": True,
            },
        },
    )

    row4 = results.loc[results["Row number"] == 4].iloc[0]
    row2 = results.loc[results["Row number"] == 2].iloc[0]

    assert bool(row4["Selected for Testing"]) is True
    assert row4["Decision_Action"] == "Force include"
    assert bool(row2["Selected for Testing"]) is False
    assert row2["Decision_Action"] == "Rejected by oversight"
    assert analysis["oversight"]["force_rows"] == ["4"]
    assert analysis["oversight"]["reject_rows"] == ["2"]
    assert analysis["oversight"]["notes"] == "Manual lab override"


def test_run_experiment_route_applies_decision_layer_after_engine(tmp_path):
    dataset = _source_data().iloc[:4].copy()
    dataset_path = tmp_path / "route_dataset.csv"
    dataset.to_csv(dataset_path, index=False)

    engine_results = pd.DataFrame({
        "Row number": [1, 2, 3, 4],
        "x1": [0.0, 1.0, 2.0, 8.0],
        "x2": [0.0, 0.5, 1.0, 8.0],
        "strength": [10.0, 12.0, 8.0, 15.0],
        "porosity": [4.0, 3.0, 2.0, 1.0],
        "cost": [2.0, 4.0, 6.0, 12.0],
        "fidelity": ["pilot", "pilot", "screen", "screen"],
        "Utility": [0.2, 0.7, 0.4, 0.9],
        "Uncertainty": [0.05, 0.1, 0.6, 0.2],
        "Selected for Testing": [False, False, True, False],
        "is_train_data": [True, True, False, False],
    })
    engine_results.attrs["llm_trace"] = {"mode": "LLM_AGENT_MODE", "raw_response": "row 4"}

    app = Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(run_experiment_bp)
    client = app.test_client()

    with client.session_transaction() as session:
        session["filepath"] = str(dataset_path)

    empty_plot = {"data": [], "layout": {}}
    with patch("app.engines.hybrid_engine.HybridEngine.run_experiment", return_value=engine_results), \
            patch("app.api.run_experiment.PlotGenerator._run_tsne", side_effect=lambda df, *args, **kwargs: df.assign(**{"tsne-2d-one": 0.0, "tsne-2d-two": 0.0})), \
            patch("app.api.run_experiment.PlotGenerator.create_tsne_input_space_plot", return_value=empty_plot), \
            patch("app.api.run_experiment.PlotGenerator.create_target_scatter_plot", return_value=empty_plot), \
            patch("app.api.run_experiment.PlotGenerator.create_uncertainty_plot", return_value=empty_plot), \
            patch("app.api.run_experiment.PlotGenerator.create_optimization_history_plot", return_value=empty_plot), \
            patch("app.api.run_experiment.PlotGenerator.create_utility_surface_plot", return_value=empty_plot), \
            patch("app.api.run_experiment.PlotGenerator.create_trajectory_plot", return_value=empty_plot), \
            patch("app.api.run_experiment.PlotGenerator.create_distance_plot", return_value=empty_plot), \
            patch("app.api.run_experiment.PlotGenerator.create_feature_importance_plot", return_value=empty_plot), \
            patch("app.api.run_experiment.PlotGenerator.create_prediction_actual_plot", return_value=empty_plot), \
            patch("app.api.run_experiment.TrajectoryTracker.get_trajectory_summary", return_value={}):
        response = client.post("/run-experiment", json={
            "model": "llm-agent",
            "curiosity": 0.5,
            "input_columns": ["x1", "x2"],
            "target_columns": _target_configs(),
            "apriori_columns": [{"name": "cost", "weight": 1.0, "optimization": "min", "threshold": 8.0}],
            "active_learning_mode": "LLM_AGENT_MODE",
            "batch_size": 1,
            "decision_settings": {
                "cost_column": "cost",
                "fidelity_column": "fidelity",
                "force_rows": "4",
                "prefer_feasible": True,
            },
        })

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["decision_analysis"]["summary"]["selected_count"] == 1
    assert payload["decision_analysis"]["selected_batch"][0]["Decision_Action"] == "Force include"
    assert payload["llm_trace"]["mode"] == "LLM_AGENT_MODE"
