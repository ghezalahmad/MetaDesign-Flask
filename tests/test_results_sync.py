import pandas as pd
from flask import Flask
from unittest.mock import patch

from app.api.results import results_bp
from app.database import Cycle, Project, Sample, db
from app.utils import session_store


def _make_app(tmp_path):
    app = Flask(__name__)
    app.config.update(
        TESTING=True,
        SECRET_KEY="test-secret",
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{tmp_path / 'results-test.db'}",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )
    db.init_app(app)
    app.register_blueprint(results_bp)
    with app.app_context():
        db.create_all()
    return app


def _create_sample(dataset_path, idx_sample, row_data, lab_results, predictions=None):
    project = Project(name="Closed Loop Test", dataset_path=str(dataset_path), session_id="test-session")
    db.session.add(project)
    db.session.flush()

    cycle = Cycle(project_id=project.id, cycle_number=1)
    cycle.set_lab_result_columns(list(lab_results.keys()))
    db.session.add(cycle)
    db.session.flush()

    sample = Sample(cycle_id=cycle.id, idx_sample=idx_sample)
    sample.set_row_data(row_data)
    sample.set_predictions(predictions or {})
    sample.set_lab_results(lab_results)
    db.session.add(sample)
    db.session.commit()
    return sample


def test_sync_sample_updates_source_dataset_by_idx_sample_and_adds_missing_columns(tmp_path):
    dataset_path = tmp_path / "designspace.csv"
    pd.DataFrame({
        "Idx_Sample": [101, 102],
        "x": [1.0, 2.0],
        "strength": [pd.NA, pd.NA],
    }).to_csv(dataset_path, index=False)

    app = _make_app(tmp_path)
    with app.app_context():
        sample = _create_sample(
            dataset_path,
            idx_sample=102,
            row_data={"Idx_Sample": 102, "x": 2.0},
            lab_results={"strength": 44.5, "slump": 120.0},
        )
        response = app.test_client().post(f"/api/results/samples/{sample.id}/sync")

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["success"] is True
        assert payload["matched_by"] == "Idx_Sample"

        synced = pd.read_csv(dataset_path)
        row = synced.loc[synced["Idx_Sample"] == 102].iloc[0]
        assert row["strength"] == 44.5
        assert row["slump"] == 120.0
        assert db.session.get(Sample, sample.id).status == "completed"


def test_sync_sample_falls_back_to_row_position_when_dataset_has_no_identity_column(tmp_path):
    dataset_path = tmp_path / "uploaded.csv"
    pd.DataFrame({
        "x": [1.0, 2.0],
        "target": [pd.NA, pd.NA],
    }).to_csv(dataset_path, index=False)

    app = _make_app(tmp_path)
    with app.app_context():
        sample = _create_sample(
            dataset_path,
            idx_sample=2,
            row_data={"Row number": 2, "x": 2.0},
            lab_results={"target": 9.75},
        )
        response = app.test_client().post(f"/api/results/samples/{sample.id}/sync")

        assert response.status_code == 200
        synced = pd.read_csv(dataset_path)
        assert synced.loc[1, "target"] == 9.75


def test_results_tsne_endpoint_returns_cycle_lab_and_error_overlays(tmp_path):
    dataset_path = tmp_path / "designspace.csv"
    pd.DataFrame({
        "Idx_Sample": [101, 102, 103],
        "x": [1.0, 2.0, 3.0],
        "constant": [5.0, 5.0, 5.0],
        "strength": [pd.NA, pd.NA, pd.NA],
    }).to_csv(dataset_path, index=False)

    app = _make_app(tmp_path)
    with app.app_context():
        sample = _create_sample(
            dataset_path,
            idx_sample=102,
            row_data={"Idx_Sample": 102, "x": 2.0, "constant": 5.0},
            lab_results={"strength": 44.5},
            predictions={"Predicted_strength": 43.0, "Utility": 0.91},
        )
        project_id = db.session.get(Cycle, sample.cycle_id).project_id

        def fake_tsne(df, *args, **kwargs):
            df = df.copy()
            df["tsne-2d-one"] = [0.0, 1.0, 2.0]
            df["tsne-2d-two"] = [2.0, 1.0, 0.0]
            return df

        with patch("app.api.results.PlotGenerator._run_tsne", side_effect=fake_tsne):
            response = app.test_client().get(
                f"/api/results/projects/{project_id}/tsne?input_columns=x&input_columns=constant"
            )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["success"] is True
        assert payload["feature_columns"] == ["x", "constant"]
        assert "Lab_Status" in payload["overlay_parameters"]
        assert any("Constant columns" in warning for warning in payload["warnings"])

        selected = [row for row in payload["rows"] if row.get("Selected_For_Lab") is True]
        assert len(selected) == 1
        assert selected[0]["Lab_Status"] == "completed"
        assert selected[0]["Measured_strength"] == 44.5
        assert selected[0]["Prediction_Error_strength"] == 1.5


def test_create_cycle_rejects_samples_from_different_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "DATA_DIR", tmp_path)
    dataset_a = tmp_path / "dataset_a.csv"
    dataset_b = tmp_path / "dataset_b.csv"
    pd.DataFrame({"Idx_Sample": [1], "x": [1.0], "target": [pd.NA]}).to_csv(dataset_a, index=False)
    pd.DataFrame({"Idx_Sample": [2], "x": [2.0], "target": [pd.NA]}).to_csv(dataset_b, index=False)

    app = _make_app(tmp_path)
    with app.app_context():
        project = Project(name="Dataset A", dataset_path=str(dataset_a), session_id="test-session")
        db.session.add(project)
        db.session.commit()

        response = app.test_client().post("/api/results/cycles", json={
            "project_id": project.id,
            "dataset_path": str(dataset_b),
            "samples": [{
                "idx_sample": 2,
                "row_data": {"Idx_Sample": 2, "x": 2.0},
                "predictions": {"Predicted_target": 4.2},
            }],
            "lab_result_columns": ["target"],
        })

        assert response.status_code == 400
        assert "different dataset" in response.get_json()["error"]
        assert Cycle.query.filter_by(project_id=project.id).count() == 0
