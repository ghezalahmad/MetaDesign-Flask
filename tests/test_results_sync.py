import pandas as pd
from flask import Flask

from app.api.results import results_bp
from app.database import Cycle, Project, Sample, db


def _make_app(tmp_path):
    app = Flask(__name__)
    app.config.update(
        TESTING=True,
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{tmp_path / 'results-test.db'}",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )
    db.init_app(app)
    app.register_blueprint(results_bp)
    with app.app_context():
        db.create_all()
    return app


def _create_sample(dataset_path, idx_sample, row_data, lab_results):
    project = Project(name="Closed Loop Test", dataset_path=str(dataset_path))
    db.session.add(project)
    db.session.flush()

    cycle = Cycle(project_id=project.id, cycle_number=1)
    cycle.set_lab_result_columns(list(lab_results.keys()))
    db.session.add(cycle)
    db.session.flush()

    sample = Sample(cycle_id=cycle.id, idx_sample=idx_sample)
    sample.set_row_data(row_data)
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
