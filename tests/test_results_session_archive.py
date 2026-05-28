import pandas as pd
from flask import Flask

from app.api.results import results_bp
from app.database import Cycle, Project, Sample, Scenario, db
from app.utils import session_store
from app.utils.settings_manager import SettingsManager


def _make_app(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "SESSION_ROOT_DIR", tmp_path / "sessions")
    SettingsManager._settings_by_path = {}
    SettingsManager._settings = None

    app = Flask(__name__)
    app.config.update(
        TESTING=True,
        SECRET_KEY="test-secret",
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{tmp_path / 'archive-test.db'}",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )
    db.init_app(app)
    app.register_blueprint(results_bp)
    with app.app_context():
        db.create_all()
    return app


def test_results_session_export_import_restores_datasets_projects_cycles_and_samples(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)
    source_session = "sessionexport123"
    target_session = "sessionimport123"
    source_designspace = tmp_path / "sessions" / source_session / "designspaces" / "designspace_archive.csv"
    source_designspace.parent.mkdir(parents=True)
    pd.DataFrame({
        "Idx_Sample": [1, 2],
        "water": [0.4, 0.5],
        "strength": [pd.NA, pd.NA],
    }).to_csv(source_designspace, index=False)

    with app.app_context():
        project = Project(
            name="Archive Project",
            dataset_path=str(source_designspace),
            session_id=source_session,
        )
        db.session.add(project)
        db.session.flush()

        scenario = Scenario(
            project_id=project.id,
            name="Budget plan",
            planned_cycles=3,
            samples_per_cycle=4,
            initial_samples=2,
            cost_per_sample=12.5,
        )
        db.session.add(scenario)
        db.session.flush()
        project.active_scenario_id = scenario.id

        cycle = Cycle(project_id=project.id, cycle_number=2, notes="second cycle")
        cycle.set_lab_result_columns(["strength"])
        db.session.add(cycle)
        db.session.flush()

        sample = Sample(cycle_id=cycle.id, idx_sample=2, status="completed")
        sample.set_row_data({"Idx_Sample": 2, "water": 0.5})
        sample.set_predictions({"Predicted_strength": 42.0, "Utility": 0.8})
        sample.set_lab_results({"strength": 43.5})
        db.session.add(sample)
        db.session.commit()

    client = app.test_client(use_cookies=False)
    export_response = client.get(
        "/api/results/session/export",
        headers={"X-MetaDesign-Session": source_session},
    )
    assert export_response.status_code == 200
    archive = export_response.get_json()
    assert archive["schema"] == "metadesign-session"
    assert archive["datasets"][0]["name"] == "designspace_archive.csv"

    import_response = client.post(
        "/api/results/session/import",
        json=archive,
        headers={"X-MetaDesign-Session": target_session},
    )
    assert import_response.status_code == 200
    payload = import_response.get_json()
    assert payload["success"] is True
    assert payload["datasets_restored"] == 1
    assert payload["projects_restored"] == 1

    restored_file = tmp_path / "sessions" / target_session / "designspaces" / "designspace_archive.csv"
    assert restored_file.exists()

    with app.app_context():
        restored_project = Project.query.filter_by(session_id=target_session).one()
        assert restored_project.name == "Archive Project"
        assert restored_project.dataset_path == str(restored_file.resolve())
        assert len(restored_project.scenarios) == 1
        assert restored_project.scenarios[0].name == "Budget plan"
        assert restored_project.active_scenario_id == restored_project.scenarios[0].id
        assert len(restored_project.cycles) == 1
        restored_cycle = restored_project.cycles[0]
        assert restored_cycle.cycle_number == 2
        assert restored_cycle.get_lab_result_columns() == ["strength"]
        assert len(restored_cycle.samples) == 1
        restored_sample = restored_cycle.samples[0]
        assert restored_sample.idx_sample == 2
        assert restored_sample.get_lab_results()["strength"] == 43.5
