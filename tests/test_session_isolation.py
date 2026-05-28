from flask import Flask

from app.api.results import results_bp
from app.api.scenarios import scenarios_bp
from app.api.settings import settings_bp
from app.utils import session_store
from app.utils.settings_manager import SettingsManager


def _make_settings_app(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "SESSION_ROOT_DIR", tmp_path / "sessions")
    SettingsManager._settings_by_path = {}
    SettingsManager._settings = None

    app = Flask(__name__)
    app.config.update(TESTING=True, SECRET_KEY="test-secret")
    app.register_blueprint(settings_bp)
    return app


def test_settings_are_isolated_by_browser_session(tmp_path, monkeypatch):
    app = _make_settings_app(tmp_path, monkeypatch)
    client_a = app.test_client()
    client_b = app.test_client()
    with client_a.session_transaction() as sess:
        sess[session_store.SESSION_ID_KEY] = "session-a"
    with client_b.session_transaction() as sess:
        sess[session_store.SESSION_ID_KEY] = "session-b"

    response = client_a.post("/api/settings", json={
        "current_dataset": "private-a.csv",
        "llm_provider": "openai",
        "llm_api_key": "do-not-save",
    })
    assert response.status_code == 200
    assert response.get_json()["success"] is True

    settings_a = client_a.get("/api/settings").get_json()["settings"]
    settings_b = client_b.get("/api/settings").get_json()["settings"]

    assert settings_a["current_dataset"] == "private-a.csv"
    assert settings_a["llm_provider"] == "openai"
    assert "llm_api_key" not in settings_a
    assert "current_dataset" not in settings_b
    assert settings_b["llm_provider"] == "ollama"


def test_dataset_registry_lists_only_current_session_files_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "SESSION_ROOT_DIR", tmp_path / "sessions")
    monkeypatch.setattr(session_store, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(session_store, "SHARED_DESIGNSPACE_DIR", tmp_path / "data" / "designspaces")

    session_root = tmp_path / "sessions" / "session-a"
    upload_dir = session_root / "uploads"
    designspace_dir = session_root / "designspaces"
    upload_dir.mkdir(parents=True)
    designspace_dir.mkdir(parents=True)
    (upload_dir / "uploaded.csv").write_text("Idx_Sample,target\n1,\n")
    (designspace_dir / "designspace_created.csv").write_text("Idx_Sample,target\n1,\n")

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "MaterialsDiscoveryExampleData.csv").write_text("Idx_Sample,target\n1,\n")

    app = Flask(__name__)
    app.config.update(TESTING=True, SECRET_KEY="test-secret")
    app.register_blueprint(scenarios_bp)
    app.register_blueprint(results_bp)

    client = app.test_client()
    with client.session_transaction() as sess:
        sess[session_store.SESSION_ID_KEY] = "session-a"

    for endpoint in ["/api/scenarios/datasets", "/api/results/datasets"]:
        response = client.get(endpoint)
        assert response.status_code == 200
        datasets = response.get_json()["datasets"]
        names = {dataset["name"] for dataset in datasets}
        sources = {dataset["name"]: dataset["source"] for dataset in datasets}

        assert names == {"uploaded.csv", "designspace_created.csv"}
        assert sources["uploaded.csv"] == "Uploaded Dataset"
        assert sources["designspace_created.csv"] == "Design Space"
