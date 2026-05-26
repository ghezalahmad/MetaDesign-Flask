from flask import Flask

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
