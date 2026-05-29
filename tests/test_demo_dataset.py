from flask import Flask

from app.api.uploads import uploads_bp
from app.utils import session_store
from app.utils.settings_manager import SettingsManager


def test_load_demo_dataset_copies_example_into_current_session(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(session_store, "SESSION_ROOT_DIR", tmp_path / "sessions")
    SettingsManager._settings_by_path = {}
    SettingsManager._settings = None

    examples_dir = tmp_path / "data" / "examples"
    examples_dir.mkdir(parents=True)
    (examples_dir / "metadesign_demo_cement.csv").write_text(
        "Idx_Sample,water_binder_ratio,fc_28d_MPa\n1,0.34,49.2\n2,0.38,\n"
    )

    app = Flask(__name__)
    app.config.update(TESTING=True, SECRET_KEY="test-secret")
    app.register_blueprint(uploads_bp)

    client = app.test_client(use_cookies=False)
    response = client.post(
        "/api/demo-datasets/load",
        json={},
        headers={"X-MetaDesign-Session": "browsersessiondemo123"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["filename"] == "metadesign_demo_cement.csv"
    assert payload["columns"] == ["Idx_Sample", "water_binder_ratio", "fc_28d_MPa"]

    session_file = tmp_path / "sessions" / "browsersessiondemo123" / "uploads" / "metadesign_demo_cement.csv"
    assert session_file.exists()

    settings_text = (tmp_path / "sessions" / "browsersessiondemo123" / "settings.json").read_text()
    assert "metadesign_demo_cement.csv" in settings_text
    assert "current_dataset_columns" in settings_text
