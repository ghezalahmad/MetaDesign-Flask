from pathlib import Path

from flask import Blueprint, Flask
from werkzeug.datastructures import MultiDict

from app.api import design_space as design_space_module
from app.api.design_space import design_space_bp
from app.utils import session_store


def _make_design_space_app(tmp_path, monkeypatch):
    session_root = tmp_path / "sessions"
    shared_root = tmp_path / "designspaces"
    shared_root.mkdir(parents=True)

    monkeypatch.setattr(session_store, "SESSION_ROOT_DIR", session_root)
    monkeypatch.setattr(design_space_module, "SHARED_DESIGNSPACE_DIR", shared_root)

    template_dir = Path(__file__).resolve().parents[1] / "templates"
    app = Flask(__name__, template_folder=str(template_dir))
    app.config.update(TESTING=True, SECRET_KEY="test-secret")
    main_bp = Blueprint("main", __name__)

    @main_bp.route("/dashboard")
    def dashboard():
        return "dashboard"

    app.register_blueprint(main_bp)
    app.register_blueprint(design_space_bp)
    return app, session_root, shared_root


def _set_session_id(client, session_id="session-a"):
    with client.session_transaction() as sess:
        sess[session_store.SESSION_ID_KEY] = session_id


def test_delete_session_design_space(tmp_path, monkeypatch):
    app, session_root, _ = _make_design_space_app(tmp_path, monkeypatch)
    client = app.test_client()
    _set_session_id(client)

    session_file = session_root / "session-a" / "designspaces" / "session.csv"
    session_file.parent.mkdir(parents=True)
    session_file.write_text("Idx_Sample,target\n1,\n")

    response = client.delete("/delete-design-space/session.csv?scope=session")

    assert response.status_code == 200
    assert response.get_json()["success"] is True
    assert not session_file.exists()


def test_delete_shared_design_space_allowed_locally(tmp_path, monkeypatch):
    app, _, shared_root = _make_design_space_app(tmp_path, monkeypatch)
    client = app.test_client()
    _set_session_id(client)

    shared_file = shared_root / "shared.csv"
    shared_file.write_text("Idx_Sample,target\n1,\n")

    response = client.delete(
        "/delete-design-space/shared.csv?scope=shared",
        base_url="http://127.0.0.1:5000",
    )

    assert response.status_code == 200
    assert response.get_json()["success"] is True
    assert not shared_file.exists()


def test_delete_shared_design_space_denied_for_public_host(tmp_path, monkeypatch):
    app, _, shared_root = _make_design_space_app(tmp_path, monkeypatch)
    client = app.test_client()
    _set_session_id(client)

    shared_file = shared_root / "shared.csv"
    shared_file.write_text("Idx_Sample,target\n1,\n")

    response = client.delete(
        "/delete-design-space/shared.csv?scope=shared",
        base_url="https://ghezalahmad-meta-design.hf.space",
    )

    assert response.status_code == 403
    assert response.get_json()["success"] is False
    assert shared_file.exists()


def test_edit_shared_design_space_denied_for_public_host(tmp_path, monkeypatch):
    app, _, shared_root = _make_design_space_app(tmp_path, monkeypatch)
    client = app.test_client()
    _set_session_id(client)

    shared_file = shared_root / "shared.csv"
    shared_file.write_text("Idx_Sample,target\n1,\n")

    response = client.post(
        "/api/design-space-data/shared.csv?scope=shared",
        json={"column": "target", "updates": {"0": "2.5"}},
        base_url="https://ghezalahmad-meta-design.hf.space",
    )

    assert response.status_code == 403
    assert response.get_json()["success"] is False
    assert shared_file.read_text() == "Idx_Sample,target\n1,\n"


def test_generate_design_space_requires_target_property(tmp_path, monkeypatch):
    app, session_root, _ = _make_design_space_app(tmp_path, monkeypatch)
    client = app.test_client()
    _set_session_id(client)

    response = client.post("/generate-design-space", data=MultiDict([
        ("material_name", "Binder"),
        ("feature_name", "water"),
        ("feature_type", "continuous"),
        ("min", "0.1"),
        ("max", "0.2"),
        ("step", "0.1"),
    ]))

    assert response.status_code == 302
    assert "error=" in response.headers["Location"]
    assert "target" in response.headers["Location"].lower()
    assert not (session_root / "session-a" / "designspaces").exists()


def test_generate_design_space_creates_session_csv(tmp_path, monkeypatch):
    app, session_root, _ = _make_design_space_app(tmp_path, monkeypatch)
    client = app.test_client()
    _set_session_id(client)

    response = client.post("/generate-design-space", data=MultiDict([
        ("material_name", "Binder"),
        ("feature_name", "water"),
        ("feature_type", "continuous"),
        ("min", "0.1"),
        ("max", "0.2"),
        ("step", "0.1"),
        ("target_name", "strength"),
    ]))

    designspace_dir = session_root / "session-a" / "designspaces"
    generated_files = list(designspace_dir.glob("designspace_Binder_*.csv"))

    assert response.status_code == 302
    assert "/design-space?generated=designspace_Binder_" in response.headers["Location"]
    assert len(generated_files) == 1
    assert "strength" in generated_files[0].read_text()


def test_generate_design_space_remains_visible_without_cookie(tmp_path, monkeypatch):
    app, session_root, _ = _make_design_space_app(tmp_path, monkeypatch)
    client_session_id = "browsersessiona123"
    client = app.test_client(use_cookies=False)

    response = client.post("/generate-design-space", data=MultiDict([
        ("client_session_id", client_session_id),
        ("material_name", "Binder"),
        ("feature_name", "water"),
        ("feature_type", "continuous"),
        ("min", "0.1"),
        ("max", "0.2"),
        ("step", "0.1"),
        ("target_name", "strength"),
    ]), follow_redirects=True)

    designspace_dir = session_root / client_session_id / "designspaces"
    generated_files = list(designspace_dir.glob("designspace_Binder_*.csv"))

    assert response.status_code == 200
    assert len(generated_files) == 1
    assert generated_files[0].name.encode() in response.data
    assert b"Current session" in response.data
    assert b"No design spaces generated yet" not in response.data
