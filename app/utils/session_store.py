"""Session-scoped storage helpers for public/demo deployments."""

import os
import uuid
from pathlib import Path

from flask import current_app, has_request_context, request, session
from werkzeug.utils import secure_filename


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
SHARED_DESIGNSPACE_DIR = DATA_DIR / "designspaces"
SESSION_ROOT_DIR = DATA_DIR / "sessions"
SESSION_ID_KEY = "metadesign_session_id"


def _request_client_session_id():
    if not has_request_context():
        return None

    candidates = [
        request.headers.get("X-MetaDesign-Session"),
        request.args.get("client_session_id"),
        request.args.get("sid"),
    ]

    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        candidates.append(request.form.get("client_session_id"))
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            if isinstance(payload, dict):
                candidates.append(payload.get("client_session_id"))

    for candidate in candidates:
        candidate = secure_filename(str(candidate or ""))
        if len(candidate) >= 12:
            return candidate[:64]

    return None


def get_session_id():
    """Return a stable id for the current browser session."""
    if not has_request_context():
        return "global"

    client_session_id = _request_client_session_id()
    if client_session_id:
        session[SESSION_ID_KEY] = client_session_id
        return client_session_id

    if current_app.config.get("TESTING") and SESSION_ID_KEY not in session:
        session[SESSION_ID_KEY] = "test-session"
    elif SESSION_ID_KEY not in session:
        session[SESSION_ID_KEY] = uuid.uuid4().hex

    return session[SESSION_ID_KEY]


def get_session_root(create=True):
    """Return the filesystem root for the current session."""
    session_id = secure_filename(get_session_id())
    path = SESSION_ROOT_DIR / session_id
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_session_upload_dir(create=True):
    path = get_session_root(create=create) / "uploads"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_session_designspace_dir(create=True):
    path = get_session_root(create=create) / "designspaces"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_session_settings_path():
    return get_session_root(create=True) / "settings.json"


def get_session_trajectory_path():
    return get_session_root(create=True) / "trajectory_history.json"


def _is_within(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def resolve_dataset_path(reference, must_exist=True):
    """
    Resolve a user/session dataset reference to a safe local path.

    New uploads and generated design spaces live under data/sessions/<id>/.
    Bundled repository datasets remain readable as shared examples.
    """
    if not reference:
        return None

    reference_str = str(reference)
    candidate_path = Path(reference_str)
    if candidate_path.is_absolute():
        resolved = candidate_path.resolve()
        allowed_roots = [get_session_root(create=True), DATA_DIR]
        if any(_is_within(resolved, root) for root in allowed_roots):
            if not must_exist or resolved.exists():
                return str(resolved)
        return None

    filename = secure_filename(os.path.basename(reference_str))
    if not filename:
        return None

    looks_like_designspace = "designspace" in reference_str.lower()
    candidates = []
    if looks_like_designspace or "designspaces" in reference_str:
        candidates.extend([
            get_session_designspace_dir(create=True) / filename,
            SHARED_DESIGNSPACE_DIR / filename,
        ])
    candidates.extend([
        get_session_upload_dir(create=True) / filename,
        get_session_designspace_dir(create=True) / filename,
        DATA_DIR / filename,
        SHARED_DESIGNSPACE_DIR / filename,
    ])

    for candidate in candidates:
        candidate = candidate.resolve()
        if any(_is_within(candidate, root) for root in [get_session_root(create=True), DATA_DIR]):
            if not must_exist or candidate.exists():
                return str(candidate)

    return None


def list_session_and_shared_datasets(include_shared_examples=False):
    """Return dataset descriptors visible to the current session.

    Public workflows should show only files the user created or uploaded in
    their own browser session. Bundled repository examples can still be exposed
    explicitly for demos/tests by passing include_shared_examples=True or by
    setting SHOW_SHARED_EXAMPLES in Flask config.
    """
    datasets = []

    roots = [
        ("Design Space", get_session_designspace_dir(create=True), "bi-grid-3x3-gap", "design_space"),
        ("Uploaded Dataset", get_session_upload_dir(create=True), "bi-file-earmark-spreadsheet", "uploaded"),
    ]

    if include_shared_examples or current_app.config.get("SHOW_SHARED_EXAMPLES"):
        roots.extend([
            ("Shared Design Space", SHARED_DESIGNSPACE_DIR, "bi-grid-3x3-gap", "shared_design_space"),
            ("Shared Example", DATA_DIR, "bi-file-earmark-spreadsheet", "shared_example"),
        ])

    seen = set()
    for source, root, icon, kind in roots:
        if not root.exists():
            continue
        for ext in ("*.csv", "*.xlsx", "*.xls"):
            for filepath in root.glob(ext):
                if filepath.name in {"scenarios.csv", "trajectory_history.json"}:
                    continue
                key = (filepath.name, source)
                if key in seen:
                    continue
                seen.add(key)
                datasets.append({
                    "name": filepath.name,
                    "path": str(filepath.resolve()),
                    "source": source,
                    "kind": kind,
                    "size_kb": round(filepath.stat().st_size / 1024, 1),
                    "icon": icon,
                })

    datasets.sort(key=lambda x: (x["source"], x["name"]))
    return datasets
