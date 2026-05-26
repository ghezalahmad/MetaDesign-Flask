import json

import pytest

from app.engines.hybrid_engine import HybridEngine
from app.utils import settings_manager as settings_module
from app.utils.settings_manager import SettingsManager


@pytest.fixture(autouse=True)
def reset_settings_manager_cache():
    SettingsManager._settings = None
    yield
    SettingsManager._settings = None


def test_settings_manager_does_not_persist_llm_api_keys(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(settings_module, "SETTINGS_FILE", str(settings_file))
    SettingsManager._settings = None

    assert SettingsManager.save_settings({
        "llm_provider": "openai",
        "llm_model": "gpt-5-mini",
        "llm_api_key": "run-secret",
        "openai_api_key": "provider-secret",
        "mistral_api_key": "legacy-secret",
        "anthropic_api_key": "anthropic-secret",
    })

    saved = json.loads(settings_file.read_text())
    assert saved["llm_provider"] == "openai"
    assert saved["llm_model"] == "gpt-5-mini"
    assert "llm_api_key" not in saved
    assert "openai_api_key" not in saved
    assert "mistral_api_key" not in saved
    assert "anthropic_api_key" not in saved


def test_cloud_llm_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OpenAI API key is required"):
        HybridEngine._get_llm_agent({
            "llm_provider": "openai",
            "llm_model": "gpt-5-mini",
        })


def test_per_run_api_key_overrides_deployment_secret(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "deployment-secret")

    agent = HybridEngine._get_llm_agent({
        "llm_provider": "anthropic",
        "llm_model": "claude-sonnet-4-20250514",
        "llm_api_key": "per-run-secret",
    })

    assert agent.provider == "anthropic"
    assert agent.model == "claude-sonnet-4-20250514"
    assert agent.api_key == "per-run-secret"


def test_llm_provider_error_text_is_not_used_for_candidate_matching():
    with pytest.raises(ValueError, match="OPENAI_ERROR"):
        HybridEngine._raise_if_llm_error("OPENAI_ERROR: invalid_api_key")
