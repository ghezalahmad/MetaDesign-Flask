
import json
import os
import logging
import copy
from flask import has_request_context
from app.utils.session_store import get_session_settings_path

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'settings.json')

# Default settings - defined once to avoid duplication
DEFAULT_SETTINGS = {
    "active_learning_mode": "ML_MODE",
    "prompt_style": "parameter-format",
    "llm_provider": "ollama",
    "ollama_model": "mistral:latest",
    "llm_model": "mistral:latest",
    "llm_strategy": "balanced",
    "hybrid_weights": {
        "w_llm": 0.5,
        "w_ml": 0.5
    },
    "alpha_value": 1.0
}

SENSITIVE_SETTING_KEYS = {
    "llm_api_key",
    "mistral_api_key",
    "openai_api_key",
    "anthropic_api_key",
    "cohere_api_key",
}


class SettingsManager:
    _settings = None
    _settings_by_path = {}

    @classmethod
    def _get_defaults(cls):
        """Return a copy of default settings."""
        return copy.deepcopy(DEFAULT_SETTINGS)

    @classmethod
    def load_settings(cls):
        """Load settings from file, or create with defaults if missing."""
        settings_file = str(get_session_settings_path()) if has_request_context() else SETTINGS_FILE

        if settings_file in cls._settings_by_path:
            return copy.deepcopy(cls._settings_by_path[settings_file])

        if not os.path.exists(settings_file):
            # Initialize with defaults and save (without calling save_settings to avoid recursion)
            settings = cls._get_defaults()
            cls._write_to_file(settings, settings_file=settings_file)
            cls._settings_by_path[settings_file] = settings
            if not has_request_context():
                cls._settings = settings
            return copy.deepcopy(settings)

        try:
            with open(settings_file, 'r') as f:
                loaded_settings = json.load(f)
                settings = cls.strip_sensitive(loaded_settings)
                if loaded_settings != settings:
                    cls._write_to_file(settings, settings_file=settings_file)
        except Exception as e:
            logging.error(f"Failed to load settings: {e}")
            settings = cls._get_defaults()  # Use defaults on error
        
        cls._settings_by_path[settings_file] = settings
        if not has_request_context():
            cls._settings = settings
        return copy.deepcopy(settings)

    @classmethod
    def _write_to_file(cls, settings, settings_file=None):
        """Internal method to write settings to file (no merge, no recursion)."""
        try:
            settings = cls.strip_sensitive(settings)
            settings_file = settings_file or (str(get_session_settings_path()) if has_request_context() else SETTINGS_FILE)
            os.makedirs(os.path.dirname(settings_file), exist_ok=True)
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=4)
            return True
        except Exception as e:
            logging.error(f"Failed to write settings file: {e}")
            return False

    @classmethod
    def save_settings(cls, new_settings):
        """Save new settings, merging with existing settings."""
        try:
            settings_file = str(get_session_settings_path()) if has_request_context() else SETTINGS_FILE
            
            # Merge new settings into current
            current = cls.load_settings()
            current.update(cls.strip_sensitive(new_settings or {}))
            current = cls.strip_sensitive(current)
            
            # Write to file
            if cls._write_to_file(current, settings_file=settings_file):
                cls._settings_by_path[settings_file] = current
                if not has_request_context():
                    cls._settings = current
                return True
            return False
        except Exception as e:
            logging.error(f"Failed to save settings: {e}")
            return False

    @classmethod
    def get_setting(cls, key, default=None):
        """Get a specific setting value."""
        settings = cls.load_settings()
        return settings.get(key, default)

    @classmethod
    def get_api_key(cls, key_name, env_var_name=None):
        """
        Get an API key securely.
        
        Priority: Environment variable > Settings file
        This allows secure deployment without storing keys in files.
        
        Args:
            key_name: The key name in settings.json (e.g., 'mistral_api_key')
            env_var_name: Optional environment variable name (defaults to uppercase of key_name)
        
        Returns:
            The API key or empty string if not found
        """
        # Check environment variable first (most secure)
        if env_var_name is None:
            env_var_name = key_name.upper()
        
        env_value = os.environ.get(env_var_name)
        if env_value:
            return env_value
        
        # Fall back to settings file (for backwards compatibility)
        return cls.get_setting(key_name, "")

    @classmethod
    def strip_sensitive(cls, settings):
        """Return a copy of settings without API keys or other secrets."""
        safe = dict(settings or {})
        for key in SENSITIVE_SETTING_KEYS:
            safe.pop(key, None)
        return safe
