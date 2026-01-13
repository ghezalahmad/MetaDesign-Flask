
import json
import os
import logging

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'settings.json')

# Default settings - defined once to avoid duplication
DEFAULT_SETTINGS = {
    "active_learning_mode": "ML_MODE",
    "prompt_style": "parameter-format",
    "hybrid_weights": {
        "w_llm": 0.5,
        "w_ml": 0.5
    },
    "alpha_value": 1.0
}


class SettingsManager:
    _settings = None

    @classmethod
    def _get_defaults(cls):
        """Return a copy of default settings."""
        return DEFAULT_SETTINGS.copy()

    @classmethod
    def load_settings(cls):
        """Load settings from file, or create with defaults if missing."""
        if not os.path.exists(SETTINGS_FILE):
            # Initialize with defaults and save (without calling save_settings to avoid recursion)
            cls._settings = cls._get_defaults()
            cls._write_to_file(cls._settings)
            return cls._settings

        try:
            with open(SETTINGS_FILE, 'r') as f:
                cls._settings = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load settings: {e}")
            cls._settings = cls._get_defaults()  # Use defaults on error
        
        return cls._settings

    @classmethod
    def _write_to_file(cls, settings):
        """Internal method to write settings to file (no merge, no recursion)."""
        try:
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=4)
            return True
        except Exception as e:
            logging.error(f"Failed to write settings file: {e}")
            return False

    @classmethod
    def save_settings(cls, new_settings):
        """Save new settings, merging with existing settings."""
        try:
            # Get current settings (from cache or file)
            if cls._settings is None:
                cls.load_settings()
            
            # Merge new settings into current
            current = cls._settings.copy() if cls._settings else cls._get_defaults()
            current.update(new_settings)
            
            # Write to file
            if cls._write_to_file(current):
                cls._settings = current
                return True
            return False
        except Exception as e:
            logging.error(f"Failed to save settings: {e}")
            return False

    @classmethod
    def get_setting(cls, key, default=None):
        """Get a specific setting value."""
        if cls._settings is None:
            cls.load_settings()
        return cls._settings.get(key, default)

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

