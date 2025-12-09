
import json
import os
import logging

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'settings.json')

class SettingsManager:
    _settings = None

    @classmethod
    def load_settings(cls):
        if not os.path.exists(SETTINGS_FILE):
             # Default settings
             cls._settings = {
                "active_learning_mode": "ML_MODE",
                "prompt_style": "parameter-format",
                "hybrid_weights": {
                    "w_llm": 0.5,
                    "w_ml": 0.5
                },
                "alpha_value": 1.0
             }
             cls.save_settings(cls._settings)
             return cls._settings

        try:
            with open(SETTINGS_FILE, 'r') as f:
                cls._settings = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load settings: {e}")
            cls._settings = {} # Fallback
        
        return cls._settings

    @classmethod
    def save_settings(cls, new_settings):
        try:
            # Merge with existing
            current = cls.load_settings()
            current.update(new_settings)
            
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(current, f, indent=4)
            
            cls._settings = current
            return True
        except Exception as e:
            logging.error(f"Failed to save settings: {e}")
            return False

    @classmethod
    def get_setting(cls, key, default=None):
        if cls._settings is None:
            cls.load_settings()
        return cls._settings.get(key, default)
