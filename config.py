import os

class Config:
    """Flask application configuration"""
    
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'super-secret-key'
    
    # JSON serialization optimizations
    JSON_SORT_KEYS = False
    JSONIFY_PRETTYPRINT_REGULAR = False  # Faster serialization for large payloads
    
    # File upload
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    
    # Performance
    SEND_FILE_MAX_AGE_DEFAULT = 0
    PROPAGATE_EXCEPTIONS = True
    
    # Session
    SESSION_TYPE = 'filesystem'