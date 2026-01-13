import os
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()

class Config:
    """Flask application configuration"""
    
    # Security - SECRET_KEY must be set in environment variables or .env file
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("No SECRET_KEY set. Create a .env file with: SECRET_KEY=your-secure-key")
    
    # Database
    BASEDIR = os.path.abspath(os.path.dirname(__file__))
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(BASEDIR, 'data', 'results.db')}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
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