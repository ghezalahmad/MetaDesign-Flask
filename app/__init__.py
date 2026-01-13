from flask import Flask
import os
import logging

# Rate limiter instance (initialized in create_app)
limiter = None

def create_app():
    global limiter
    logging.debug("Inside create_app function")
    
    # Setup template and static directories (your existing code)
    template_dir = os.path.abspath(os.path.dirname(__file__))
    template_dir = os.path.join(template_dir, '..', 'templates')
    static_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'static')
    
    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
    
    # Load config from config.py (NEW)
    app.config.from_object('config.Config')
    
    # Override with your existing secret key if needed
    # app.config['SECRET_KEY'] = 'super-secret-key'  # Already set in Config

    # Create data directories if they don't exist (your existing code)
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    designspaces_dir = os.path.join(data_dir, 'designspaces')
    os.makedirs(designspaces_dir, exist_ok=True)

    logging.debug("Flask app object created")
    
    # Initialize SQLAlchemy database
    from app.database import init_db
    init_db(app)
    logging.debug("✅ SQLite database initialized")
    
    # Add response compression for large JSON payloads (NEW - OPTIONAL)
    try:
        from flask_compress import Compress
        Compress(app)
        logging.debug("✅ Flask-Compress enabled")
    except ImportError:
        logging.warning("⚠️ flask-compress not installed. Install with: pip install flask-compress")

    # Add rate limiting to prevent DoS attacks
    try:
        from flask_limiter import Limiter
        from flask_limiter.util import get_remote_address
        
        limiter = Limiter(
            get_remote_address,
            app=app,
            default_limits=["200 per day", "50 per hour"],
            storage_uri="memory://",
        )
        logging.debug("✅ Flask-Limiter enabled (200/day, 50/hour default)")
    except ImportError:
        logging.warning("⚠️ flask-limiter not installed. Install with: pip install flask-limiter")

    with app.app_context():
        # Import and register main blueprint (core views)
        logging.debug("Importing routes")
        from . import routes
        logging.debug("Registering main blueprint")
        app.register_blueprint(routes.main_bp)
        
        # Register API blueprints
        logging.debug("Registering API blueprints")
        from app.api import all_blueprints
        for bp in all_blueprints:
            app.register_blueprint(bp)
            logging.debug(f"  ✅ Registered: {bp.name}")
        
        logging.debug("All blueprints registered")

    logging.debug("Returning Flask app")
    logging.info("✅ Flask app initialized successfully")
    
    return app