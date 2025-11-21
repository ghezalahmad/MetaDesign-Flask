from flask import Flask
import os
import logging

def create_app():
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
    
    # Add response compression for large JSON payloads (NEW - OPTIONAL)
    try:
        from flask_compress import Compress
        Compress(app)
        logging.debug("✅ Flask-Compress enabled")
    except ImportError:
        logging.warning("⚠️ flask-compress not installed. Install with: pip install flask-compress")

    with app.app_context():
        # Import and register blueprints (your existing code)
        logging.debug("Importing routes")
        from . import routes
        logging.debug("Registering blueprint")
        app.register_blueprint(routes.main_bp)
        logging.debug("Blueprint registered")

    logging.debug("Returning Flask app")
    logging.info("✅ Flask app initialized successfully")
    
    return app