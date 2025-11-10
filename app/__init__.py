from flask import Flask
import os
import logging

def create_app():
    logging.debug("Inside create_app function")
    template_dir = os.path.abspath(os.path.dirname(__file__))
    template_dir = os.path.join(template_dir, '..', 'templates')
    static_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'static')
    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
    app.config['SECRET_KEY'] = 'super-secret-key'
    logging.debug("Flask app object created")

    with app.app_context():
        # Import and register blueprints
        logging.debug("Importing routes")
        from . import routes
        logging.debug("Registering blueprint")
        app.register_blueprint(routes.main_bp)
        logging.debug("Blueprint registered")

    logging.debug("Returning Flask app")
    return app
