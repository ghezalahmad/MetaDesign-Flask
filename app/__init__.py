from flask import Flask
import os

def create_app():
    print("Creating Flask app")
    template_dir = os.path.abspath(os.path.dirname(__file__))
    template_dir = os.path.join(template_dir, '..', 'templates')
    static_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'static')
    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
    app.config['SECRET_KEY'] = 'super-secret-key'

    with app.app_context():
        # Import and register blueprints
        from . import routes
        app.register_blueprint(routes.main_bp)

    return app
