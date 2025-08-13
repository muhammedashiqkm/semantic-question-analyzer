import logging
from logging.handlers import RotatingFileHandler
import os
from flask import Flask, jsonify
from flask_jwt_extended import JWTManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_marshmallow import Marshmallow
from flask_cors import CORS
from config import Config
import google.generativeai as genai

# --- Initialize Extensions ---
jwt = JWTManager()
ma = Marshmallow()
cors = CORS()
limiter = Limiter(
    key_func=get_remote_address,
)

# --- Application Factory ---
def create_app():
    """Create and configure the Flask application."""
    
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # --- Logging Configuration ---
    if not app.debug and not app.testing:
        os.makedirs("logs", exist_ok=True)
        file_handler = RotatingFileHandler(
            'logs/app.log', 
            maxBytes=10240, 
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('Application startup')

    # --- Initialize Extensions with App ---
    jwt.init_app(app)
    ma.init_app(app)
    limiter.init_app(app)
    cors.init_app(app, origins=app.config['CORS_ORIGINS'])

    # --- Configure Google AI ---
    try:
        api_key = app.config['GOOGLE_API_KEY']
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)
        app.logger.info("Google AI SDK configured successfully.")
    except (ValueError, KeyError) as e:
        app.logger.error(f"FATAL: Google AI Configuration Error: {e}")
        raise

    # --- Register Blueprints ---
    from .auth import auth_bp
    from .api import api_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(api_bp)

    # --- Custom Error Handlers ---
    
    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        """Custom JSON response for expired JWT tokens."""
        return jsonify({"error": "Access token has expired"}), 401

    @jwt.invalid_token_loader
    def invalid_token_callback(error):
        """Custom JSON response for invalid JWT tokens."""
        return jsonify({"error": "Invalid access token"}), 401

    @jwt.unauthorized_loader
    def missing_token_callback(error):
        """Custom JSON response for requests missing the Authorization header."""
        return jsonify({"error": "Authorization header is missing"}), 401

    @app.errorhandler(404)
    def not_found_error(error):
        """Custom JSON response for 404 Not Found errors."""
        return jsonify({"error": "This resource was not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed_error(error):
        """Custom JSON response for 405 Method Not Allowed errors."""
        return jsonify({"error": "The method is not allowed for the requested URL"}), 405

    @app.errorhandler(500)
    def internal_server_error(error):
        """Custom JSON response for 500 Internal Server errors."""
        app.logger.error(f"Internal Server Error: {error}")
        return jsonify({"error": "An internal server error occurred"}), 500

    return app
