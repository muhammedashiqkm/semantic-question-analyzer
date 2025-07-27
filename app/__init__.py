import logging
from logging.handlers import RotatingFileHandler
import os
from flask import Flask
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
    get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# --- Application Factory ---
def create_app():
    """Create and configure the Flask application."""
    
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # --- Logging Configuration ---
    if not app.debug and not app.testing:
        # Create a log file handler
        if not os.path.exists('logs'):
            os.mkdir('logs')
        file_handler = RotatingFileHandler(
            'logs/app.log', 
            maxBytes=10240, 
            backupCount=10
        )
        
        # Set the log format
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        
        # Set the log level
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
        # Log the error, but don't exit. Let the server fail to start gracefully.
        app.logger.error(f"FATAL: Google AI Configuration Error: {e}")
        raise  # Re-raise the exception to prevent the app from starting

    # --- Register Blueprints ---
    from .auth import auth_bp
    from .api import api_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(api_bp)

    return app