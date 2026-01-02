import logging
from logging.handlers import RotatingFileHandler
import os
from flask import Flask, jsonify, current_app
from flask_jwt_extended import JWTManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_marshmallow import Marshmallow
from flask_cors import CORS
from config import Config
from openai import OpenAI
from google import genai

jwt = JWTManager()
ma = Marshmallow()
cors = CORS()
limiter = Limiter(key_func=get_remote_address)


openai_client = None 
deepseek_client = None 
gemini_client = None


def create_app():
    """Create and configure the Flask application."""
    
    app = Flask(__name__)
    app.config.from_object(Config)
    
    if not app.debug and not app.testing:
        os.makedirs("logs", exist_ok=True)
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        )
        file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.addHandler(stream_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('Application startup')

    jwt.init_app(app)
    ma.init_app(app)
    limiter.init_app(app)
    cors.init_app(app, origins="*")

    with app.app_context():
        global gemini_client, openai_client, deepseek_client
        
        try:
            google_api_key = current_app.config.get('GOOGLE_API_KEY')
            if not google_api_key:
                app.logger.warning("GOOGLE_API_KEY not found. Gemini provider will be unavailable.")
            else:
                gemini_client = genai.Client(api_key=google_api_key)
                app.logger.info("Google AI (Gemini) SDK configured.")
        except Exception as e:
            app.logger.error(f"FATAL: Google AI Configuration Error: {e}")

        openai_api_key = current_app.config.get('OPENAI_API_KEY')
        if not openai_api_key:
            app.logger.warning("OPENAI_API_KEY not found. OpenAI provider will be unavailable.")
        else:
            openai_client = OpenAI(api_key=openai_api_key)
            app.logger.info("OpenAI client initialized.")

        deepseek_api_key = current_app.config.get('DEEPSEEK_API_KEY')
        if not deepseek_api_key:
            app.logger.warning("DEEPSEEK_API_KEY not found. DeepSeek provider will be unavailable.")
        else:
            deepseek_client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
            app.logger.info("DeepSeek client initialized.")


    from .auth import auth_bp
    from .api import api_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(api_bp)

    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        return jsonify({"error": "Access token has expired"}), 401

    @jwt.invalid_token_loader
    def invalid_token_callback(error):
        return jsonify({"error": "Invalid access token"}), 401

    @jwt.unauthorized_loader
    def missing_token_callback(error):
        return jsonify({"error": "Authorization header is missing"}), 401

    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({"error": "This resource was not found"}), 404

    @app.errorhandler(500)
    def internal_server_error(error):
        app.logger.error(f"Internal Server Error: {error}")
        return jsonify({"error": "An internal server error occurred"}), 500

    return app