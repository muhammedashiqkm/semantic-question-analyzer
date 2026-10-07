"""
The semantic question analyzer.

One job: given a question a teacher is writing and the bank it belongs to, say
whether that bank already holds the same question. It embeds both sides, keeps
what is close enough, and asks a model to confirm each candidate - so a near
match is a judgement, not a number above a threshold.

Callers authenticate with a fixed key (see security.py). There is no login, no
session and no token.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from flask import Flask, current_app, jsonify
from flask_cors import CORS
from flask_marshmallow import Marshmallow
from google import genai
from openai import OpenAI

from config import Config

ma = Marshmallow()
cors = CORS()

openai_client = None
deepseek_client = None
gemini_client = None


def _configure_logging(app: Flask) -> None:
    if app.debug or app.testing:
        return

    os.makedirs("logs", exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]"
    )

    file_handler = RotatingFileHandler("logs/app.log", maxBytes=10240, backupCount=10)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    app.logger.addHandler(file_handler)
    app.logger.addHandler(stream_handler)
    app.logger.setLevel(logging.INFO)


def _configure_models(app: Flask) -> None:
    """
    Builds a client for every provider that has a key.

    A missing key is a warning, not a failure: a deployment may be set up for
    one provider alone, and every request names the provider it wants. /health
    reports which are actually available.
    """
    global gemini_client, openai_client, deepseek_client

    google_api_key = current_app.config.get("GOOGLE_API_KEY")
    if google_api_key:
        try:
            gemini_client = genai.Client(api_key=google_api_key)
            app.logger.info("Google AI (Gemini) client ready.")
        except Exception as error:
            app.logger.error("Google AI configuration failed: %s", error)
    else:
        app.logger.warning("GOOGLE_API_KEY is not set - Gemini is unavailable.")

    openai_api_key = current_app.config.get("OPENAI_API_KEY")
    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)
        app.logger.info("OpenAI client ready.")
    else:
        app.logger.warning("OPENAI_API_KEY is not set - OpenAI is unavailable.")

    deepseek_api_key = current_app.config.get("DEEPSEEK_API_KEY")
    if deepseek_api_key:
        deepseek_client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
        app.logger.info("DeepSeek client ready.")
    else:
        app.logger.warning("DEEPSEEK_API_KEY is not set - DeepSeek is unavailable.")


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_object(Config)

    _configure_logging(app)

    ma.init_app(app)
    cors.init_app(app, origins=app.config.get("CORS_ORIGINS") or ["*"])

    with app.app_context():
        _configure_models(app)

    from .api import api_bp

    app.register_blueprint(api_bp)

    if not app.config.get("API_KEY"):
        app.logger.warning(
            "API_KEY is not set - every caller is accepted. Set it once the portal has it."
        )

    # The same shape the endpoints use, so a caller reads every failure the
    # same way - including the ones Flask raises before any endpoint runs.
    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({
            "error": "Not Found",
            "message": "No such endpoint. This service has /health, /check_similarity, "
                       "/group_similar_questions and /convert-to-latex.",
        }), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            "error": "Method Not Allowed",
            "message": "That endpoint exists but not for this HTTP method.",
        }), 405

    @app.errorhandler(500)
    def internal_server_error(error):
        app.logger.error("Internal server error: %s", error)
        return jsonify({
            "error": "Internal Server Error",
            "message": "An internal server error occurred.",
        }), 500

    app.logger.info("Application startup")
    return app
