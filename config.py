import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration settings."""

    # Where this service listens.
    #
    # One place decides it: run.py reads it for a development run, the
    # Dockerfile binds Gunicorn to it, and docker-compose publishes it. Change
    # PORT in .env and all three follow - which is what stops a service from
    # listening on one port while its container publishes another.
    HOST = os.environ.get("HOST", "127.0.0.1")
    PORT = int(os.environ.get("PORT", 5009))

    # Who may call this service.
    #
    # One fixed key, sent as X-API-Key. It replaced a username and password
    # that the browser had to hold in order to sign in - see app/security.py.
    # Empty means every caller is accepted, which suits only a machine nothing
    # else can reach.
    API_KEY = os.environ.get("API_KEY", "")

    # Model providers. A provider with no key is simply unavailable, /health
    # says so, and a request naming it is refused with a clear message.
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

    # How close two questions must be before a model is asked to judge them.
    # Below this they are not candidates at all; above it, the model decides.
    SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", 0.85))

    # Which model each provider uses, per role.
    GEMINI_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL")
    OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL")

    GEMINI_REASONING_MODEL = os.environ.get("GEMINI_REASONING_MODEL")
    OPENAI_REASONING_MODEL = os.environ.get("OPENAI_REASONING_MODEL")
    DEEPSEEK_REASONING_MODEL = os.environ.get("DEEPSEEK_REASONING_MODEL")

    CORS_ORIGINS = [origin for origin in os.environ.get("CORS_ORIGINS", "*").split(",") if origin]
