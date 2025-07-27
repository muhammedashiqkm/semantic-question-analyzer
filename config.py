import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration settings."""
    SECRET_KEY = os.environ.get('JWT_SECRET_KEY')
    ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME')
    ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD')
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

    # Token expiration can be set via environment variable
    JWT_EXPIRATION_HOURS = int(os.environ.get('JWT_EXPIRATION_HOURS', 1))
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=JWT_EXPIRATION_HOURS)

    # Similarity threshold can be set via environment variable
    SIMILARITY_THRESHOLD = float(os.environ.get('SIMILARITY_THRESHOLD', 0.85))

    # Embedding model name can be set via environment variable
    EMBEDDING_MODEL_NAME = os.environ.get('EMBEDDING_MODEL_NAME', 'text-embedding-004')

    # CORS origins can be set via a comma-separated environment variable
    CORS_ORIGINS_STRING = os.environ.get('CORS_ORIGINS', '*')
    CORS_ORIGINS = CORS_ORIGINS_STRING.split(',')