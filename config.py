# config.py

import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration settings."""
    # Core Flask & Security
    SECRET_KEY = os.environ.get('JWT_SECRET_KEY')
    ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME')
    ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD')
    
    # API Keys
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')

    # JWT Settings
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=int(os.environ.get('JWT_EXPIRATION_HOURS', 1)))

    # Custom App Settings
    SIMILARITY_THRESHOLD = float(os.environ.get('SIMILARITY_THRESHOLD', 0.85))
    
    # --- Specific Model Name Configuration ---
    GEMINI_EMBEDDING_MODEL = os.environ.get('GEMINI_EMBEDDING_MODEL')
    OPENAI_EMBEDDING_MODEL = os.environ.get('OPENAI_EMBEDDING_MODEL')

    GEMINI_REASONING_MODEL = os.environ.get('GEMINI_REASONING_MODEL')
    OPENAI_REASONING_MODEL = os.environ.get('OPENAI_REASONING_MODEL')
    DEEPSEEK_REASONING_MODEL = os.environ.get('DEEPSEEK_REASONING_MODEL')
    
    # CORS & Rate Limiting
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    RATELIMIT_STORAGE_URI = os.environ.get('RATELIMIT_STORAGE_URI',"memory://")