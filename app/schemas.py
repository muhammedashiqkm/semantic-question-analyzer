from marshmallow import validate
from . import ma

# --- Define supported models and providers for validation ---
SUPPORTED_EMBEDDING_PROVIDERS = ["gemini", "openai"]
SUPPORTED_REASONING_PROVIDERS = ["gemini", "openai", "deepseek"]

class LoginSchema(ma.Schema):
    """Schema for login request."""
    username = ma.Str(required=True)
    password = ma.Str(required=True)

class SimilarityCheckSchema(ma.Schema):
    """Schema for similarity check request."""
    questions_url = ma.URL(required=True)
    question = ma.Str(required=True)
    
    # --- Optional and Validated Fields ---
    embedding_provider = ma.Str(required=False, validate=validate.OneOf(SUPPORTED_EMBEDDING_PROVIDERS))
    embedding_model = ma.Str(required=False)
    reasoning_provider = ma.Str(required=False, validate=validate.OneOf(SUPPORTED_REASONING_PROVIDERS))
    reasoning_model = ma.Str(required=False)

class GroupingSchema(ma.Schema):
    """Schema for question grouping request."""
    questions_url = ma.URL(required=True)

    # --- Optional and Validated Fields ---
    embedding_provider = ma.Str(required=False, validate=validate.OneOf(SUPPORTED_EMBEDDING_PROVIDERS))
    embedding_model = ma.Str(required=False)