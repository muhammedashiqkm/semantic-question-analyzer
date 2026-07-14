from marshmallow import validate
from . import ma


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
    embedding_provider = ma.Str(required=True, validate=validate.OneOf(SUPPORTED_EMBEDDING_PROVIDERS))
    reasoning_provider = ma.Str(required=True, validate=validate.OneOf(SUPPORTED_REASONING_PROVIDERS))

class GroupingSchema(ma.Schema):
    """Schema for question grouping request."""
    questions_url = ma.URL(required=True)
    embedding_provider = ma.Str(required=True, validate=validate.OneOf(SUPPORTED_EMBEDDING_PROVIDERS))
    reasoning_provider = ma.Str(required=True, validate=validate.OneOf(SUPPORTED_REASONING_PROVIDERS))
    
    
class LatexConversionSchema(ma.Schema):
    """Schema for HTML to LaTeX conversion request."""
    html_contents = ma.List(ma.Str(), required=True, validate=validate.Length(min=1))
    reasoning_provider = ma.Str(required=False, validate=validate.OneOf(SUPPORTED_REASONING_PROVIDERS), load_default="gemini")
