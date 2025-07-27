from . import ma

class LoginSchema(ma.Schema):
    """Schema for login request."""
    username = ma.Str(required=True)
    password = ma.Str(required=True)

class SimilarityCheckSchema(ma.Schema):
    """Schema for similarity check request."""
    questions_url = ma.URL(required=True)
    question = ma.Str(required=True)

class GroupingSchema(ma.Schema):
    """Schema for question grouping request."""
    questions_url = ma.URL(required=True)