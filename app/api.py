import logging
import numpy as np
from typing import Tuple, Dict, Any, Optional

from flask import request, jsonify, Blueprint, current_app, Response
from flask_jwt_extended import jwt_required
from marshmallow import ValidationError
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

from .helpers import (
    fetch_questions_from_url, clean_html, get_embeddings,
    validate_question_quality, AIServiceUnavailableError
)
from .schemas import SimilarityCheckSchema, GroupingSchema
from . import limiter

api_bp = Blueprint('api', __name__)
similarity_schema = SimilarityCheckSchema()
grouping_schema = GroupingSchema()


JsonResponse = Tuple[Response, int]

def get_model_from_provider(provider_type: str, provider_name: str) -> Optional[str]:
    """Looks up the configured model name for a given provider."""
    key = f"{provider_name.upper()}_{provider_type.upper()}_MODEL"
    model_name = current_app.config.get(key)
    return str(model_name) if model_name else None

@api_bp.route('/health', methods=['GET'])
def health_check() -> JsonResponse:
    """Provides a simple health check endpoint."""
    return jsonify({"status": "api_healthy"}), 200

@api_bp.route('/check_similarity', methods=['POST'])
@jwt_required()
@limiter.limit("10 per minute")
def check_similarity() -> JsonResponse:
    """Checks a new question for similarity against a list of existing questions."""
    try:
        data: Dict[str, Any] = similarity_schema.load(request.get_json())
    except ValidationError as err:
        return jsonify(err.messages), 400

    embedding_provider = data['embedding_provider']
    reasoning_provider = data['reasoning_provider']

    embedding_model = get_model_from_provider('embedding', embedding_provider)
    reasoning_model = get_model_from_provider('reasoning', reasoning_provider)

    if not all([embedding_model, reasoning_model]):
        return jsonify({"error": "Server configuration error: model name not found for a specified provider."}), 500

    try:
        # Step 1: Validate incoming question quality
        is_question_valid = validate_question_quality(data['question'], reasoning_provider, reasoning_model)
        if not is_question_valid:
            return jsonify({"error": "Invalid or poor-quality question provided."}), 400
        
        # Step 2: Fetch and process existing questions
        existing_questions = fetch_questions_from_url(data['questions_url'])
        if existing_questions is None:
            return jsonify({"error": "Resource not found at URL or could not be parsed."}), 404
        if not existing_questions:
            return jsonify({"response": "no", "reason": "No existing questions to compare against."}), 200

        # Step 3: Generate embeddings
        existing_questions_text = [clean_html(q.get('Question', '')) for q in existing_questions]
        all_texts = [data['question']] + existing_questions_text
        
        embeddings = get_embeddings(all_texts, provider=embedding_provider, model_name=embedding_model)
        if not embeddings:
            return jsonify({"error": "Failed to generate embeddings."}), 500

        # Step 4: Calculate similarity
        new_q_embedding = np.array([embeddings[0]])
        existing_q_embeddings = np.array(embeddings[1:])
        similarities = cosine_similarity(new_q_embedding, existing_q_embeddings)[0]

        threshold = float(current_app.config['SIMILARITY_THRESHOLD'])
        matched_questions = [
            existing_questions[i] for i, score in enumerate(similarities) if score >= threshold
        ]
        
        if matched_questions:
            return jsonify({"response": "yes", "matched_questions": matched_questions}), 200
        else:
            return jsonify({"response": "no"}), 200

    except AIServiceUnavailableError as e:
        return jsonify({"error": str(e)}), 503
    except Exception:
        logging.error("An unexpected error occurred in check_similarity", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500


@api_bp.route('/group_similar_questions', methods=['POST'])
@jwt_required()
@limiter.limit("5 per minute")
def group_similar_questions() -> JsonResponse:
    """Groups a list of questions by semantic similarity."""
    try:
        data: Dict[str, Any] = grouping_schema.load(request.get_json())
    except ValidationError as err:
        return jsonify(err.messages), 400

    embedding_provider = data['embedding_provider']
    embedding_model = get_model_from_provider('embedding', embedding_provider)

    if not embedding_model:
        return jsonify({"error": "Server configuration error: model name not found for the specified provider."}), 500

    try:
        questions = fetch_questions_from_url(data['questions_url'])
        if not questions or len(questions) < 2:
            return jsonify({"response": "no", "reason": "Not enough questions to form a group."}), 200

        questions_text = [clean_html(q.get('Question', '')) for q in questions]
        embeddings = get_embeddings(questions_text, provider=embedding_provider, model_name=embedding_model)

        if not embeddings:
            return jsonify({"error": "Failed to generate embeddings."}), 500

        distance_threshold = 1 - float(current_app.config['SIMILARITY_THRESHOLD'])
        clustering = AgglomerativeClustering(
            n_clusters=None, metric='cosine', linkage='average', distance_threshold=distance_threshold
        ).fit(embeddings)

        groups: Dict[int, list] = {}
        for i, label in enumerate(clustering.labels_):
            groups.setdefault(int(label), []).append(questions[i])

        matched_groups = [group for group in groups.values() if len(group) > 1]

        if matched_groups:
            return jsonify({"response": "yes", "matched_groups": matched_groups}), 200
        else:
            return jsonify({"response": "no"}), 200

    except AIServiceUnavailableError as e:
        return jsonify({"error": str(e)}), 503
    except Exception:
        logging.error("An unexpected error occurred in group_similar_questions", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500