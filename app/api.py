# app/api.py

import logging
import numpy as np
from flask import request, jsonify, Blueprint, current_app
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

def get_model_from_provider(provider_type, provider_name):
    """Looks up the configured model name for a given provider."""
    key = f"{provider_name.upper()}_{provider_type.upper()}_MODEL"
    return current_app.config.get(key)

@api_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "api_healthy"}), 200

@api_bp.route('/check_similarity', methods=['POST'])
@jwt_required()
@limiter.limit("10 per minute")
def check_similarity():
    try:
        data = similarity_schema.load(request.get_json())
    except ValidationError as err:
        return jsonify(err.messages), 400

    # Get the required providers from the request
    embedding_provider = data['embedding_provider']
    reasoning_provider = data['reasoning_provider']
    
    # Look up the specific model names from the app config
    embedding_model = get_model_from_provider('embedding', embedding_provider)
    reasoning_model = get_model_from_provider('reasoning', reasoning_provider)
    
    if not all([embedding_model, reasoning_model]):
        return jsonify({"error": "Server configuration error: model name not found for a specified provider."}), 500
        
    try:
        # Step 1: Validate incoming question quality
        validation = validate_question_quality(data['question'], reasoning_provider, reasoning_model)
        if not validation.get('is_valid'):
            return jsonify({"error": "Invalid question provided", "reason": validation.get('reason')}), 400

        # Step 2: Fetch and process existing questions
        existing_questions = fetch_questions_from_url(data['questions_url'])
        if existing_questions is None:
            return jsonify({"error": "Resource not found at URL or could not be parsed."}), 404
        if not existing_questions:
            return jsonify({"response": "no", "reason": "No existing questions to compare against."})

        # Step 3: Generate embeddings
        existing_questions_text = [clean_html(q.get('Question', '')) for q in existing_questions]
        all_texts = [data['question']] + existing_questions_text
        embeddings = get_embeddings(all_texts, provider=embedding_provider, model_name=embedding_model)

        # Step 4: Calculate similarity
        new_q_embedding = np.array([embeddings[0]])
        existing_q_embeddings = np.array(embeddings[1:])
        similarities = cosine_similarity(new_q_embedding, existing_q_embeddings)[0]
        
        threshold = current_app.config['SIMILARITY_THRESHOLD']
        matched_questions = [
            existing_questions[i] for i, score in enumerate(similarities) if score >= threshold
        ]

        return jsonify({"response": "yes", "matched_questions": matched_questions}) if matched_questions else jsonify({"response": "no"})

    except AIServiceUnavailableError as e:
        return jsonify({"error": "AI service provider is currently unavailable. Please try again later."}), 503
    except Exception as e:
        logging.error(f"An unexpected error occurred in check_similarity: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

@api_bp.route('/group_similar_questions', methods=['POST'])
@jwt_required()
@limiter.limit("5 per minute")
def group_similar_questions():
    try:
        data = grouping_schema.load(request.get_json())
    except ValidationError as err:
        return jsonify(err.messages), 400

    embedding_provider = data['embedding_provider']
    embedding_model = get_model_from_provider('embedding', embedding_provider)

    if not embedding_model:
        return jsonify({"error": "Server configuration error: model name not found for the specified provider."}), 500

    try:
        questions = fetch_questions_from_url(data['questions_url'])
        if not questions or len(questions) < 2:
            return jsonify({"response": "no", "reason": "Not enough questions to form a group."})

        questions_text = [clean_html(q.get('Question', '')) for q in questions]
        embeddings = get_embeddings(questions_text, provider=embedding_provider, model_name=embedding_model)

        distance_threshold = 1 - current_app.config['SIMILARITY_THRESHOLD']
        clustering = AgglomerativeClustering(
            n_clusters=None, metric='cosine', linkage='average', distance_threshold=distance_threshold
        ).fit(embeddings)

        groups = {}
        for i, label in enumerate(clustering.labels_):
            groups.setdefault(label, []).append(questions[i])

        matched_groups = [group for group in groups.values() if len(group) > 1]
        
        return jsonify({"response": "yes", "matched_groups": matched_groups}) if matched_groups else jsonify({"response": "no"})

    except AIServiceUnavailableError as e:
        return jsonify({"error": "AI service provider is currently unavailable. Please try again later."}), 503
    except Exception as e:
        logging.error(f"An unexpected error occurred in group_similar_questions: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500