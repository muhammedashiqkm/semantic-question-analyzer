import logging
import numpy as np
from flask import request, jsonify, Blueprint, current_app
from flask_jwt_extended import jwt_required
from marshmallow import ValidationError
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from .helpers import fetch_questions_from_url, clean_html, get_embeddings
from .schemas import SimilarityCheckSchema, GroupingSchema

api_bp = Blueprint('api', __name__)

similarity_schema = SimilarityCheckSchema()
grouping_schema = GroupingSchema()

@api_bp.route('/check_similarity', methods=['POST'])
@jwt_required()
def check_similarity():
    """
    Checks if a new question is semantically similar to any in a list from a URL.
    Requires JWT authentication.
    """
    try:
        data = similarity_schema.load(request.get_json())
    except ValidationError as err:
        return jsonify(err.messages), 400

    questions_url = data['questions_url']
    new_question_text = data['question']
    
    logging.info(f"Checking similarity for question at URL: {questions_url}")

    existing_questions = fetch_questions_from_url(questions_url)
    if existing_questions is None:
        return jsonify({"error": "Failed to fetch or parse questions from the provided URL."}), 500

    if not existing_questions:
        return jsonify({"response": "no", "reason": "No existing questions found at URL."})

    existing_questions_text = [clean_html(q.get('Question', '')) for q in existing_questions]
    all_texts = [new_question_text] + existing_questions_text

    embeddings = get_embeddings(all_texts)
    if embeddings is None:
        return jsonify({"error": "Failed to generate text embeddings."}), 500

    new_question_embedding = np.array([embeddings[0]])
    existing_questions_embeddings = np.array(embeddings[1:])

    similarities = cosine_similarity(new_question_embedding, existing_questions_embeddings)[0]
    
    # Use threshold from app config
    threshold = current_app.config['SIMILARITY_THRESHOLD']
    matched_questions = [
        existing_questions[i] for i, score in enumerate(similarities) if score >= threshold
    ]
    
    if matched_questions:
        return jsonify({"response": "yes", "matched_questions": matched_questions})
    else:
        return jsonify({"response": "no"})


@api_bp.route('/group_similar_questions', methods=['POST'])
@jwt_required()
def group_similar_questions():
    """
    Fetches questions from a URL and groups them into semantically similar clusters.
    Requires JWT authentication.
    """
    try:
        data = grouping_schema.load(request.get_json())
    except ValidationError as err:
        return jsonify(err.messages), 400
        
    questions_url = data['questions_url']
    logging.info(f"Grouping questions from URL: {questions_url}")

    questions = fetch_questions_from_url(questions_url)
    if questions is None:
        return jsonify({"error": "Failed to fetch or parse questions from the provided URL."}), 500

    if not questions or len(questions) < 2:
        return jsonify({"response": "no", "reason": "Not enough questions to form a group."})

    questions_text = [clean_html(q.get('Question', '')) for q in questions]
    embeddings = get_embeddings(questions_text)
    if embeddings is None:
        return jsonify({"error": "Failed to generate text embeddings."}), 500

    # Use threshold from app config
    threshold = current_app.config['SIMILARITY_THRESHOLD']
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='cosine',
        linkage='average',
        distance_threshold=(1 - threshold)
    ).fit(embeddings)

    labels = clustering.labels_
    
    groups = {}
    for i, label in enumerate(labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(questions[i])

    matched_groups = [group for group in groups.values() if len(group) > 1]

    if matched_groups:
        return jsonify({"response": "yes", "matched_groups": matched_groups})
    else:
        return jsonify({"response": "no"})