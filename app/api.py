import logging
import numpy as np
from typing import Tuple, Dict, Any, Optional

from flask import request, jsonify, Blueprint, current_app, Response
from marshmallow import ValidationError
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

from .helpers import (
    fetch_questions_from_url, clean_html, get_embeddings,
    verify_matches_with_llm, AIServiceUnavailableError,
    convert_html_to_latex_with_llm
)
from .schemas import SimilarityCheckSchema, GroupingSchema, LatexConversionSchema
from .security import require_api_key
from .errors import (JsonResponse, bad_request, internal, invalid_payload,
                     unavailable, upstream_failed)


api_bp = Blueprint('api', __name__)
similarity_schema = SimilarityCheckSchema()
grouping_schema = GroupingSchema()
latex_schema = LatexConversionSchema()


def get_model_from_provider(provider_type: str, provider_name: str) -> Optional[str]:
    """Looks up the configured model name for a given provider."""
    key = f"{provider_name.upper()}_{provider_type.upper()}_MODEL"
    model_name = current_app.config.get(key)
    return str(model_name) if model_name else None


def providers_for(provider_type: str) -> list:
    """Which providers this deployment can actually use for a role."""
    known = ("gemini", "openai", "deepseek")
    return [name for name in known if get_model_from_provider(provider_type, name)]


def unusable_provider(provider_type: str, provider_name: str) -> Optional[JsonResponse]:
    """
    Says why a named provider cannot be used, or nothing if it can.

    Two different answers, which used to be one 500 between them: a provider
    this service has never heard of is the caller's mistake and no amount of
    retrying will help, while a provider that exists but has no key here is
    this deployment's gap and may be fixed by the time they try again.
    """
    if get_model_from_provider(provider_type, provider_name):
        return None

    available = providers_for(provider_type)
    if not available:
        return unavailable(
            f"No {provider_type} provider is configured on this service. "
            "Set its API key and model name, then try again."
        )

    return bad_request(
        f"'{provider_name}' is not a provider this service can use for {provider_type}. "
        f"It has: {', '.join(available)}."
    )

@api_bp.route('/health', methods=['GET'])
def health_check() -> JsonResponse:
    """
    Whether this service can actually do its work.

    Open, so a monitor can reach it without the key - it names what is
    configured and never the keys themselves. "healthy" says the process is up;
    "providers" is the part worth reading, because a provider with no key is
    unavailable and a request naming it will be refused.
    """
    from . import deepseek_client, gemini_client, openai_client

    embedding = {
        "gemini": bool(gemini_client) and bool(current_app.config.get("GEMINI_EMBEDDING_MODEL")),
        "openai": bool(openai_client) and bool(current_app.config.get("OPENAI_EMBEDDING_MODEL")),
    }
    reasoning = {
        "gemini": bool(gemini_client) and bool(current_app.config.get("GEMINI_REASONING_MODEL")),
        "openai": bool(openai_client) and bool(current_app.config.get("OPENAI_REASONING_MODEL")),
        "deepseek": bool(deepseek_client) and bool(current_app.config.get("DEEPSEEK_REASONING_MODEL")),
    }

    return jsonify({
        "status": "healthy",
        "service": "semantic-question-analyzer",
        "version": "1.0.0",
        "api_key_required": bool(current_app.config.get("API_KEY")),
        "similarity_threshold": current_app.config.get("SIMILARITY_THRESHOLD"),
        "providers": {
            "embedding": [name for name, ready in embedding.items() if ready],
            "reasoning": [name for name, ready in reasoning.items() if ready],
        },
        "usable": any(embedding.values()) and any(reasoning.values()),
    }), 200

@api_bp.route('/check_similarity', methods=['POST'])
@require_api_key
def check_similarity() -> JsonResponse:
    """Checks a new question for similarity against a list of existing questions."""
    try:
        data: Dict[str, Any] = similarity_schema.load(request.get_json())
    except ValidationError as err:
        return invalid_payload(err.messages)

    embedding_provider = data['embedding_provider']
    reasoning_provider = data['reasoning_provider']

    refused = (unusable_provider('embedding', embedding_provider)
               or unusable_provider('reasoning', reasoning_provider))
    if refused:
        return refused

    embedding_model = get_model_from_provider('embedding', embedding_provider)
    reasoning_model = get_model_from_provider('reasoning', reasoning_provider)

    try:
        existing_questions = fetch_questions_from_url(data['questions_url'])
        if existing_questions is None:
            # The bank is fetched from a URL the CALLER gives, so this is not a
            # 404 of ours - answering with one reads as "no such endpoint".
            return upstream_failed(
                "The question bank could not be read from questions_url. Check that the "
                "address is right and reachable from this service."
            )
        if not existing_questions:
            return jsonify({"response": "no", "reason": "No existing questions to compare against."}), 200
        
        existing_questions_text = [clean_html(q.get('Question', '')) for q in existing_questions]
        all_texts = [data['question']] + existing_questions_text
        
        embeddings = get_embeddings(all_texts, provider=embedding_provider, model_name=embedding_model)
        if not embeddings:
            return upstream_failed(
                f"The {embedding_provider} embedding model returned nothing for these questions."
            )

        new_q_embedding = np.array([embeddings[0]])
        existing_q_embeddings = np.array(embeddings[1:])
        similarities = cosine_similarity(new_q_embedding, existing_q_embeddings)[0]

        threshold = float(current_app.config['SIMILARITY_THRESHOLD'])
        
        matched_candidates = [
            existing_questions[i] for i, score in enumerate(similarities) if score >= threshold
        ]
        
        if matched_candidates:
            verified_matches = verify_matches_with_llm(
                ref_question=data['question'],
                candidates=matched_candidates,
                provider=reasoning_provider,
                model_name=reasoning_model,
                task_type='similarity'
            )
            
            if verified_matches:
                return jsonify({"response": "yes", "matched_questions": verified_matches}), 200
            else:
                return jsonify({"response": "no"}), 200
        else:
            return jsonify({"response": "no"}), 200

    except AIServiceUnavailableError as e:
        return unavailable(str(e))
    except Exception:
        logging.error("An unexpected error occurred in check_similarity", exc_info=True)
        return internal()


@api_bp.route('/group_similar_questions', methods=['POST'])
@require_api_key
def group_similar_questions() -> JsonResponse:
    """Groups a list of questions by semantic similarity with double verification."""
    try:
        data: Dict[str, Any] = grouping_schema.load(request.get_json())
    except ValidationError as err:
        return invalid_payload(err.messages)

    embedding_provider = data['embedding_provider']
    reasoning_provider = data['reasoning_provider']

    refused = (unusable_provider('embedding', embedding_provider)
               or unusable_provider('reasoning', reasoning_provider))
    if refused:
        return refused

    embedding_model = get_model_from_provider('embedding', embedding_provider)
    reasoning_model = get_model_from_provider('reasoning', reasoning_provider)

    try:
        questions = fetch_questions_from_url(data['questions_url'])
        if not questions or len(questions) < 2:
            return jsonify({"response": "no", "reason": "Not enough questions to form a group."}), 200

        questions_text = [clean_html(q.get('Question', '')) for q in questions]
        embeddings = get_embeddings(questions_text, provider=embedding_provider, model_name=embedding_model)

        if not embeddings:
            return upstream_failed(
                f"The {embedding_provider} embedding model returned nothing for these questions."
            )

        distance_threshold = 1 - float(current_app.config['SIMILARITY_THRESHOLD'])
        clustering = AgglomerativeClustering(
            n_clusters=None, metric='cosine', linkage='average', distance_threshold=distance_threshold
        ).fit(embeddings)

        initial_groups: Dict[int, list] = {}
        for i, label in enumerate(clustering.labels_):
            initial_groups.setdefault(int(label), []).append(questions[i])

        verified_groups = []
        
        for group in initial_groups.values():
            if len(group) < 2:
                continue

            anchor_question = group[0]
            anchor_text = clean_html(anchor_question.get('Question', ''))
            candidates = group[1:]

            confirmed_matches = verify_matches_with_llm(
                ref_question=anchor_text,
                candidates=candidates,
                provider=reasoning_provider,
                model_name=reasoning_model,
                task_type='grouping'
            )

            final_group = [anchor_question] + confirmed_matches

            if len(final_group) > 1:
                verified_groups.append(final_group)

        if verified_groups:
            return jsonify({"response": "yes", "matched_groups": verified_groups}), 200
        else:
            return jsonify({"response": "no"}), 200

    except AIServiceUnavailableError as e:
        return unavailable(str(e))
    except Exception:
        logging.error("An unexpected error occurred in group_similar_questions", exc_info=True)
        return internal()
    
    
@api_bp.route('/convert-to-latex', methods=['POST'])
@require_api_key
def convert_to_latex() -> JsonResponse:
    """Converts a list of HTML questions into LaTeX."""
    try:
        data = latex_schema.load(request.get_json())
    except ValidationError as err:
        return invalid_payload(err.messages)

    html_contents = data['html_contents']
    provider = data['reasoning_provider']

    # Get the configured model name for this provider
    model_name = get_model_from_provider('reasoning', provider)
    if not model_name:
        refused = unusable_provider('reasoning', provider)
        if refused:
            return refused

    try:
        # Convert each question, preserving input order
        latex_codes = [
            convert_html_to_latex_with_llm(html_content, provider, model_name)
            for html_content in html_contents
        ]

        return jsonify({"latex_codes": latex_codes}), 200

    except AIServiceUnavailableError as e:
        return unavailable(str(e))
    except Exception:
        logging.error("An unexpected error occurred in convert_to_latex", exc_info=True)
        return internal()