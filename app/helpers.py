import logging
import requests
import json
from json import JSONDecodeError
from typing import List, Dict, Any, Optional

from flask import current_app
from google import genai
from google.genai import types 
from bs4 import BeautifulSoup
from . import openai_client, deepseek_client, gemini_client

class AIServiceUnavailableError(Exception):
    """Custom exception for when an external AI service is unavailable."""
    pass

def get_ai_client(provider: str):
    """
    Returns a pre-initialized AI client for the given provider.
    """
    if provider == 'openai':
        if not openai_client:
            raise ValueError("OpenAI client is not initialized. Check API key.")
        return openai_client
    if provider == 'deepseek':
        if not deepseek_client:
            raise ValueError("DeepSeek client is not initialized. Check API key.")
        return deepseek_client
    if provider == 'gemini':
        if not gemini_client:
            raise ValueError("Gemini client is not initialized. Check API key.")
        return gemini_client
    return None

def clean_html(raw_html: str) -> str:
    """Removes HTML tags from a string."""
    if not raw_html:
        return ""
    return BeautifulSoup(raw_html, "lxml").get_text().strip()

def fetch_questions_from_url(url: str) -> Optional[List[Dict[str, Any]]]:
    """Fetches and parses question data from a given URL."""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, list) else None
    except (requests.exceptions.RequestException, JSONDecodeError) as e:
        logging.error(f"Error fetching or parsing URL {url}: {e}")
        return None

def validate_question_quality(question_text: str, provider: str, model_name: str) -> bool:
    """
    Uses an LLM to check if a question is valid.
    """
    system_prompt = (
        "You are an expert evaluator. Analyze the user's question. "
        "Determine if it is grammatically correct, complete, and makes logical sense. "
        "Respond ONLY with a valid JSON object with a single key: 'is_valid' (boolean)."
    )
    try:
        if provider == 'gemini':
            client = get_ai_client('gemini')
            response = client.models.generate_content(
                model=model_name,
                contents=f"{system_prompt}\n\nQuestion: \"{question_text}\"",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            result = json.loads(response.text)
            return result.get('is_valid', False)

        elif provider in ['openai', 'deepseek']:
            client = get_ai_client(provider)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: \"{question_text}\""}
                ],
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            return result.get('is_valid', False)
        
        else:
            raise ValueError(f"Unsupported reasoning provider: {provider}")
    
    except Exception as e:
        logging.error(f"An unexpected error occurred with the '{provider}' service during quality check: {e}")
        raise AIServiceUnavailableError(f"The AI service for '{provider}' is currently unavailable.")


def get_embeddings(texts: List[str], provider: str, model_name: str) -> List[List[float]]:
    """
    Generates embeddings for a list of texts.
    """
    try:
        if provider == 'gemini':
            client = get_ai_client('gemini')
            result = client.models.embed_content(
                model=model_name,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=768 
                )
            )
            embeddings = [e.values for e in result.embeddings]
            return embeddings
    
        elif provider == 'openai':
            client = get_ai_client('openai')
            response = client.embeddings.create(input=texts, model=model_name)
            return [item.embedding for item in response.data]

        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    except Exception as e:
        logging.error(f"An unexpected error occurred with the '{provider}' service during embedding: {e}")
        raise AIServiceUnavailableError(f"The AI embedding service for '{provider}' is currently unavailable.")