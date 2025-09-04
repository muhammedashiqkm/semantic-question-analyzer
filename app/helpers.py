import logging
import requests
import json
from json import JSONDecodeError
from typing import List, Dict, Any, Optional

from flask import current_app, Response
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from openai import APIStatusError, APIConnectionError, OpenAI
from bs4 import BeautifulSoup
from . import openai_client, deepseek_client

class AIServiceUnavailableError(Exception):
    """Custom exception for when an external AI service is unavailable."""
    pass

def get_ai_client(provider: str) -> Optional[OpenAI]:
    """
    Returns a pre-initialized AI client for the given provider.
    
    :param provider: The name of the AI provider ('openai' or 'deepseek').
    :return: An initialized OpenAI client instance or None.
    """
    if provider == 'openai':
        if not openai_client:
            raise ValueError("OpenAI client is not initialized. Check API key.")
        return openai_client
    if provider == 'deepseek':
        if not deepseek_client:
            raise ValueError("DeepSeek client is not initialized. Check API key.")
        return deepseek_client
    return None

def clean_html(raw_html: str) -> str:
    """
    Removes HTML tags from a string.
    
    :param raw_html: The input string containing HTML.
    :return: The cleaned text.
    """
    if not raw_html:
        return ""
    return BeautifulSoup(raw_html, "lxml").get_text().strip()

def fetch_questions_from_url(url: str) -> Optional[List[Dict[str, Any]]]:
    """
    Fetches and parses question data from a given URL.

    :param url: The URL to fetch data from.
    :return: A list of dictionaries or None if an error occurs.
    """
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, list) else None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from URL {url}: {e}")
        return None
    except JSONDecodeError:
        logging.error(f"Failed to decode JSON from URL {url}")
        return None

def validate_question_quality(question_text: str, provider: str, model_name: str) -> bool:
    """
    Uses an LLM to check if a question is valid.
    Returns ONLY a boolean: True if valid, False otherwise.
    Raises AIServiceUnavailableError if the AI service itself is down.
    """
    system_prompt = (
        "You are an expert evaluator. Analyze the user's question. "
        "Determine if it is grammatically correct, complete, and makes logical sense. "
        "Respond ONLY with a valid JSON object with a single key: 'is_valid' (boolean)."
    )

    try:
        if provider == 'gemini':
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                f"{system_prompt}\n\nQuestion: \"{question_text}\"",
                generation_config={"response_mime_type": "application/json"}
            )
            result = json.loads(response.text)
            return result.get('is_valid', False)

        elif provider in ['openai', 'deepseek']:
            client = get_ai_client(provider)
            if not client:
                raise ValueError(f"Could not get client for provider: {provider}")
            
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

   
    except (google_exceptions.ServiceUnavailable, google_exceptions.RetryError, APIStatusError, APIConnectionError) as e:
        logging.error(f"AI Service Unavailable (provider: {provider}) during quality check: {e}")
        raise AIServiceUnavailableError(f"The {provider} quality check service is unavailable.")
    
    
    except JSONDecodeError:
        logging.error(f"Failed to parse JSON from {provider} during quality check.")
        return False



def get_embeddings(texts: List[str], provider: str, model_name: str) -> List[List[float]]:
    """
    Generates embeddings for a list of texts using the specified provider.
    
    :param texts: A list of strings to embed.
    :param provider: The embedding provider ('gemini' or 'openai').
    :param model_name: The specific model name to use.
    :return: A list of embedding vectors.
    """
    if provider == 'gemini':
        try:
            result = genai.embed_content(
                model=model_name,
                content=texts,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embedding = result['embedding']
            return [embedding] if not isinstance(embedding[0], list) else embedding
        except (google_exceptions.ServiceUnavailable, google_exceptions.RetryError) as e:
            logging.error(f"AI Service Unavailable (provider: {provider}): {e}")
            raise AIServiceUnavailableError(f"The {provider} embedding service is unavailable.")
    

    elif provider == 'openai':
        try:
            client = get_ai_client('openai')
            if not client:
                raise ValueError("Could not get OpenAI client")
            response = client.embeddings.create(input=texts, model=model_name)
            return [item.embedding for item in response.data]
        except (APIStatusError, APIConnectionError) as e:
            logging.error(f"AI Service Unavailable (provider: {provider}): {e}")
            raise AIServiceUnavailableError(f"The {provider} embedding service is unavailable.")

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
