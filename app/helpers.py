import logging
import requests
import json
from json import JSONDecodeError
from flask import current_app
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from openai import APIStatusError, APIConnectionError
from bs4 import BeautifulSoup
from . import openai_client, deepseek_client

class AIServiceUnavailableError(Exception):
    """Custom exception for when an external AI service is unavailable."""
    pass

def get_ai_client(provider):
    """Returns a pre-initialized AI client for the given provider."""
    if provider == 'openai':
        if not openai_client:
            raise ValueError("OpenAI client is not initialized. Check API key.")
        return openai_client
    if provider == 'deepseek':
        if not deepseek_client:
            raise ValueError("DeepSeek client is not initialized. Check API key.")
        return deepseek_client
    return None

def clean_html(raw_html):
    """Removes HTML tags from a string."""
    if not raw_html:
        return ""
    return BeautifulSoup(raw_html, "lxml").get_text().strip()

def fetch_questions_from_url(url):
    """Fetches and parses question data from a given URL."""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, list) else None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from URL {url}: {e}")
        return None

def validate_question_quality(question_text, provider, model_name):
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
    except Exception as e:
        logging.error(f"Failed to validate question with {provider}: {e}")
        return False


def get_embeddings(texts, provider, model_name):
    """Generates embeddings for a list of texts using the specified provider."""
    if provider == 'gemini':
        try:
            result = genai.embed_content(
                model=model_name,
                content=texts,
                task_type="RETRIEVAL_DOCUMENT"
            )
            return result['embedding']
        except (google_exceptions.ServiceUnavailable, google_exceptions.RetryError) as e:
            logging.error(f"AI Service Unavailable (provider: {provider}): {e}")
            raise AIServiceUnavailableError(f"The {provider} embedding service is unavailable.")
        except Exception as e:
            logging.error(f"Error generating embeddings with {provider}: {e}")
            raise

    elif provider == 'openai':
        try:
            client = get_ai_client('openai')
            response = client.embeddings.create(input=texts, model=model_name)
            return [item.embedding for item in response.data]
        except (APIStatusError, APIConnectionError) as e:
            logging.error(f"AI Service Unavailable (provider: {provider}): {e}")
            raise AIServiceUnavailableError(f"The {provider} embedding service is unavailable.")
        except Exception as e:
            logging.error(f"Error generating embeddings with {provider}: {e}")
            raise

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")