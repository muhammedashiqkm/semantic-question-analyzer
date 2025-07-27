import logging
import requests
import google.generativeai as genai
from bs4 import BeautifulSoup

EMBEDDING_MODEL = 'text-embedding-004'

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
        
        if isinstance(data, list):
            return data
        else:
            logging.error("JSON from URL is not a list as expected.")
            return None

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from URL {url}: {e}")
        return None
    except ValueError:
        logging.error(f"Error decoding JSON from URL {url}.")
        return None

def get_embeddings(texts):
    """Generates embeddings for a list of texts using the Google AI model."""
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=texts,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return result['embedding']
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        return None