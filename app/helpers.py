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

def verify_matches_with_llm(
    ref_question: str, 
    candidates: List[Dict[str, Any]], 
    provider: str, 
    model_name: str,
    task_type: str = "similarity"
) -> List[Dict[str, Any]]:
    """
    Uses an LLM to double-check matches with prompts tailored for specific tasks.
    
    :param ref_question: The question to compare against (New Question or Anchor).
    :param candidates: The list of candidate questions.
    :param task_type: 'similarity' (default) or 'grouping'.
    """
    if not candidates:
        return []

    # Format candidates
    candidates_formatted = "\n".join([
        f"ID {i}: {clean_html(c.get('Question', ''))}" 
        for i, c in enumerate(candidates)
    ])

    if task_type == "grouping":
        system_prompt = (
            "You are an expert Semantic Grouper. "
            "The 'Anchor Question' defines the specific topic and intent of a group. "
            "Compare the 'Candidate Questions' against this Anchor. "
            "Identify which candidates share the EXACT same semantic meaning and intent as the Anchor, suitable for merging into a single group. "
            "Respond ONLY with a valid JSON object with a single key: 'match_ids' (list of integers). "
            "If none match, return 'match_ids': []."
        )
        user_content = f"Anchor Question: \"{ref_question}\"\n\nCandidate Questions:\n{candidates_formatted}"
    
    else:
        system_prompt = (
            "You are a strict Duplicate Question Detector. "
            "Compare the 'New Question' with the 'Candidate Questions'. "
            "Identify which candidates are semantically identical (meaning exactly the same thing) to the New Question. "
            "Respond ONLY with a valid JSON object with a single key: 'match_ids' (list of integers). "
            "If none match, return 'match_ids': []."
        )
        user_content = f"New Question: \"{ref_question}\"\n\nCandidate Questions:\n{candidates_formatted}"

    try:
        if provider == 'gemini':
            client = get_ai_client('gemini')
            response = client.models.generate_content(
                model=model_name,
                contents=f"{system_prompt}\n\n{user_content}",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            result = json.loads(response.text)
            
        elif provider in ['openai', 'deepseek']:
            client = get_ai_client(provider)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
        else:
            raise ValueError(f"Unsupported reasoning provider: {provider}")

        match_ids = result.get('match_ids', [])
        confirmed_matches = [candidates[i] for i in match_ids if 0 <= i < len(candidates)]
        return confirmed_matches

    except Exception as e:
        logging.error(f"An unexpected error occurred with the '{provider}' service during double verification ({task_type}): {e}")
        raise AIServiceUnavailableError(f"The AI service for '{provider}' is currently unavailable for verification.")

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
    
    
def convert_html_to_latex_with_llm(html_content: str, provider: str, model_name: str) -> str:
    """
    Converts messy HTML content directly into perfectly formatted LaTeX using an LLM.
    """
    prompt = f"""
    You are an expert academic formatting assistant for University Question Papers. 
    Your exact task is to convert the following messy HTML question into clean, compilable, and standardized LaTeX.

    CRITICAL FORMATTING RULES:

    1. GENERAL CLEANUP & TEXT:
       - The input may be copy-pasted from ChatGPT, MS Word, or websites. IGNORE any leftover CSS, classes, markdown artifacts, or formatting junk. Extract the core academic text.
       - Fix broken or messy tags (like <b>, <strong>, <i>, <em>) using \\textbf{{}} and \\textit{{}}.
       - If content is centered in HTML, wrap it in \\begin{{center}} ... \\end{{center}}.
       - Do not change the logic, meaning, or variables of the question.

    2. TABLES (CRITICAL MARGIN RULE):
       - NEVER use the standard \\begin{{tabular}} environment.
       - ALWAYS use \\begin{{tabularx}}{{\\linewidth}} for tables to ensure they fit A4 margins.
       - Use the 'X' column specifier for any column containing descriptive text (e.g., {{|c|X|X|}}).
       - IMPORTANT: The '&' character MUST be used as the column separator. Do NOT escape '&' if it is being used to separate table cells.

    3. IMAGES & PLACEHOLDERS (CRITICAL PRESERVATION RULE):
       - The input text will contain image placeholders formatted exactly like [IMAGE_PLACEHOLDER_0] or [IMAGE_PLACEHOLDER].
       - You MUST leave these placeholders EXACTLY as they are in your final LaTeX output.
       - DO NOT escape the square brackets or underscores inside the placeholder (i.e., do NOT change it to \[IMAGE\_PLACEHOLDER\_0\]). The downstream system requires this exact string to inject the images.
       - If you happen to see any leftover raw <img src="..."> HTML tags, leave them exactly as they are without converting them.
       
       
    4. MATH, CHEMISTRY & SYMBOLS:
       - Preserve math equations perfectly using standard LaTeX math mode ($...$ or \\(...\\)).
       - Wrap chemical formulas (e.g., H2SO4) and reaction types (e.g., SN1, SN2) in math mode (e.g., $H_2SO_4$, $S_N1$).
       - CRITICAL: Escape special characters outside of math mode: % to \\%, # to \\#, $ to \\$, and _ to \\_. 
       - Only escape '&' to \\& if it appears as text (e.g., "Research & Development") and is NOT a table separator.

    5. LISTS:
       - Standardize lists using \\begin{{itemize}} or \\begin{{enumerate}}.
       - In table cells, avoid complex list environments unless the column is an 'X' type.

    OUTPUT FORMAT:
    Return ONLY the raw LaTeX code. Do NOT wrap the response in markdown blocks like ```latex. No conversational text.

    Messy HTML to clean:
    {html_content}
"""

    try:
        if provider == 'gemini':
            client = get_ai_client('gemini')
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            text = response.text
            
        elif provider == 'openai':
            client = get_ai_client('openai')
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            text = response.choices[0].message.content
            
        elif provider == 'deepseek':
            client = get_ai_client('deepseek')
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            text = response.choices[0].message.content
            
        else:
            raise ValueError(f"Unsupported reasoning provider: {provider}")

        # Safety: Strip any markdown blocks if the LLM adds them (e.g. ```latex ... ```)
        cleaned_text = text.strip()
        if cleaned_text.startswith("```"):
            # remove the first line (e.g., ```latex)
            cleaned_text = cleaned_text.split("\n", 1)[-1] 
        if cleaned_text.endswith("```"):
            # remove the last line
            cleaned_text = cleaned_text.rsplit("\n", 1)[0]

        return cleaned_text.strip()

    except Exception as e:
        logging.error(f"An error occurred with the '{provider}' service during LaTeX conversion: {e}")
        raise AIServiceUnavailableError(f"The AI service for '{provider}' is currently unavailable.")