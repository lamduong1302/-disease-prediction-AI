import json
import logging
import time
from typing import Any, Dict, Optional

import requests

# Configure logging
logger = logging.getLogger(__name__)


def _extract_first_json_object(text: str) -> Optional[dict]:
    """
    Extract the first valid top-level JSON object from a model response.
    Handles cases where JSON is inside ```json code blocks or mixed with text.
    """
    if not text:
        return None

    # Prefer JSON inside fenced code blocks
    # Example: ```json { ... } ```
    lower = text.lower()
    fence_start = lower.find("```json")
    if fence_start != -1:
        after = text[fence_start + len("```json") :]
        fence_end = after.find("```")
        if fence_end != -1:
            candidate = after[:fence_end].strip()
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception as e:
                logger.debug(f"Failed to parse JSON from code block: {e}")
                pass

    # Fallback: scan for balanced braces for first JSON object
    start = None
    depth = 0
    in_str = False
    escape = False

    for i, ch in enumerate(text):
        if start is None:
            if ch == "{":
                start = i
                depth = 1
            continue

        # We are inside candidate
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                snippet = text[start : i + 1]
                try:
                    parsed = json.loads(snippet)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception as e:
                    logger.debug(f"Failed to parse JSON object: {e}")
                    return None

    return None


def _validate_json_response(data: dict) -> bool:
    """
    Validate that the response has the expected keys.
    """
    required_keys = {"explanation", "recommendations", "nutrition_tips"}
    return all(key in data for key in required_keys)


def call_gemini(
    *,
    api_key: str,
    model: str,
    prompt: str,
    timeout_s: int = 25,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Call Gemini via Google Generative Language REST API.
    Returns parsed JSON if possible; otherwise empty dict.
    
    Args:
        api_key: Gemini API key
        model: Model name (e.g., "gemini-1.5-flash")
        prompt: The prompt to send
        timeout_s: Request timeout in seconds
        max_retries: Number of retries on failure
    
    Returns:
        Parsed JSON response or empty dict on failure
    """
    if not api_key:
        logger.debug("Gemini API key not provided")
        return {}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            # Request JSON response format
            "responseMimeType": "application/json",
        },
    }

    for attempt in range(max_retries):
        try:
            logger.debug(f"Calling Gemini API (attempt {attempt + 1}/{max_retries})")
            resp = requests.post(
                url,
                json=payload,
                timeout=timeout_s,
                headers={"Content-Type": "application/json"},
            )
            
            # Handle rate limiting with exponential backoff
            if resp.status_code == 429:
                wait_time = 2 ** attempt
                logger.warning(f"Rate limited; waiting {wait_time}s before retry")
                time.sleep(wait_time)
                continue
            
            if resp.status_code != 200:
                logger.warning(
                    f"Gemini API returned status {resp.status_code}: "
                    f"{resp.text[:200]}"
                )
                continue

            data = resp.json()
            
            # Extract text from response
            # Response shape: candidates[0].content.parts[0].text
            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            
            if not text:
                logger.warning("Gemini returned empty response text")
                continue

            # Try to extract JSON from response
            parsed = _extract_first_json_object(text)
            
            if isinstance(parsed, dict):
                # Validate the response structure
                if _validate_json_response(parsed):
                    logger.info("Successfully parsed valid Gemini response")
                    return parsed
                else:
                    logger.warning(
                        f"Response missing required keys. Got: {list(parsed.keys())}"
                    )
                    # Still return it - UI will handle missing keys gracefully
                    return parsed
            
            # Fallback: return raw text
            logger.info("Returning response as raw text")
            return {"raw_text": text}
            
        except requests.exceptions.Timeout:
            logger.warning(f"Gemini API request timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error calling Gemini API: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode Gemini response JSON: {e}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error calling Gemini API: {e}", exc_info=True)
            continue

    logger.error(f"Failed to get valid response from Gemini after {max_retries} attempts")
    return {}

