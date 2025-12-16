"""LLM-powered resume analyzer with Ollama, used to interpret requirements per candidate."""
import json
import logging
from typing import Dict, Any

from jd_analyzer_llm import OLLAMA_URL, MODEL
import requests

logger = logging.getLogger(__name__)


def analyze_resume_with_ollama(resume_text: str, config: Dict[str, Any], timeout: int = 90) -> Dict[str, Any]:
    """
    Ask Ollama to evaluate how well a single resume meets each must-have and nice-to-have.

    Timeout of 90s allows for:
    - Long resumes (up to 5 pages)
    - 20+ requirements to evaluate
    - Processing on moderate hardware

    Returns a dict of the form:
    {
      "must_haves": {
        "<requirement>": {"met": true/false, "evidence": "short quote or explanation"},
        ...
      },
      "nice_to_haves": {
        "<requirement>": {"met": true/false, "evidence": "short quote or explanation"},
        ...
      },
      "summary": "one-line summary of candidate fit",
      "risks": ["optional risk bullet 1", "optional risk bullet 2", ...]
    }

    It is critical that the LLM returns valid JSON only.
    """
    must_haves = config.get("must_haves", [])
    nice_to_haves = config.get("nice_to_haves", [])
    job = config.get("job", {})
    job_title = job.get("title", "Unknown role")

    # Build a clear, strict prompt to get consistent JSON
    prompt = f"""You are helping a recruiter evaluate a single candidate for the role: {job_title}.

Job must-have requirements:
{json.dumps(must_haves, indent=2)}

Job nice-to-have skills:
{json.dumps(nice_to_haves, indent=2)}

Resume text:
\"\"\"{resume_text}\"\"\"

For each must-have and nice-to-have, decide if the resume clearly meets the requirement.

CRITICAL RULES:
1. Be conservative: mark a requirement as met ONLY if there is clear evidence in the resume.
2. Do NOT infer skills that are not explicitly or very strongly implied.
3. When a requirement is met, include a short evidence string (quoted phrase or sentence).
4. If not met, use an empty evidence string "".
5. Only evaluate the requirements listed above. Do NOT invent new requirements.

Respond with ONLY valid JSON using exactly this structure:
{{
  "must_haves": {{
    "<must_have_1>": {{"met": true or false, "evidence": "<short evidence or empty string>"}},
    "<must_have_2>": {{"met": true or false, "evidence": "<short evidence or empty string>"}},
    ...
  }},
  "nice_to_haves": {{
    "<nice_to_have_1>": {{"met": true or false, "evidence": "<short evidence or empty string>"}},
    "<nice_to_have_2>": {{"met": true or false, "evidence": "<short evidence or empty string>"}},
    ...
  }},
  "summary": "one-line summary of candidate fit for this role",
  "risks": ["optional risk bullet 1", "optional risk bullet 2"]
}}"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "num_predict": 1500,
            },
        },
        timeout=timeout,
    )

    if response.status_code != 200:
        raise Exception(f"Ollama API error (resume): {response.status_code}")

    result = response.json()
    response_text = result.get("response", "")

    # Clean markdown fences if present - improved validation
    response_text = response_text.strip()
    if response_text.startswith("```"):
        # Extract content between first and last ```
        parts = response_text.split("```")
        for part in parts[1::2]:  # Check odd indices (inside fences)
            part = part.strip()
            # Remove language identifier if present (e.g., "json")
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                response_text = part
                break
        else:
            # If no valid JSON found in fences, try the whole response
            response_text = response_text.replace("```json", "").replace("```", "").strip()

    # Validate JSON before parsing
    if not response_text.startswith("{"):
        raise ValueError(f"LLM response does not start with JSON object: {response_text[:100]}...")
    
    data = json.loads(response_text)

    # Ensure we have the expected top-level keys
    for key in ["must_haves", "nice_to_haves"]:
        if key not in data or not isinstance(data[key], dict):
            raise ValueError(f"LLM resume output missing or invalid '{key}' section")

    # Ensure every configured requirement appears at least once;
    # if not, fill in as unmet.
    mh_decisions = data.get("must_haves", {})
    for mh in must_haves:
        if mh not in mh_decisions:
            mh_decisions[mh] = {"met": False, "evidence": ""}
    data["must_haves"] = mh_decisions

    nh_decisions = data.get("nice_to_haves", {})
    for nh in nice_to_haves:
        if nh not in nh_decisions:
            nh_decisions[nh] = {"met": False, "evidence": ""}
    data["nice_to_haves"] = nh_decisions

    # Normalize summary/risks keys
    if "summary" not in data or not isinstance(data["summary"], str):
        data["summary"] = ""
    risks = data.get("risks")
    if not isinstance(risks, list):
        data["risks"] = []

    return data


