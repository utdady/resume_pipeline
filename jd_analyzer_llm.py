"""LLM-powered job description analyzer with Ollama, with rule-based fallback."""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import requests
import yaml

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"

logger = logging.getLogger(__name__)


def _build_summary(text: str, max_len: int = 200) -> str:
    """Create a short one-line summary from the JD text."""
    return text[:max_len].replace("\n", " ").strip() + "..."


def analyze_jd_with_ollama(jd_text: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Use Ollama to extract JD requirements and return a config compatible with jd_meta.yaml.
    Raises exceptions if Ollama is unavailable or returns invalid JSON.
    """
    prompt = f"""Analyze this job description and extract structured requirements.

Extract these fields from the job description:

1. job_title: The position title (e.g., "Senior Software Engineer", "Financial Analyst", "Electrical Engineer")

2. domain: Choose ONE from [software, electrical, mechanical, data, financial, business]
   - software: programming, development, APIs, databases, web applications, software engineering
   - electrical: power systems, circuits, protection, CAPE/ASPEN, Gridscale X, electrical engineering
   - mechanical: CAD, manufacturing, design, thermodynamics, mechanical engineering
   - data: analytics, machine learning, statistics, data science, data engineering
   - financial: finance, investment banking, accounting, financial modeling, CFA
   - business: management, consulting, operations, strategy, business analysis

3. must_haves: CRITICAL requirements (degree, years, specific skills/tools)
   - Extract ONLY concrete skills, NOT "experience" or "knowledge"
   - Examples: ["bachelor electrical engineering", "python", "sql", "cape software", "5 years"]
   - NOT: ["years", "experience", "proficiency", "knowledge", "skills"]
   - Clean compound terms: "sql experience" → "sql", "python programming" → "python"
   - Recognize synonyms: "Gridscale X" = "CAPE software", "fault analysis" = "short circuit"

4. nice_to_haves: Preferred/bonus skills
   - Similar rules as must_haves
   - Examples: ["aws", "docker", "agile", "tableau", "kubernetes"]

5. experience: Extract years from patterns like "4-6 years", "5+ years"
   {{
     "min_years": 4,
     "max_years": 6,
     "preferred_years": 5
   }}
   - If range: "4-6 years" → min: 4, max: 6, preferred: 5
   - If single: "5 years" → min: 5, max: 7, preferred: 5
   - If minimum: "5+ years" → min: 5, max: 10, preferred: 5

6. title_keywords: Role-relevant terms for matching
   - Examples: ["engineer", "power", "systems", "protection", "developer", "analyst"]

CRITICAL RULES:
1. Clean compound terms: "sql experience" → "sql"
2. Recognize synonyms: "Gridscale X" = "CAPE software"
3. Extract actual skills/tools, NOT generic words
4. Return ONLY valid JSON, no markdown, no explanation

Job Description:
{jd_text}

Respond with ONLY this JSON structure:
{{
  "job_title": "...",
  "domain": "...",
  "must_haves": [...],
  "nice_to_haves": [...],
  "experience": {{"min_years": 0, "max_years": 0, "preferred_years": 0}},
  "title_keywords": [...]
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
                "num_predict": 1000,
            },
        },
        timeout=timeout,
    )

    if response.status_code != 200:
        raise Exception(f"Ollama API error: {response.status_code}")

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
    
    extracted = json.loads(response_text)

    required = ["job_title", "domain", "must_haves", "nice_to_haves", "experience", "title_keywords"]
    for field in required:
        if field not in extracted:
            raise ValueError(f"Missing required field: {field}")

    # Build config compatible with jd_meta.yaml
    config = {
        "metadata_version": 1,
        "job": {
            "requisition_id": "AUTO-GENERATED",
            "title": extracted["job_title"],
            "domain": extracted["domain"],
            "summary": _build_summary(jd_text),
        },
        "must_haves": extracted["must_haves"],
        "nice_to_haves": extracted["nice_to_haves"],
        "title_keywords": extracted["title_keywords"],
        "experience": extracted["experience"],
        "weights": {
            "must_have_coverage": 0.45,
            "skill_overlap": 0.25,
            "title_similarity": 0.15,
            "years_exp": 0.15,
        },
        "thresholds": {
            "advance": 0.72,
            "review": 0.50,
        },
        "notes": f"Auto-generated using Ollama LLM ({MODEL}). Domain: {extracted['domain']}",
    }

    return config


def check_ollama_available() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def analyze_jd_hybrid(
    jd_text: str,
    output_path: Path,
    use_llm: bool = True,
    job_title: Optional[str] = None,
    requisition_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Hybrid approach: Try LLM, fallback to rule-based analyzer.
    
    Args:
        jd_text: The job description text to analyze
        output_path: Path where the YAML config will be written
        use_llm: If True, attempt to use Ollama LLM for analysis
        job_title: Optional job title to override auto-detection
        requisition_id: Optional requisition ID to include in config
    
    Returns:
        Dict containing the job description configuration with:
            - job: Dict with title, domain, requisition_id, summary
            - must_haves: List of critical requirements
            - nice_to_haves: List of preferred skills
            - title_keywords: Keywords for role matching
            - experience: Dict with min_years, max_years, preferred_years
            - weights: Scoring weights
            - thresholds: Score thresholds
    
    Behavior:
        - If use_llm=True and Ollama is available: Uses LLM for analysis
        - If LLM fails or is unavailable: Falls back to rule-based NLP analysis
        - Always writes the resulting config to output_path as YAML
    """
    if use_llm and check_ollama_available():
        try:
            logger.info("Analyzing job description with Ollama LLM...")
            config = analyze_jd_with_ollama(jd_text)

            if job_title:
                config["job"]["title"] = job_title
            if requisition_id:
                config["job"]["requisition_id"] = requisition_id

            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

            logger.info(
                "Used Ollama AI (Domain: %s). Must-haves: %d, Nice-to-haves: %d",
                config["job"].get("domain", "unknown"),
                len(config["must_haves"]),
                len(config["nice_to_haves"]),
            )
            return config
        except Exception as exc:
            logger.warning("Ollama JD analysis failed: %s. Falling back to rule-based analysis...", exc)

    # Fallback to existing rule-based analyzer
    from jd_analyzer import generate_jd_meta

    logger.info("Using rule-based JD analysis...")
    generate_jd_meta(jd_text, output_path, job_title, requisition_id)

    with open(output_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Used rule-based JD analyzer")
    return config
