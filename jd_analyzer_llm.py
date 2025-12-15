"""LLM-powered job description analyzer with Ollama, with rule-based fallback."""
import json
from pathlib import Path
from typing import Optional, Dict, Any

import requests
import yaml

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"


def _build_summary(text: str, max_len: int = 200) -> str:
    """Create a short one-line summary from the JD text."""
    return text[:max_len].replace("\n", " ").strip() + "..."


def analyze_jd_with_ollama(jd_text: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Use Ollama to extract JD requirements and return a config compatible with jd_meta.yaml.
    Raises exceptions if Ollama is unavailable or returns invalid JSON.
    """
    prompt = f"""Analyze this job description and extract structured requirements.

CRITICAL RULES:
1. Extract ONLY actual skills and qualifications, NOT generic words
2. DO NOT include: "years", "experience", "knowledge", "proficiency", "skills", "ability"
3. Clean compound terms: "sql experience" -> "sql", "python programming" -> "python"
4. Recognize variations: "Gridscale X" = "CAPE software", "fault analysis" = "short circuit"
5. Return ONLY valid JSON, no markdown blocks, no explanations

Job Description:
{jd_text}

Extract:
- job_title: string (auto-detect from JD)
- domain: one of [software, electrical, mechanical, data, financial, business]
- must_haves: list of CLEAN skill terms (bachelor, python, cape, NOT "years of experience")
- nice_to_haves: list of CLEAN preferred skills
- experience: {{min_years: int, max_years: int, preferred_years: int}}
- title_keywords: list of role-relevant terms

Respond with ONLY this JSON structure:
{{
  "job_title": "...",
  "domain": "...",
  "must_haves": [...],
  "nice_to_haves": [...],
  "experience": {{...}},
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

    # Clean markdown fences if present
    if "```json" in response_text:
        response_text = response_text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in response_text:
        response_text = response_text.split("```", 1)[1].split("```", 1)[0]

    extracted = json.loads(response_text.strip())

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
    Returns the config dict and writes YAML to output_path.
    """
    if use_llm and check_ollama_available():
        try:
            print("Analyzing with Ollama LLM...")
            config = analyze_jd_with_ollama(jd_text)

            if job_title:
                config["job"]["title"] = job_title
            if requisition_id:
                config["job"]["requisition_id"] = requisition_id

            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

            print(f"Used Ollama AI (Domain: {config['job'].get('domain', 'unknown')})")
            print(f"  Must-haves: {len(config['must_haves'])}")
            print(f"  Nice-to-haves: {len(config['nice_to_haves'])}")
            return config
        except Exception as exc:
            print(f"Warning: Ollama failed: {exc}")
            print("Falling back to rule-based analysis...")

    # Fallback to existing rule-based analyzer
    from jd_analyzer import generate_jd_meta

    print("Using rule-based analysis...")
    generate_jd_meta(jd_text, output_path, job_title, requisition_id)

    with open(output_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("Used rule-based analyzer")
    return config
