"""Auto-extract job requirements from any job description and generate jd_meta.yaml."""
import argparse
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml


# Domain detection keywords
DOMAIN_KEYWORDS = {
    "software": [
        "software", "developer", "programming", "code", "application", "api",
        "python", "java", "javascript", "react", "node", "full stack", "backend", "frontend"
    ],
    "electrical": [
        "electrical", "power systems", "protection", "transmission", "substation",
        "circuit", "relay", "cape", "aspen", "grid", "renewable energy", "solar"
    ],
    "mechanical": [
        "mechanical", "cad", "solidworks", "manufacturing", "design", "machinery",
        "thermodynamics", "fluids", "materials", "automotive"
    ],
    "data": [
        "data", "analyst", "scientist", "machine learning", "ai", "analytics",
        "sql", "python", "statistics", "modeling", "visualization"
    ],
    "business": [
        "business", "analyst", "project manager", "consultant", "strategy",
        "finance", "marketing", "sales", "operations", "management"
    ],
    "financial": [
        "financial analyst", "investment banking", "finance", "financial modeling",
        "valuation", "dcf", "financial statements", "accounting", "cfa", "mba",
        "excel", "bloomberg", "m&a", "mergers", "acquisitions"
    ]
}


def detect_domain(text: str) -> str:
    """Detect the primary domain of the job description."""
    normalized = text.lower()
    domain_scores = {}
    
    # Check financial domain first (more specific)
    if "financial analyst" in normalized or "investment banking" in normalized:
        financial_score = sum(1 for keyword in DOMAIN_KEYWORDS["financial"] if keyword in normalized)
        if financial_score > 2:
            return "financial"
    
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in normalized)
        if score > 0:
            domain_scores[domain] = score
    
    if domain_scores:
        return max(domain_scores.items(), key=lambda x: x[1])[0]
    return "general"


def extract_experience_requirements(text: str) -> Dict:
    """Extract years of experience requirements."""
    normalized = text.lower()
    
    # Patterns for experience ranges
    patterns = [
        r"(\d+)\s*[-–]\s*(\d+)\s*years?",
        r"(\d+)\+?\s*years?\s*(?:of\s*)?experience",
        r"minimum\s+of\s+(\d+)\s*years?",
        r"at\s+least\s+(\d+)\s*years?",
        r"(\d+)\s*to\s*(\d+)\s*years?",
    ]
    
    min_years = None
    max_years = None
    preferred_years = None
    
    for pattern in patterns:
        matches = re.finditer(pattern, normalized, re.IGNORECASE)
        for match in matches:
            groups = match.groups()
            if len(groups) == 2:
                # Range found
                start, end = int(groups[0]), int(groups[1])
                min_years = min(start, end) if min_years is None else min(min_years, start)
                max_years = max(start, end) if max_years is None else max(max_years, end)
                preferred_years = (start + end) / 2
            elif len(groups) == 1:
                # Single number
                years = int(groups[0])
                if min_years is None or years < min_years:
                    min_years = years
                if max_years is None or years > max_years:
                    max_years = years
    
    # Defaults if not found
    if min_years is None:
        min_years = 2
    if max_years is None:
        max_years = min_years + 2
    if preferred_years is None:
        preferred_years = (min_years + max_years) / 2
    
    return {
        "min_years": int(min_years),
        "max_years": int(max_years),
        "preferred_years": int(preferred_years)
    }


def extract_must_haves(text: str) -> List[str]:
    """Extract must-have requirements using pattern matching."""
    normalized = text.lower()
    must_haves = []
    
    # Sections that typically contain requirements
    requirement_sections = []
    
    # Find sections with "required", "must have", "essential", etc.
    section_patterns = [
        r"(?:required|must have|essential|mandatory|qualifications|requirements)[\s\S]{0,500}",
        r"minimum\s+requirements?[\s\S]{0,500}",
        r"basic\s+qualifications?[\s\S]{0,500}",
    ]
    
    for pattern in section_patterns:
        matches = re.finditer(pattern, normalized, re.IGNORECASE)
        for match in matches:
            requirement_sections.append(match.group(0))
    
    # If no specific section found, use entire text
    if not requirement_sections:
        requirement_sections = [normalized]
    
    # Extract keywords from requirement sections
    all_text = " ".join(requirement_sections)
    
    # Common must-have patterns
    must_have_indicators = [
        r"required[:\s]+([^\.]+)",
        r"must\s+have[:\s]+([^\.]+)",
        r"essential[:\s]+([^\.]+)",
        r"mandatory[:\s]+([^\.]+)",
    ]
    
    extracted_phrases = []
    for pattern in must_have_indicators:
        matches = re.finditer(pattern, all_text, re.IGNORECASE)
        for match in matches:
            phrase = match.group(1).strip()
            if len(phrase) > 3:  # Filter out very short phrases
                extracted_phrases.append(phrase)
    
    # Extract common technical terms and skills
    # Look for bullet points or list items
    bullet_pattern = r"[•\-\*]\s*([^•\-\*\n]+)"
    bullets = re.findall(bullet_pattern, all_text)
    
    # Combine extracted phrases and bullets
    all_candidates = extracted_phrases + bullets
    
    # Extract key terms (2-3 word phrases that are likely skills)
    # Common patterns: "X experience", "X degree", "X certification"
    skill_patterns = [
        r"(\w+\s+\w+)\s+(?:experience|degree|certification|knowledge|proficiency)",
        r"(?:bachelor|master|phd|bs|ms|ph\.?d\.?)\s+(?:in\s+)?(\w+(?:\s+\w+)?)",
        r"(\w+\s+\w+)\s+required",
    ]
    
    for pattern in skill_patterns:
        matches = re.finditer(pattern, all_text, re.IGNORECASE)
        for match in matches:
            term = match.group(1).strip().lower()
            if len(term) > 3 and term not in ["the", "and", "for", "with"]:
                must_haves.append(term)
    
    # Extract single important keywords (common technical terms)
    # Focus on specific skills/tools, not generic terms
    important_keywords = [
        "bachelor", "degree", "certification", "license"
    ]
    
    # Also look for domain-specific must-haves
    domain = detect_domain(text)
    if domain == "electrical":
        important_keywords.extend(["cape", "aspen", "protection", "short circuit", "power systems", 
                                   "transmission", "relay", "breaker", "substation"])
    elif domain == "software":
        important_keywords.extend(["programming", "coding", "development", "api", "javascript", 
                                   "python", "java", "react", "node"])
    elif domain == "data":
        important_keywords.extend(["sql", "python", "analytics", "statistics", "machine learning"])
    
    # Financial domain keywords
    if "financial" in normalized or "finance" in normalized or "banking" in normalized:
        important_keywords.extend(["excel", "financial modeling", "financial analysis", "valuation", 
                                   "dcf", "financial statements", "accounting", "cfa", "mba"])
    elif domain == "mechanical":
        important_keywords.extend(["cad", "solidworks", "autocad", "manufacturing", "design"])
    
    # Extract these keywords if they appear in requirement sections
    for keyword in important_keywords:
        if keyword in all_text:
            must_haves.append(keyword)
    
    # Deduplicate and clean
    must_haves = list(set([mh.strip() for mh in must_haves if len(mh.strip()) > 2]))
    
    # Filter out generic terms that aren't useful
    generic_terms = {"years", "experience", "knowledge", "proficiency", "required", 
                     "must", "have", "essential", "and", "the", "with", "of", "in"}
    must_haves = [mh for mh in must_haves if mh.lower() not in generic_terms]
    
    # Limit to top 10-15 most relevant
    return must_haves[:15]


def extract_nice_to_haves(text: str) -> List[str]:
    """Extract nice-to-have/preferred requirements."""
    normalized = text.lower()
    nice_to_haves = []
    
    # Find sections with "preferred", "nice to have", "plus", etc.
    preferred_sections = []
    
    preferred_patterns = [
        r"(?:preferred|nice to have|plus|bonus|advantage)[\s\S]{0,500}",
        r"additional\s+qualifications?[\s\S]{0,500}",
    ]
    
    for pattern in preferred_patterns:
        matches = re.finditer(pattern, normalized, re.IGNORECASE)
        for match in matches:
            preferred_sections.append(match.group(0))
    
    # If no specific section, look for "preferred" mentions throughout
    if not preferred_sections:
        # Look for sentences with "preferred"
        sentences = re.split(r'[.!?]+', normalized)
        preferred_sections = [s for s in sentences if "preferred" in s or "plus" in s or "bonus" in s]
    
    all_text = " ".join(preferred_sections) if preferred_sections else normalized
    
    # Extract bullet points from preferred sections
    bullet_pattern = r"[•\-\*]\s*([^•\-\*\n]+)"
    bullets = re.findall(bullet_pattern, all_text)
    
    # Extract skill patterns
    skill_patterns = [
        r"(\w+\s+\w+)\s+(?:experience|knowledge|familiarity|exposure)",
        r"(?:experience\s+with|knowledge\s+of|familiar\s+with)\s+(\w+(?:\s+\w+)?)",
    ]
    
    for pattern in skill_patterns:
        matches = re.finditer(pattern, all_text, re.IGNORECASE)
        for match in matches:
            term = match.group(1).strip().lower()
            if len(term) > 3:
                nice_to_haves.append(term)
    
    # Add common nice-to-have keywords
    nice_keywords = [
        "sql", "python", "excel", "power bi", "tableau", "agile", "scrum",
        "certification", "master", "advanced", "leadership", "communication"
    ]
    
    for keyword in nice_keywords:
        if keyword in normalized:
            nice_to_haves.append(keyword)
    
    # Deduplicate and clean
    nice_to_haves = list(set([nh.strip() for nh in nice_to_haves if len(nh.strip()) > 2]))
    
    return nice_to_haves[:25]


def extract_title_keywords(text: str, job_title: str = None) -> List[str]:
    """Extract title keywords for role alignment."""
    if job_title:
        # Extract keywords from job title
        title_words = re.findall(r'\b\w+\b', job_title.lower())
        return [w for w in title_words if len(w) > 3][:10]
    
    # Try to extract from text
    # Look for "Job Title:", "Position:", etc.
    title_patterns = [
        r"(?:job\s+title|position|role)[:\s]+([^\n]+)",
        r"^([A-Z][^:]+?)(?:\s+Job|\s+Position)",
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            title = match.group(1).strip()
            title_words = re.findall(r'\b\w+\b', title.lower())
            return [w for w in title_words if len(w) > 3][:10]
    
    # Fallback: extract common role-related words
    common_role_words = ["engineer", "analyst", "developer", "manager", "specialist", "coordinator"]
    found = [w for w in common_role_words if w in text.lower()]
    return found[:10]


def extract_job_title(text: str) -> str:
    """Extract job title from text."""
    title_patterns = [
        r"(?:job\s+title|position|role)[:\s]+([^\n]+)",
        r"^([A-Z][^:]+?)(?:\s+Job|\s+Position)",
        r"#\s*([^\n]+)",  # Markdown header
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    
    # Try first line if it looks like a title
    first_line = text.split('\n')[0].strip()
    if len(first_line) < 100 and not first_line.startswith('http'):
        return first_line
    
    return "Job Position"


def generate_jd_meta(
    text: str,
    output_path: Path,
    job_title: str = None,
    requisition_id: str = None
) -> None:
    """Generate jd_meta.yaml from job description text."""
    
    # Extract components
    if not job_title:
        job_title = extract_job_title(text)
    
    domain = detect_domain(text)
    must_haves = extract_must_haves(text)
    nice_to_haves = extract_nice_to_haves(text)
    title_keywords = extract_title_keywords(text, job_title)
    experience = extract_experience_requirements(text)
    
    # Generate summary
    summary = text[:200].replace('\n', ' ').strip() + "..."
    
    # Build YAML structure
    jd_meta = {
        "metadata_version": 1,
        "job": {
            "requisition_id": requisition_id or "AUTO-GENERATED",
            "title": job_title,
            "domain": domain,
            "summary": summary
        },
        "must_haves": must_haves,
        "nice_to_haves": nice_to_haves,
        "title_keywords": title_keywords,
        "experience": experience,
        "weights": {
            "must_have_coverage": 0.45,
            "skill_overlap": 0.25,
            "title_similarity": 0.15,
            "years_exp": 0.15
        },
        "thresholds": {
            "advance": 0.72,
            "review": 0.50
        },
        "notes": f"Auto-generated from job description. Domain: {domain}"
    }
    
    # Write YAML file
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(jd_meta, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"✓ Generated jd_meta.yaml at {output_path}")
    print(f"  Domain: {domain}")
    print(f"  Must-haves: {len(must_haves)}")
    print(f"  Nice-to-haves: {len(nice_to_haves)}")
    print(f"  Experience: {experience['min_years']}-{experience['max_years']} years")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Auto-extract job requirements and generate jd_meta.yaml"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to job description file (TXT) or '-' for stdin",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("jd_meta.yaml"),
        help="Output YAML file path (default: jd_meta.yaml)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Job title (auto-detected if not provided)",
    )
    parser.add_argument(
        "--req-id",
        type=str,
        default=None,
        help="Requisition ID (optional)",
    )
    args = parser.parse_args()
    
    # Read input
    if str(args.input) == "-":
        text = sys.stdin.read()
    else:
        if not args.input.exists():
            print(f"Error: File not found: {args.input}")
            return
        text = args.input.read_text(encoding="utf-8", errors="ignore")
    
    if not text.strip():
        print("Error: Empty job description")
        return
    
    # Generate metadata
    generate_jd_meta(
        text,
        args.output,
        job_title=args.title,
        requisition_id=args.req_id
    )


if __name__ == "__main__":
    import sys
    main()

