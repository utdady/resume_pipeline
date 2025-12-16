"""Auto-extract job requirements from any job description and generate jd_meta.yaml.
Enhanced with NLP using spaCy for human-like understanding."""
import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import yaml

from jd_analyzer_llm import analyze_jd_hybrid

logger = logging.getLogger(__name__)

try:
    import spacy
    from spacy import displacy
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    logger.warning(
        "spaCy not installed. Install with: pip install spacy && python -m spacy download en_core_web_sm"
    )
    logger.warning("Falling back to basic regex parsing...")

# Generic terms to filter out from extracted requirements
GENERIC_TERMS_MUST_HAVE = {
    "years", "experience", "knowledge", "proficiency", "required", 
    "must", "have", "essential", "and", "the", "with", "of", "in",
    "minimum", "qualifications", "requirements", "troubleshoot",
    "exposure", "machine", "web", "detail", "build", "company",
    "responsibilities", "applications", "teams", "participate",
    "reviews", "platforms", "qualifications", "skills", "presentations",
    "transactions", "research", "forecasts", "projections", "level"
}

GENERIC_TERMS_NICE_TO_HAVE = {
    "years", "experience", "knowledge", "proficiency", "preferred",
    "nice", "have", "plus", "bonus", "and", "the", "with", "of", "in",
    "additional", "qualifications", "troubleshoot", "exposure", "machine",
    "web", "detail", "build", "company", "responsibilities", "applications",
    "teams", "participate", "reviews", "platforms", "qualifications",
    "skills", "presentations", "transactions", "research", "forecasts",
    "projections", "level", "knowledge", "methodology"
}


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

# Load spaCy model (lazy loading)
_nlp_model = None

def get_nlp_model():
    """Lazy load spaCy model."""
    global _nlp_model
    if not NLP_AVAILABLE:
        return None
    if _nlp_model is None:
        try:
            _nlp_model = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found.")
            logger.warning("Install with: python -m spacy download en_core_web_sm")
            return None
    return _nlp_model


def detect_domain(text: str) -> str:
    """Detect the primary domain of the job description using NLP-enhanced analysis."""
    normalized = text.lower()
    domain_scores = {}
    
    # Use NLP to understand context better
    nlp = get_nlp_model()
    if nlp:
        doc = nlp(text)
        # Look for domain-related entities and keywords in context
        for ent in doc.ents:
            ent_text = ent.text.lower()
            for domain, keywords in DOMAIN_KEYWORDS.items():
                if any(kw in ent_text for kw in keywords):
                    domain_scores[domain] = domain_scores.get(domain, 0) + 2
    
    # Check financial domain first (more specific)
    if "financial analyst" in normalized or "investment banking" in normalized:
        financial_score = sum(1 for keyword in DOMAIN_KEYWORDS["financial"] if keyword in normalized)
        if financial_score > 2:
            return "financial"
    
    # Traditional keyword matching
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in normalized)
        if score > 0:
            domain_scores[domain] = domain_scores.get(domain, 0) + score
    
    if domain_scores:
        return max(domain_scores.items(), key=lambda x: x[1])[0]
    return "general"


def extract_experience_requirements(text: str) -> Dict:
    """Extract years of experience requirements using NLP for better understanding."""
    normalized = text.lower()
    
    nlp = get_nlp_model()
    if nlp:
        doc = nlp(text)
        # Use dependency parsing to find experience requirements
        for sent in doc.sents:
            # Look for numeric entities and their relationships to "experience"
            for token in sent:
                if token.like_num and token.text.isdigit():
                    # Check if this number is related to experience
                    for child in token.children:
                        if "year" in child.text.lower() or "experience" in child.text.lower():
                            years = int(token.text)
                            # Check for ranges safely (avoid IndexError at sentence boundaries)
                            try:
                                if token.nbor(1).text in ["-", "–", "to"]:
                                    try:
                                        end_years = int(token.nbor(2).text)
                                        return {
                                            "min_years": min(years, end_years),
                                            "max_years": max(years, end_years),
                                            "preferred_years": int((years + end_years) / 2),
                                        }
                                    except ValueError:
                                        # Not a clean integer after the range marker; fall back to single value
                                        pass
                            except IndexError:
                                # No neighbor tokens available for range detection; ignore and treat as single value
                                pass
                            return {
                                "min_years": years,
                                "max_years": years + 2,
                                "preferred_years": years
                            }
    
    # Fallback to regex patterns
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
                start, end = int(groups[0]), int(groups[1])
                min_years = min(start, end) if min_years is None else min(min_years, start)
                max_years = max(start, end) if max_years is None else max(max_years, end)
                preferred_years = (start + end) / 2
            elif len(groups) == 1:
                years = int(groups[0])
                if min_years is None or years < min_years:
                    min_years = years
                if max_years is None or years > max_years:
                    max_years = years
    
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


def extract_skills_with_nlp(text: str, section_text: str, is_required: bool = True) -> List[str]:
    """Extract skills and technologies using NLP dependency parsing."""
    skills = []
    nlp = get_nlp_model()
    
    if not nlp:
        return skills
    
    doc = nlp(section_text)
    
    # Common skill-related patterns in dependency trees
    skill_indicators = {
        "experience with", "proficiency in", "knowledge of", "familiarity with",
        "expertise in", "skills in", "ability to", "capable of"
    }
    
    # Extract noun phrases that are likely skills/technologies
    for sent in doc.sents:
        # Look for technical terms (often proper nouns or compound nouns)
        for token in sent:
            # Skills are often nouns or proper nouns
            if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
                # Check if it's part of a skill phrase
                skill_text = token.text
                
                # Check for compound nouns (e.g., "financial modeling", "machine learning")
                if token.head.pos_ == "NOUN" and token.dep_ in ["compound", "amod"]:
                    # Get the full compound phrase
                    phrase_tokens = [t.text for t in token.subtree if t.pos_ in ["NOUN", "PROPN", "ADJ"]]
                    if len(phrase_tokens) > 1:
                        skill_text = " ".join(phrase_tokens[:3])  # Limit to 3-word phrases
                
                # Check context - is this mentioned with skill indicators?
                sent_lower = sent.text.lower()
                if any(indicator in sent_lower for indicator in skill_indicators):
                    if len(skill_text) > 2 and skill_text.lower() not in ["years", "experience", "degree"]:
                        skills.append(skill_text.lower())
                
                # Also check for direct object relationships (e.g., "use Python", "know SQL")
                if token.dep_ == "dobj" or token.dep_ == "pobj":
                    # Check if the verb is skill-related
                    verb = token.head
                    if verb.pos_ == "VERB" and verb.lemma_ in ["use", "know", "have", "possess", "utilize"]:
                        if len(skill_text) > 2:
                            skills.append(skill_text.lower())
    
    # Extract entities that might be technologies/skills
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "EVENT"]:
            ent_text = ent.text.lower()
            # Filter out common non-skill entities
            if ent_text not in ["years", "experience", "degree", "bachelor", "master"]:
                if len(ent_text) > 2:
                    skills.append(ent_text)
    
    # Extract from bullet points more intelligently
    bullet_pattern = r"[•\-\*]\s*([^•\-\*\n]+)"
    bullets = re.findall(bullet_pattern, section_text)
    
    for bullet in bullets:
        bullet_doc = nlp(bullet)
        # Extract the main noun phrases from bullet points
        for chunk in bullet_doc.noun_chunks:
            chunk_text = chunk.text.lower().strip()
            # Filter out generic terms
            if (len(chunk_text) > 3 and 
                chunk_text not in ["years of", "experience in", "knowledge of", "the", "and", "or"]):
                # Check if it contains a skill indicator
                if any(word in chunk_text for word in ["experience", "proficiency", "knowledge", "skills", "degree"]):
                    # Extract the actual skill (usually after "in" or "with")
                    parts = re.split(r"\s+(?:in|with|of)\s+", chunk_text)
                    if len(parts) > 1:
                        skill = parts[-1].strip()
                        if len(skill) > 2:
                            skills.append(skill)
                else:
                    # Might be a skill itself
                    if len(chunk_text.split()) <= 3:  # Limit to 3-word phrases
                        skills.append(chunk_text)
    
    return list(set(skills))


def extract_must_haves(text: str) -> List[str]:
    """Extract must-have requirements using NLP for better understanding."""
    normalized = text.lower()
    must_haves = []
    
    # Find sections that typically contain requirements
    requirement_sections = []
    
    # Find sections with "required", "must have", "essential", etc.
    section_patterns = [
        r"(?:required|must have|essential|mandatory|qualifications|requirements)[\s\S]{0,1000}",
        r"minimum\s+requirements?[\s\S]{0,1000}",
        r"basic\s+qualifications?[\s\S]{0,1000}",
    ]
    
    for pattern in section_patterns:
        matches = re.finditer(pattern, normalized, re.IGNORECASE)
        for match in matches:
            requirement_sections.append(match.group(0))
    
    # If no specific section found, use entire text but focus on requirement-like sentences
    if not requirement_sections:
        nlp = get_nlp_model()
        if nlp:
            doc = nlp(text)
            # Find sentences with requirement indicators
            for sent in doc.sents:
                sent_lower = sent.text.lower()
                if any(word in sent_lower for word in ["required", "must", "essential", "mandatory", "minimum"]):
                    requirement_sections.append(sent.text)
        else:
            requirement_sections = [normalized]
    
    all_text = " ".join(requirement_sections)
    
    # Use NLP to extract skills
    nlp_skills = extract_skills_with_nlp(text, all_text, is_required=True)
    must_haves.extend(nlp_skills)
    
    # Also extract using traditional patterns for compatibility
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
            if len(phrase) > 3:
                extracted_phrases.append(phrase)
    
    # Extract bullet points
    bullet_pattern = r"[•\-\*]\s*([^•\-\*\n]+)"
    bullets = re.findall(bullet_pattern, all_text)
    
    # Process bullets with NLP - extract actual skills and technologies
    for bullet in bullets:
        bullet_clean = bullet.strip()
        if len(bullet_clean) > 5:
            # Extract key information from bullet
            nlp = get_nlp_model()
            if nlp:
                bullet_doc = nlp(bullet_clean)
                
                # Look for specific patterns: "X in Y", "X with Y", "experience with Y"
                # Extract the technology/skill (usually the object)
                for token in bullet_doc:
                    # Find technologies/tools (often proper nouns or specific nouns)
                    if token.pos_ == "PROPN" or (token.pos_ == "NOUN" and token.text[0].isupper()):
                        skill = token.text.lower()
                        # Check if it's part of a compound (e.g., "Python programming")
                        if token.head.pos_ in ["NOUN", "VERB"]:
                            # Get compound phrase
                            compound = [t.text.lower() for t in token.subtree 
                                       if t.pos_ in ["NOUN", "PROPN", "ADJ"] and not t.is_stop]
                            if len(compound) > 1:
                                skill = " ".join(compound[:2])
                        if len(skill) > 2 and skill not in ["years", "experience", "degree"]:
                            must_haves.append(skill)
                    
                    # Extract skills from "proficiency in X", "experience with X", etc.
                    if token.lemma_ in ["proficiency", "experience", "knowledge", "familiarity"]:
                        # Find the object of this preposition
                        for child in token.children:
                            if child.dep_ == "prep":  # preposition
                                for grandchild in child.children:
                                    if grandchild.dep_ == "pobj":  # object of preposition
                                        skill = grandchild.text.lower()
                                        # Get compound if exists
                                        if grandchild.head.pos_ in ["NOUN", "PROPN"]:
                                            compound = [t.text.lower() for t in grandchild.subtree 
                                                       if t.pos_ in ["NOUN", "PROPN", "ADJ"] and not t.is_stop]
                                            if len(compound) > 1:
                                                skill = " ".join(compound[:2])
                                        if len(skill) > 2:
                                            must_haves.append(skill)
                
                # Also extract noun phrases that look like skills
                for chunk in bullet_doc.noun_chunks:
                    chunk_text = chunk.text.lower().strip()
                    # Skip if it's too generic
                    if (len(chunk_text) > 3 and 
                        chunk_text not in ["years of", "experience in", "knowledge of", 
                                          "the", "and", "or", "minimum", "required"]):
                        # Extract skill from patterns like "X experience", "X proficiency"
                        if "experience" in chunk_text or "proficiency" in chunk_text:
                            parts = re.split(r"\s+(?:in|with|of)\s+", chunk_text)
                            if len(parts) > 1:
                                skill = parts[-1].strip()
                                if len(skill) > 2:
                                    must_haves.append(skill)
                        elif "degree" in chunk_text:
                            # Extract degree field
                            parts = re.split(r"\s+degree\s+(?:in\s+)?", chunk_text)
                            if len(parts) > 1:
                                field = parts[-1].strip()
                                if len(field) > 2:
                                    must_haves.append(f"bachelor's degree in {field}")
                        else:
                            # Might be a skill itself if it's a technical term
                            if len(chunk_text.split()) <= 3:
                                must_haves.append(chunk_text)
            else:
                # Fallback: extract key terms
                words = re.findall(r'\b\w+\b', bullet_clean.lower())
                # Remove stop words
                stop_words = {"the", "a", "an", "and", "or", "in", "on", "at", "with", "for", "of", "minimum", "required"}
                key_words = [w for w in words if w not in stop_words and len(w) > 3]
                if key_words:
                    must_haves.append(" ".join(key_words[:3]))
    
    # Extract degree requirements
    degree_patterns = [
        r"(?:bachelor|master|phd|bs|ms|ph\.?d\.?)\s+(?:degree\s+)?(?:in\s+)?([^,\.]+)",
    ]
    for pattern in degree_patterns:
        matches = re.finditer(pattern, all_text, re.IGNORECASE)
        for match in matches:
            degree_field = match.group(1).strip().lower()
            if len(degree_field) > 2:
                must_haves.append(f"bachelor's degree in {degree_field}")
    
    # Domain-specific must-haves
    domain = detect_domain(text)
    domain_keywords = {
        "electrical": ["cape", "aspen", "protection", "short circuit", "power systems", 
                       "transmission", "relay", "breaker", "substation"],
        "software": ["programming", "coding", "development", "api", "javascript", 
                     "python", "java", "react", "node"],
        "data": ["sql", "python", "analytics", "statistics", "machine learning"],
        "financial": ["excel", "financial modeling", "financial analysis", "valuation", 
                      "dcf", "financial statements", "accounting", "cfa", "mba"],
        "mechanical": ["cad", "solidworks", "autocad", "manufacturing", "design"]
    }
    
    if domain in domain_keywords:
        for keyword in domain_keywords[domain]:
            if keyword in all_text:
                must_haves.append(keyword)
    
    # Deduplicate and clean
    must_haves = list(set([mh.strip() for mh in must_haves if len(mh.strip()) > 2]))
    
    # Filter out generic terms and single-word generic nouns
    # More aggressive filtering
    filtered_must_haves = []
    for mh in must_haves:
        mh_lower = mh.lower()
        # Skip if it's a generic term or contains only generic words
        if mh_lower in GENERIC_TERMS_MUST_HAVE:
            continue
        # Skip single words that are too generic (unless they're technical terms)
        if len(mh.split()) == 1 and mh_lower in ["master", "bachelor", "degree"]:
            continue
        # Skip if it's mostly generic words
        words = mh_lower.split()
        generic_word_count = sum(1 for w in words if w in GENERIC_TERMS_MUST_HAVE)
        if len(words) > 0 and generic_word_count / len(words) > 0.5:
            continue
        filtered_must_haves.append(mh)
    
    must_haves = filtered_must_haves
    
    # Limit to top 15 most relevant
    return must_haves[:15]


def extract_nice_to_haves(text: str) -> List[str]:
    """Extract nice-to-have/preferred requirements using NLP."""
    normalized = text.lower()
    nice_to_haves = []
    
    # Find sections with "preferred", "nice to have", "plus", etc.
    preferred_sections = []
    
    preferred_patterns = [
        r"(?:preferred|nice to have|plus|bonus|advantage)[\s\S]{0,1000}",
        r"additional\s+qualifications?[\s\S]{0,1000}",
    ]
    
    for pattern in preferred_patterns:
        matches = re.finditer(pattern, normalized, re.IGNORECASE)
        for match in matches:
            preferred_sections.append(match.group(0))
    
    # If no specific section, use NLP to find preferred sentences
    if not preferred_sections:
        nlp = get_nlp_model()
        if nlp:
            doc = nlp(text)
            for sent in doc.sents:
                sent_lower = sent.text.lower()
                if any(word in sent_lower for word in ["preferred", "nice to have", "plus", "bonus", "advantage"]):
                    preferred_sections.append(sent.text)
        else:
            sentences = re.split(r'[.!?]+', normalized)
            preferred_sections = [s for s in sentences if "preferred" in s or "plus" in s or "bonus" in s]
    
    all_text = " ".join(preferred_sections) if preferred_sections else normalized
    
    # Use NLP to extract skills
    nlp_skills = extract_skills_with_nlp(text, all_text, is_required=False)
    nice_to_haves.extend(nlp_skills)
    
    # Extract bullet points
    bullet_pattern = r"[•\-\*]\s*([^•\-\*\n]+)"
    bullets = re.findall(bullet_pattern, all_text)
    
    # Process bullets with NLP
    for bullet in bullets:
        bullet_clean = bullet.strip()
        if len(bullet_clean) > 5:
            nlp = get_nlp_model()
            if nlp:
                bullet_doc = nlp(bullet_clean)
                important_words = [t.text.lower() for t in bullet_doc 
                                 if t.pos_ in ["NOUN", "PROPN"] and not t.is_stop]
                if important_words:
                    if len(important_words) >= 2:
                        phrase = " ".join(important_words[:3])
                        if len(phrase) > 3:
                            nice_to_haves.append(phrase)
                    else:
                        if len(important_words[0]) > 3:
                            nice_to_haves.append(important_words[0])
            else:
                words = re.findall(r'\b\w+\b', bullet_clean.lower())
                stop_words = {"the", "a", "an", "and", "or", "in", "on", "at", "with", "for", "of", "preferred"}
                key_words = [w for w in words if w not in stop_words and len(w) > 3]
                if key_words:
                    nice_to_haves.append(" ".join(key_words[:3]))
    
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
    
    # Add common nice-to-have keywords if mentioned
    nice_keywords = [
        "sql", "python", "excel", "power bi", "tableau", "agile", "scrum",
        "certification", "master", "advanced", "leadership", "communication",
        "aws", "azure", "docker", "kubernetes", "machine learning", "ai"
    ]
    
    for keyword in nice_keywords:
        if keyword in normalized:
            nice_to_haves.append(keyword)
    
    # Deduplicate and clean
    nice_to_haves = list(set([nh.strip() for nh in nice_to_haves if len(nh.strip()) > 2]))
    
    # Filter out generic terms
    filtered_nice_to_haves = []
    for nh in nice_to_haves:
        nh_lower = nh.lower()
        if nh_lower in GENERIC_TERMS_NICE_TO_HAVE:
            continue
        # Skip if it's mostly generic words
        words = nh_lower.split()
        generic_word_count = sum(1 for w in words if w in GENERIC_TERMS_NICE_TO_HAVE)
        if len(words) > 0 and generic_word_count / len(words) > 0.5:
            continue
        filtered_nice_to_haves.append(nh)
    
    nice_to_haves = filtered_nice_to_haves
    
    return nice_to_haves[:25]


def extract_title_keywords(text: str, job_title: str = None) -> List[str]:
    """Extract title keywords for role alignment using NLP."""
    if job_title:
        nlp = get_nlp_model()
        if nlp:
            doc = nlp(job_title)
            # Extract important nouns and proper nouns
            keywords = [t.text.lower() for t in doc if t.pos_ in ["NOUN", "PROPN"] and len(t.text) > 3]
            return keywords[:10]
        else:
            title_words = re.findall(r'\b\w+\b', job_title.lower())
            return [w for w in title_words if len(w) > 3][:10]
    
    # Try to extract from text
    title_patterns = [
        r"(?:job\s+title|position|role)[:\s]+([^\n]+)",
        r"^([A-Z][^:]+?)(?:\s+Job|\s+Position)",
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            title = match.group(1).strip()
            nlp = get_nlp_model()
            if nlp:
                doc = nlp(title)
                keywords = [t.text.lower() for t in doc if t.pos_ in ["NOUN", "PROPN"] and len(t.text) > 3]
                return keywords[:10]
            else:
                title_words = re.findall(r'\b\w+\b', title.lower())
                return [w for w in title_words if len(w) > 3][:10]
    
    # Fallback
    common_role_words = ["engineer", "analyst", "developer", "manager", "specialist", "coordinator"]
    found = [w for w in common_role_words if w in text.lower()]
    return found[:10]


def extract_job_title(text: str) -> str:
    """Extract job title from text using NLP."""
    title_patterns = [
        r"(?:job\s+title|position|role)[:\s]+([^\n]+)",
        r"^([A-Z][^:]+?)(?:\s+Job|\s+Position)",
        r"#\s*([^\n]+)",  # Markdown header
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    
    # Use NLP to find the title in first few sentences
    nlp = get_nlp_model()
    if nlp:
        first_para = text.split('\n\n')[0] if '\n\n' in text else text.split('\n')[0]
        doc = nlp(first_para)
        # Look for proper nouns or noun phrases at the start
        for sent in doc.sents[:2]:  # Check first 2 sentences
            # If sentence is short and contains role words, it might be the title
            if len(sent.text) < 100:
                role_words = ["engineer", "analyst", "developer", "manager", "specialist", "director"]
                if any(word in sent.text.lower() for word in role_words):
                    return sent.text.strip()
    
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
    
    # Generate summary using NLP if available
    nlp = get_nlp_model()
    if nlp:
        doc = nlp(text)
        # Get first sentence or first 200 chars
        if doc.sents:
            summary = next(doc.sents).text
            if len(summary) > 200:
                summary = summary[:200] + "..."
        else:
            summary = text[:200].replace('\n', ' ').strip() + "..."
    else:
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
        "notes": f"Auto-generated from job description using NLP. Domain: {domain}"
    }
    
    # Write YAML file
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(jd_meta, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    logger.info("Generated jd_meta.yaml at %s", output_path)
    logger.info(
        "Domain: %s | Must-haves: %d | Nice-to-haves: %d | Experience: %s-%s years",
        domain,
        len(must_haves),
        len(nice_to_haves),
        experience["min_years"],
        experience["max_years"],
    )
    if NLP_AVAILABLE and get_nlp_model():
        logger.info("Using NLP-enhanced parsing")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Auto-extract job requirements and generate jd_meta.yaml (NLP-enhanced, Ollama-aware)"
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
    parser.add_argument(
        "--llm",
        dest="llm",
        action="store_true",
        default=False,
        help="Use Ollama LLM for analysis (off by default; falls back to rules if unavailable)",
    )
    parser.add_argument(
        "--no-llm",
        dest="llm",
        action="store_false",
        help="Force rule-based analysis (skip LLM)",
    )
    args = parser.parse_args()

    # Basic logging configuration for CLI usage
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    
    # Read input
    if str(args.input) == "-":
        text = sys.stdin.read()
    else:
        if not args.input.exists():
            logger.error("Error: File not found: %s", args.input)
            return
        text = args.input.read_text(encoding="utf-8", errors="ignore")
    
    if not text.strip():
        logger.error("Error: Empty job description")
        return
    
    # Generate metadata (LLM hybrid)
    analyze_jd_hybrid(
        text,
        args.output,
        use_llm=args.llm,
        job_title=args.title,
        requisition_id=args.req_id,
    )


if __name__ == "__main__":
    main()
