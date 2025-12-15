"""NLP-enhanced resume scoring with detailed per-candidate reports."""
import argparse
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import pandas as pd
import yaml

from parser import extract_text

try:
    import spacy
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

# Global NLP model (lazy loaded)
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
            return None
    return _nlp_model


def load_config(config_path: Path) -> Dict:
    """Load job description metadata from YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(text: str) -> str:
    """Normalize text for keyword matching."""
    return text.lower().strip()


def find_keywords_nlp(text: str, keyword: str, nlp_model) -> bool:
    """Use NLP to find if a keyword appears in context (more human-like)."""
    if not nlp_model:
        return False
    
    doc = nlp_model(text)
    keyword_lower = keyword.lower()
    keyword_words = keyword_lower.split()
    
    # For compound terms, check if words appear together in context
    if len(keyword_words) > 1:
        # Check if all words appear in the same sentence or nearby sentences
        for sent in doc.sents:
            sent_tokens = [t.text.lower() for t in sent]
            
            # Check if all keyword words appear in this sentence as full tokens
            if all(any(tok == word for tok in sent_tokens) for word in keyword_words):
                # Find positions of each word
                word_positions = []
                for word in keyword_words:
                    try:
                        # Find all occurrences of the word
                        positions = [i for i, w in enumerate(sent_tokens) if w == word]
                        if positions:
                            word_positions.append(positions)
                    except:
                        break
                
                # If we found all words, check if they're reasonably close
                if len(word_positions) == len(keyword_words):
                    # Check proximity - words should be within 8 tokens of each other
                    all_positions = [pos for positions in word_positions for pos in positions]
                    if max(all_positions) - min(all_positions) <= 8:
                        return True
        
        # Also check for variations (e.g., "financial modeling" vs "modeling financial")
        # by checking if words appear in noun chunks together
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            chunk_tokens = chunk_text.split()
            if all(word in chunk_tokens for word in keyword_words):
                return True
    
    # Check for lemmatized forms for single-word keywords only
    # (for multi-word phrases we already require full-token matches above)
    if len(keyword_words) == 1:
        keyword_doc = nlp_model(keyword)
        if keyword_doc:
            keyword_lemma = keyword_doc[0].lemma_ if len(keyword_doc) > 0 else keyword_lower
            for token in doc:
                if token.lemma_.lower() == keyword_lemma.lower():
                    return True
    
    return False


def find_keywords(text: str, keywords: List[str], use_nlp: bool = True) -> Set[str]:
    """Find which keywords appear in the text (case-insensitive, with optional NLP)."""
    normalized = normalize_text(text)
    found = set()
    nlp_model = get_nlp_model() if use_nlp else None
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        
        # Try NLP-enhanced matching first if available
        if nlp_model and use_nlp:
            if find_keywords_nlp(text, keyword, nlp_model):
                found.add(keyword_lower)
                continue
        
        # Fallback to regex matching
        # Handle multi-word keywords better
        if ' ' in keyword_lower:
            # For phrases, check if all words appear in order
            pattern = r"\b" + r"\s+".join([re.escape(word) for word in keyword_lower.split()]) + r"\b"
        else:
            # Single word - use word boundaries
            pattern = r"\b" + re.escape(keyword_lower) + r"\b"
        
        if re.search(pattern, normalized, re.IGNORECASE):
            found.add(keyword_lower)
    
    return found


def extract_years_experience(text: str) -> float:
    """Extract years of experience from resume text."""
    normalized = normalize_text(text)
    
    # Common patterns: "5 years", "5+ years", "5 years of experience", etc.
    patterns = [
        r"(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)",
        r"(\d+)\+?\s*years?\s*in",
        r"(\d+)\+?\s*years?\s*with",
        r"experience[:\s]+(\d+)\+?\s*years?",
    ]
    
    years_found = []
    for pattern in patterns:
        matches = re.findall(pattern, normalized)
        for match in matches:
            try:
                years_found.append(float(match))
            except ValueError:
                continue
    
    # Also look for ranges like "4-6 years"
    range_pattern = r"(\d+)\s*[-–]\s*(\d+)\s*years?"
    range_matches = re.findall(range_pattern, normalized)
    for start, end in range_matches:
        try:
            avg = (float(start) + float(end)) / 2
            years_found.append(avg)
        except ValueError:
            continue
    
    if years_found:
        return max(years_found)  # Take the highest value found
    return 0.0


def score_must_have_coverage(text: str, must_haves: List[str]) -> Tuple[float, List[str], List[str]]:
    """Score based on must-have keyword coverage (0-1)."""
    found_lower = find_keywords(text, must_haves)
    total = len(must_haves)
    if total == 0:
        return 1.0, [], []
    
    found = [mh for mh in must_haves if mh.lower() in found_lower]
    score = len(found) / total
    missing = [mh for mh in must_haves if mh.lower() not in found_lower]
    return score, missing, found


def score_skill_overlap(text: str, nice_to_haves: List[str]) -> Tuple[float, List[str]]:
    """Score based on nice-to-have keyword overlap (0-1)."""
    found_lower = find_keywords(text, nice_to_haves)
    total = len(nice_to_haves)
    if total == 0:
        return 0.0, []
    
    found = [nh for nh in nice_to_haves if nh.lower() in found_lower]
    # Cap at 1.0, but allow bonus for many matches
    score = min(len(found) / max(total * 0.5, 10), 1.0)  # Normalize to reasonable range
    return score, found


def score_title_similarity(text: str, title_keywords: List[str]) -> float:
    """Score based on title keyword presence (0-1)."""
    found = find_keywords(text, title_keywords)
    total = len(title_keywords)
    if total == 0:
        return 0.0
    
    return len(found) / total


def score_years_experience(years: float, min_years: int, preferred_years: int, max_years: int) -> float:
    """Score based on years of experience (0-1)."""
    if years < min_years:
        return years / min_years  # Partial credit for less than minimum
    elif years <= preferred_years:
        return 1.0  # Full credit for preferred range
    elif years <= max_years:
        # Slight penalty for being at max
        return 1.0 - ((years - preferred_years) / (max_years - preferred_years)) * 0.2
    else:
        # Penalty for exceeding max
        return max(0.5, 1.0 - (years - max_years) * 0.1)


def calculate_final_score(
    must_have_score: float,
    skill_score: float,
    title_score: float,
    years_score: float,
    weights: Dict[str, float],
) -> float:
    """Calculate weighted final score."""
    return (
        must_have_score * weights["must_have_coverage"]
        + skill_score * weights["skill_overlap"]
        + title_score * weights["title_similarity"]
        + years_score * weights["years_exp"]
    )


def determine_recommendation(
    score: float,
    all_must_haves_present: bool,
    advance_threshold: float,
    review_threshold: float,
) -> str:
    """Determine recommendation based on score and must-haves."""
    if score >= advance_threshold and all_must_haves_present:
        return "Advance"
    elif score >= review_threshold:
        return "Review"
    else:
        return "Reject"


def generate_explanation(
    must_have_score: float,
    skill_score: float,
    title_score: float,
    years_score: float,
    years_found: float,
    missing_must_haves: List[str],
    found_skills: List[str],
    all_must_haves_present: bool,
) -> str:
    """Generate human-readable explanation of the score."""
    parts = []
    
    # Must-have coverage
    mh_pct = int(must_have_score * 100)
    parts.append(f"Must-have coverage: {mh_pct}% ({len(missing_must_haves)} missing)")
    if missing_must_haves:
        parts.append(f"  Missing: {', '.join(missing_must_haves[:3])}")
        if len(missing_must_haves) > 3:
            parts.append(f"  ... and {len(missing_must_haves) - 3} more")
    
    # Skill overlap
    skill_pct = int(skill_score * 100)
    parts.append(f"Skill overlap: {skill_pct}% ({len(found_skills)} nice-to-haves found)")
    if found_skills:
        parts.append(f"  Found: {', '.join(found_skills[:5])}")
        if len(found_skills) > 5:
            parts.append(f"  ... and {len(found_skills) - 5} more")
    
    # Title similarity
    title_pct = int(title_score * 100)
    parts.append(f"Title similarity: {title_pct}%")
    
    # Years experience
    years_pct = int(years_score * 100)
    parts.append(f"Years experience: {years_pct}% ({years_found:.1f} years found)")
    
    # Overall assessment
    if all_must_haves_present:
        parts.append("✓ All must-haves present")
    else:
        parts.append("✗ Some must-haves missing")
    
    return " | ".join(parts)


def generate_candidate_report(
    resume_name: str,
    must_haves: List[str],
    nice_to_haves: List[str],
    found_must_haves: List[str],
    found_nice_to_haves: List[str],
    reports_dir: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Create CSV and Excel reports showing binary coverage for each requirement."""
    if not reports_dir:
        return None, None
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    found_must = {mh.lower() for mh in found_must_haves}
    found_nice = {nh.lower() for nh in found_nice_to_haves}
    
    rows = []
    for mh in must_haves:
        rows.append({
            "requirement": mh,
            "category": "must_have",
            "met_binary": 1 if mh.lower() in found_must else 0,
            "met": "Yes" if mh.lower() in found_must else "No",
        })
    
    for nh in nice_to_haves:
        rows.append({
            "requirement": nh,
            "category": "nice_to_have",
            "met_binary": 1 if nh.lower() in found_nice else 0,
            "met": "Yes" if nh.lower() in found_nice else "No",
        })
    
    df = pd.DataFrame(rows)
    resume_stem = Path(resume_name).stem
    csv_path = reports_dir / f"{resume_stem}_requirements.csv"
    excel_path = reports_dir / f"{resume_stem}_requirements.xlsx"
    
    df.to_csv(csv_path, index=False)
    
    try:
        df.to_excel(excel_path, index=False)
    except Exception as exc:
        print(f"Warning: failed to write Excel report for {resume_name}: {exc}")
        excel_path = None
    
    return csv_path, excel_path


def score_resume(resume_path: Path, config: Dict) -> Dict:
    """Score a single resume against the job description.

    Optionally uses an LLM (Ollama) to interpret whether each requirement is met,
    falling back to NLP/rule-based scoring when LLM is unavailable or fails.
    """
    try:
        text = extract_text(resume_path)
    except Exception as e:
        return {
            "file": resume_path.name,
            "score": 0.0,
            "recommendation": "Reject",
            "explanation": f"Error parsing file: {e}",
            "years_found": 0.0,
            "must_haves_present": 0,
            "must_haves_total": len(config.get("must_haves", [])),
        }
    
    # Extract components
    must_haves = config.get("must_haves", [])
    nice_to_haves = config.get("nice_to_haves", [])
    title_keywords = config.get("title_keywords", [])
    experience = config.get("experience", {})
    weights = config.get("weights", {})
    thresholds = config.get("thresholds", {})
    
    # Optional: LLM-powered requirement evaluation
    llm_result = None
    use_llm_resume = config.get("_use_llm_resume", False)
    if use_llm_resume:
        try:
            from jd_analyzer_llm import check_ollama_available
            from resume_analyzer_llm import analyze_resume_with_ollama

            if check_ollama_available():
                llm_result = analyze_resume_with_ollama(text, config)
            else:
                print("Ollama not available for resume analysis; using NLP-based scoring.")
        except Exception as exc:
            print(f"Warning: LLM resume analysis failed for {resume_path.name}: {exc}")
            llm_result = None

    # Calculate component scores
    if llm_result is not None:
        mh_decisions = llm_result.get("must_haves", {})
        nh_decisions = llm_result.get("nice_to_haves", {})

        # Must-haves
        found_must_haves = [mh for mh in must_haves if mh_decisions.get(mh, {}).get("met")]
        missing_must_haves = [mh for mh in must_haves if mh not in found_must_haves]
        total_mh = len(must_haves)
        must_have_score = len(found_must_haves) / total_mh if total_mh > 0 else 1.0

        # Nice-to-haves
        found_skills = [nh for nh in nice_to_haves if nh_decisions.get(nh, {}).get("met")]
        total_nh = len(nice_to_haves)
        if total_nh == 0:
            skill_score = 0.0
        else:
            # Reuse the same normalization logic as score_skill_overlap
            skill_score = min(len(found_skills) / max(total_nh * 0.5, 10), 1.0)
    else:
        must_have_score, missing_must_haves, found_must_haves = score_must_have_coverage(text, must_haves)
        skill_score, found_skills = score_skill_overlap(text, nice_to_haves)
    title_score = score_title_similarity(text, title_keywords)
    
    years_found = extract_years_experience(text)
    years_score = score_years_experience(
        years_found,
        experience.get("min_years", 4),
        experience.get("preferred_years", 5),
        experience.get("max_years", 6),
    )
    
    # Calculate final score
    final_score = calculate_final_score(
        must_have_score, skill_score, title_score, years_score, weights
    )
    
    # Determine recommendation
    all_must_haves_present = len(missing_must_haves) == 0
    recommendation = determine_recommendation(
        final_score,
        all_must_haves_present,
        thresholds.get("advance", 0.72),
        thresholds.get("review", 0.50),
    )
    
    # Generate explanation
    explanation = generate_explanation(
        must_have_score,
        skill_score,
        title_score,
        years_score,
        years_found,
        missing_must_haves,
        found_skills,
        all_must_haves_present,
    )

    # Append any LLM-derived summary/risks if available
    if llm_result is not None:
        extra_parts = []
        summary = llm_result.get("summary", "")
        if summary:
            extra_parts.append(f"LLM summary: {summary}")
        risks = llm_result.get("risks") or []
        if risks:
            extra_parts.append("LLM risk flags: " + "; ".join(str(r) for r in risks[:5]))
        if extra_parts:
            explanation = explanation + " | " + " | ".join(extra_parts)
    
    return {
        "file": resume_path.name,
        "score": round(final_score, 3),
        "recommendation": recommendation,
        "explanation": explanation,
        "years_found": round(years_found, 1),
        "must_haves_present": len(must_haves) - len(missing_must_haves),
        "must_haves_total": len(must_haves),
        "must_have_coverage": round(must_have_score, 3),
        "skill_overlap": round(skill_score, 3),
        "title_similarity": round(title_score, 3),
        "years_score": round(years_score, 3),
        "found_must_haves": found_must_haves,
        "found_nice_to_haves": found_skills,
        "missing_must_haves": missing_must_haves,
    }


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Score resumes against job description (SCS-2237843)"
    )
    parser.add_argument(
        "resumes",
        type=Path,
        nargs="+",
        help="Path(s) to resume files (PDF/DOCX/TXT) or directory containing resumes",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("jd_meta.yaml"),
        help="Path to job description metadata YAML (default: jd_meta.yaml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scores.csv"),
        help="Output CSV file path (default: scores.csv)",
    )
    parser.add_argument(
        "--llm-resume",
        action="store_true",
        help="Use Ollama LLM to interpret resumes (falls back to NLP when unavailable)",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("candidate_reports"),
        help="Directory to store per-candidate requirement reports (CSV & Excel)",
    )
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Disable generation of per-candidate requirement reports",
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)

    # Attach runtime flag so score_resume can see whether to use LLM
    if args.llm_resume:
        config = dict(config)  # shallow copy to avoid mutating original object
        config["_use_llm_resume"] = True
    
    # Collect resume files (deduplicate)
    resume_files = []
    seen = set()
    for path in args.resumes:
        path = Path(path)
        if path.is_file():
            if path.suffix.lower() in {".pdf", ".docx", ".txt"}:
                if path not in seen:
                    resume_files.append(path)
                    seen.add(path)
        elif path.is_dir():
            for ext in ["*.pdf", "*.docx", "*.txt"]:
                for file_path in path.glob(ext):
                    if file_path not in seen:
                        resume_files.append(file_path)
                        seen.add(file_path)
                for file_path in path.glob(ext.upper()):
                    if file_path not in seen:
                        resume_files.append(file_path)
                        seen.add(file_path)
    
    if not resume_files:
        print("No resume files found!")
        return
    
    print(f"Processing {len(resume_files)} resume(s)...")
    
    # Score each resume
    results = []
    for resume_path in resume_files:
        print(f"  Scoring {resume_path.name}...")
        result = score_resume(resume_path, config)
        
        # Generate per-candidate requirement reports unless disabled
        if not args.no_reports and args.reports_dir:
            csv_report, excel_report = generate_candidate_report(
                resume_path.name,
                config.get("must_haves", []),
                config.get("nice_to_haves", []),
                result.get("found_must_haves", []),
                result.get("found_nice_to_haves", []),
                args.reports_dir,
            )
            if csv_report:
                result["requirement_report_csv"] = str(csv_report)
            if excel_report:
                result["requirement_report_excel"] = str(excel_report)
        
        results.append(result)
    
    # Create DataFrame and sort by score
    summary_rows = []
    for res in results:
        summary = res.copy()
        summary.pop("found_must_haves", None)
        summary.pop("found_nice_to_haves", None)
        summary.pop("missing_must_haves", None)
        summary_rows.append(summary)
    
    df = pd.DataFrame(summary_rows)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    print(f"\nTop candidates:")
    print(df[["file", "score", "recommendation", "years_found", "must_haves_present"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
