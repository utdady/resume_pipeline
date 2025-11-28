"""Rule-based resume scoring for Electrical Engineer 2 - Protection & Control position."""
import argparse
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import yaml

from parser import extract_text


def load_config(config_path: Path) -> Dict:
    """Load job description metadata from YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(text: str) -> str:
    """Normalize text for keyword matching."""
    return text.lower().strip()


def find_keywords(text: str, keywords: List[str]) -> Set[str]:
    """Find which keywords appear in the text (case-insensitive)."""
    normalized = normalize_text(text)
    found = set()
    for keyword in keywords:
        # Use word boundaries for better matching
        pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
        if re.search(pattern, normalized):
            found.add(keyword.lower())
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


def score_must_have_coverage(text: str, must_haves: List[str]) -> Tuple[float, List[str]]:
    """Score based on must-have keyword coverage (0-1)."""
    found = find_keywords(text, must_haves)
    total = len(must_haves)
    if total == 0:
        return 1.0, []
    
    score = len(found) / total
    missing = [mh for mh in must_haves if mh.lower() not in found]
    return score, missing


def score_skill_overlap(text: str, nice_to_haves: List[str]) -> Tuple[float, List[str]]:
    """Score based on nice-to-have keyword overlap (0-1)."""
    found = find_keywords(text, nice_to_haves)
    total = len(nice_to_haves)
    if total == 0:
        return 0.0, []
    
    # Cap at 1.0, but allow bonus for many matches
    score = min(len(found) / max(total * 0.5, 10), 1.0)  # Normalize to reasonable range
    return score, list(found)


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


def score_resume(resume_path: Path, config: Dict) -> Dict:
    """Score a single resume against the job description."""
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
    
    # Calculate component scores
    must_have_score, missing_must_haves = score_must_have_coverage(text, must_haves)
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
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
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
        results.append(result)
    
    # Create DataFrame and sort by score
    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    print(f"\nTop candidates:")
    print(df[["file", "score", "recommendation", "years_found", "must_haves_present"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
