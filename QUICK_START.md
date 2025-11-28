# Quick Start Guide

## Complete Workflow: Analyze Any Job Description & Score Resumes

### Step 1: Analyze Job Description

```bash
# Analyze a job description file
py -3.11 jd_analyzer.py job_description.txt --output jd_meta.yaml

# With custom title and requisition ID
py -3.11 jd_analyzer.py job_description.txt --title "Data Scientist" --req-id "DS-2024-001" --output jd_meta.yaml
```

**What it does:**
- Detects domain (software/electrical/mechanical/data/business)
- Extracts must-have requirements
- Extracts nice-to-have skills
- Parses experience requirements (e.g., "4-6 years")
- Generates `jd_meta.yaml` configuration file

### Step 2: Prepare Resumes

```bash
# Copy resumes to the resumes/ folder
copy "C:\path\to\resumes\*.pdf" resumes\
copy "C:\path\to\resumes\*.docx" resumes\
```

### Step 3: Score Resumes

```bash
# Score all resumes using the generated config
py -3.11 baseline.py resumes/ --config jd_meta.yaml --output scores.csv

# Or use the helper script
py -3.11 process_all.py
```

### Step 4: View Results

Open `scores.csv` in Excel or view in terminal:
```bash
py -3.11 -c "import pandas as pd; df = pd.read_csv('scores.csv'); print(df[['file', 'score', 'recommendation']].to_string())"
```

## Example: Complete Workflow

```bash
# 1. Analyze job description
py -3.11 jd_analyzer.py new_job_post.txt --output jd_meta_new.yaml

# 2. Score resumes
py -3.11 baseline.py resumes/ --config jd_meta_new.yaml --output results.csv

# 3. View top candidates
py -3.11 -c "import pandas as pd; df = pd.read_csv('results.csv'); print(df.head(10))"
```

## Tips

- **Job Description Format**: Plain text (.txt) works best. The analyzer looks for sections with "Required", "Must Have", "Preferred", etc.
- **Must-Haves**: The analyzer extracts keywords from requirement sections. You can manually edit `jd_meta.yaml` to refine them.
- **Scoring**: Candidates need score ≥ 0.72 AND all must-haves for "Advance" recommendation.

