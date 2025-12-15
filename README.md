# Resume Matching Pipeline

## Overview

This pipeline scores resumes against **any** job description using rule-based matching. It can:

1. **Auto-analyze job descriptions** - Extract requirements from any JD and generate scoring config
2. **Score resumes** - Evaluate candidates against the job requirements

The system extracts text from PDF, DOCX, and TXT files, then evaluates candidates based on:

- **Must-have coverage (45%)**: Critical requirements (CAPE, ASPEN, short-circuit analysis, etc.)
- **Skill overlap (25%)**: Nice-to-have skills (solar, renewables, SQL, etc.)
- **Title similarity (15%)**: Role alignment keywords
- **Years experience (15%)**: Experience level matching (4-6 years preferred)

## Scoring Rules

- **Advance**: Score ≥ 0.72 AND all 7 must-haves present
- **Review**: Score ≥ 0.50
- **Reject**: Score < 0.50

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test with sample resume
python baseline.py sample_resume.txt

# 3. Check results
cat scores.csv
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # Or on Windows:
   py -3.11 -m pip install -r requirements.txt
   ```

2. **Install NLP model (for enhanced JD parsing and resume scoring):**
   ```bash
   # Option 1: Use the setup script
   python setup_nlp.py
   
   # Option 2: Manual installation
   python -m spacy download en_core_web_sm
   ```
   
   The NLP pipeline powers both job description analysis and resume scoring by:
   - Using dependency parsing to understand relationships (e.g., "3 years of Python experience")
   - Identifying skills and technologies through Named Entity Recognition
   - Matching compound skill phrases and synonyms in resumes (e.g., "financial modeling" vs "model financials")
   - Better distinguishing between must-haves and nice-to-haves using linguistic context
   - Extracting compound phrases and technical terms more accurately
   
   **Note:** The parser and scorer will work without spaCy but will fall back to basic regex parsing.

3. **Two ways to use the pipeline:**

   **Option A: Use existing jd_meta.yaml** (for predefined job descriptions)
   - The default `jd_meta.yaml` is configured for Electrical Engineer 2 position
   - You can manually edit it for other jobs
   
   **Option B: Auto-generate jd_meta.yaml** (recommended for new job descriptions)
   - Use `jd_analyzer.py` to parse any job description
   - Automatically extracts requirements, skills, and experience needs

## Usage

### Step 1: Analyze Job Description (Optional - for new jobs)

If you have a new job description, analyze it first:

```bash
# Analyze a job description file
py -3.11 jd_analyzer.py job_description.txt --output jd_meta.yaml

# Or analyze from stdin
cat job_description.txt | py -3.11 jd_analyzer.py - --output jd_meta.yaml

# Specify job title and requisition ID
py -3.11 jd_analyzer.py job_description.txt --title "Software Engineer" --req-id "REQ-12345"
```

The analyzer will:
- Detect domain (software/electrical/mechanical/data/business) using NLP context analysis
- Extract must-have requirements using dependency parsing and entity recognition
- Extract nice-to-have skills with better context understanding
- Parse experience requirements (e.g., "4-6 years") using linguistic analysis
- Generate `jd_meta.yaml` configuration

### AI-Powered JD Analysis with Ollama (Optional)

For higher-quality requirement extraction you can use a local LLM via **Ollama**.

- See `setup_ollama.md` for installation and model setup.
- The system automatically falls back to the rule-based analyzer if Ollama is not available.

**CLI:**
```bash
# Default: rule-based analyzer with spaCy NLP (no LLM)
py -3.11 jd_analyzer.py job_description.txt --output jd_meta.yaml

# Opt in to Ollama LLM (will still fallback safely if Ollama isn't running)
py -3.11 jd_analyzer.py job_description.txt --llm --output jd_meta.yaml

# Explicitly force rule-based only (equivalent to default)
py -3.11 jd_analyzer.py job_description.txt --no-llm --output jd_meta.yaml
```

**Streamlit UI (`app.py`):**
- Go to **Score Resumes → Step 1**.
- Upload or paste a job description.
- Use the **🧠 AI** checkbox to enable/disable Ollama JD analysis.
- If Ollama isn't running, the UI will show a warning and automatically use the rule-based analyzer.

**NLP Features:**
- Understands relationships between skills and experience levels
- Extracts compound technical terms (e.g., "financial modeling", "machine learning")
- Better identifies requirements vs preferences through linguistic cues
- Processes bullet points and structured text more intelligently

### Step 2: Score Resumes

**Score a single resume:**
```bash
py -3.11 baseline.py resume.pdf
```

**Score multiple resumes:**
```bash
py -3.11 baseline.py resume1.pdf resume2.docx resume3.txt
```

**Score all resumes in a directory:**
```bash
py -3.11 baseline.py resumes/
```

**Generate per-candidate requirement matrices (CSV + Excel):**
```bash
# Default reports folder: candidate_reports/
py -3.11 baseline.py resumes/ --config jd_meta.yaml --output results.csv --reports-dir candidate_reports

# Skip report generation if you only want the summary CSV
py -3.11 baseline.py resumes/ --config jd_meta.yaml --no-reports
```

**Use custom config file:**
```bash
py -3.11 baseline.py resumes/ --config jd_meta_auto.yaml
```

**Specify output file:**
```bash
py -3.11 baseline.py resumes/ --output results.csv
```

### NLP-Enhanced Resume Reading

- spaCy dependency parsing + noun-phrase matching catches compound skills (REST APIs, financial modeling, etc.)
- Keyword detection works even when resumes paraphrase requirements (e.g., "experience building REST services")
- Every candidate now receives a binary grade for each must-have and nice-to-have in both CSV and Excel form for quick auditing

### Helper Tools

**Process all resumes in resumes/ folder:**
```bash
py -3.11 process_all.py
```

**Extract text from a resume:**
```bash
py -3.11 parser.py resume.pdf
py -3.11 parser.py resume.pdf --out resume_text.txt
```

## Output

The pipeline generates `scores.csv` with the following columns:

- `file`: Resume filename
- `score`: Final weighted score (0-1)
- `recommendation`: Advance / Review / Reject
- `explanation`: Human-readable breakdown
- `years_found`: Extracted years of experience
- `must_haves_present`: Number of must-haves found
- `must_haves_total`: Total must-haves required
- `must_have_coverage`: Must-have score component
- `skill_overlap`: Skill overlap score component
- `title_similarity`: Title similarity score component
- `years_score`: Years experience score component

Results are sorted by score (highest first).

## Example Commands

```bash
# Score a single resume
python baseline.py candidate_resume.pdf

# Score all PDFs in current directory
python baseline.py *.pdf

# Score resumes from a folder
python baseline.py ./resumes/

# Process all resumes in resumes/ directory (recommended for batch processing)
python process_all.py

# Extract text from a resume for inspection
python parser.py candidate_resume.pdf --out extracted.txt
```

## Batch Processing 50 Resumes

1. **Copy your 50 resume files to the `resumes/` directory:**
   ```bash
   # Copy resumes from another location
   copy C:\path\to\resumes\*.pdf resumes\
   copy C:\path\to\resumes\*.docx resumes\
   ```

2. **Process all resumes:**
   ```bash
   python process_all.py
   ```

   Or directly:
   ```bash
   python baseline.py resumes/
   ```

3. **View results:**
   ```bash
   # Open scores.csv in Excel or view in terminal
   python -c "import pandas as pd; df = pd.read_csv('scores.csv'); print(df[['file', 'score', 'recommendation']].head(20))"
   ```

## Must-Have Requirements

The must-have requirements are automatically extracted from the job description. For the default Electrical Engineer 2 position, these include:

1. Bachelor's degree in Electrical Engineering
2. Electrical engineer (role/title)
3. Power systems experience
4. Short circuit analysis
5. Protection engineering
6. CAPE software experience
7. ASPEN software experience

**Note:** For other job descriptions, must-haves will be different and automatically detected by `jd_analyzer.py`.

## File Structure

```
resume_pipeline/
├── requirements.txt         # Python dependencies
├── parser.py               # Text extraction utility
├── jd_analyzer.py          # Auto-extract requirements from job descriptions
├── jd_meta.yaml            # Job description metadata (default: Electrical Engineer 2)
├── baseline.py             # Main scoring script
├── process_all.py          # Batch processing helper
├── README.md               # This file
├── resumes/                # Input resumes folder
├── sample_resume.txt       # Sample resume for testing
├── sample_jd.txt           # Sample job description
└── scores.csv              # Output (generated)
```

## Web UI (Streamlit)

Launch the interactive web interface:

```bash
# Install streamlit if not already installed
py -3.11 -m pip install streamlit

# Launch the app
streamlit run app.py
```

The web UI provides:
- **Job Description Input** - Paste text or upload file
- **Resume Upload** - Upload multiple resumes at once
- **Auto Analysis** - Automatically extracts requirements
- **Requirement Editing** - Add/remove must-haves, nice-to-haves, and adjust scoring weights
- **Interactive Results** - Filter, sort, and view detailed breakdowns
- **CSV Export** - Download results for further analysis

Open your browser to `http://localhost:8501` when the app starts.

## Notes

- The parser handles PDF, DOCX, and TXT files
- Text extraction may vary by file format quality
- Years of experience are extracted using pattern matching
- Keyword matching is case-insensitive with word boundaries
- Missing must-haves are listed in the explanation field

