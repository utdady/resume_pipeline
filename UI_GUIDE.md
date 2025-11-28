# Web UI User Guide

## Quick Start

### Launch the Web Interface

**Option 1: Using the batch file (Windows)**
```bash
run_ui.bat
```

**Option 2: Using command line**
```bash
streamlit run app.py
```

**Option 3: Using Python directly**
```bash
py -3.11 -m streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## How to Use

### Step 1: Enter Job Description

1. Go to the **"Score Resumes"** page
2. Choose input method:
   - **Paste Text**: Copy and paste the job description directly
   - **Upload File**: Upload a TXT, DOCX, or PDF file
3. (Optional) Enter Job Title and Requisition ID
4. Click **"Analyze & Score Resumes"**

### Step 2: Upload Resumes

1. Click **"Browse files"** in the resume upload section
2. Select one or more resume files (PDF, DOCX, or TXT)
3. You can upload multiple files at once

### Step 3: View Results

After processing, you'll see:
- **Summary Statistics** - Total candidates, recommendations breakdown
- **Results Table** - Sortable, filterable table with all scores
- **Detailed View** - Click on any candidate for detailed breakdown
- **Download CSV** - Export results for further analysis

## Features

### Interactive Filters
- **Minimum Score** - Filter by score threshold
- **Recommendation** - Filter by Advance/Review/Reject
- **Must-Haves** - Filter by minimum must-haves present

### Results Display
- **Ranked List** - Candidates sorted by score (highest first)
- **Score Breakdown** - See component scores (must-haves, skills, title, experience)
- **Explanations** - Detailed text explaining why each score was given
- **Export** - Download filtered results as CSV

### Auto Analysis
The system automatically:
- Detects job domain (software/electrical/financial/etc.)
- Extracts must-have requirements
- Extracts nice-to-have skills
- Parses experience requirements
- Generates scoring configuration

## Tips

1. **Job Description Quality**: More detailed JDs produce better results
2. **Resume Format**: PDF files generally extract better than DOCX
3. **Batch Processing**: Upload all resumes at once for faster processing
4. **Review Extracted Requirements**: Check the "Extracted Requirements" section to verify the system understood the JD correctly

## Troubleshooting

**App won't start:**
- Make sure Streamlit is installed: `py -3.11 -m pip install streamlit`
- Check Python version: `py -3.11 --version`

**Resume parsing errors:**
- Some PDFs may have extraction issues (this is normal)
- Try converting problematic PDFs to TXT format

**Slow processing:**
- Large PDF files take longer to process
- Processing 50+ resumes may take a few minutes

