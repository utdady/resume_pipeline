"""Helper script to process all resumes in the resumes/ directory."""
import subprocess
import sys
from pathlib import Path

def main():
    resumes_dir = Path("resumes")
    
    if not resumes_dir.exists():
        print(f"Error: '{resumes_dir}' directory not found!")
        print(f"Please create it and add your resume files (PDF/DOCX/TXT)")
        sys.exit(1)
    
    # Count resume files
    resume_files = []
    for ext in ["*.pdf", "*.docx", "*.txt", "*.PDF", "*.DOCX", "*.TXT"]:
        resume_files.extend(resumes_dir.glob(ext))
    
    if not resume_files:
        print(f"No resume files found in '{resumes_dir}' directory!")
        print("Supported formats: PDF, DOCX, TXT")
        sys.exit(1)
    
    print(f"Found {len(resume_files)} resume file(s)")
    print(f"Processing all resumes...\n")
    
    # Build command - use py launcher for Python 3.11 if available
    import shutil
    python_cmd = shutil.which("py")
    if python_cmd:
        cmd = [python_cmd, "-3.11", "baseline.py", str(resumes_dir), "--output", "scores.csv"]
    else:
        cmd = [sys.executable, "baseline.py", str(resumes_dir), "--output", "scores.csv"]
    
    # Run the scoring
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ Successfully processed {len(resume_files)} resumes")
        print(f"✓ Results saved to scores.csv")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error processing resumes: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

