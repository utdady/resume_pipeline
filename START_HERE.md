# 🚀 How to Run the Resume Matching Pipeline UI

## Quick Start (Easiest Method)

### Option 1: Double-Click the Batch File (Windows)
1. Navigate to: `C:\Users\addyb\resume_pipeline`
2. Double-click: `run_ui.bat`
3. Your browser will automatically open to `http://localhost:8501`

### Option 2: Command Line

**Open PowerShell or Command Prompt:**
1. Press `Windows Key + R`
2. Type `powershell` and press Enter
3. Navigate to the project folder:
   ```powershell
   cd C:\Users\addyb\resume_pipeline
   ```
4. Run the app:
   ```powershell
   streamlit run app.py
   ```
   OR
   ```powershell
   py -3.11 -m streamlit run app.py
   ```

## Step-by-Step Instructions

### Step 1: Open Terminal
- **Windows**: Press `Win + X` → Select "Windows PowerShell" or "Terminal"
- Or search for "PowerShell" in Start Menu

### Step 2: Navigate to Project Folder
```powershell
cd C:\Users\addyb\resume_pipeline
```

### Step 3: Launch the App
```powershell
streamlit run app.py
```

### Step 4: Browser Opens Automatically
- The app will open at: `http://localhost:8501`
- If it doesn't open automatically, manually go to that URL

## What You'll See

1. **Home Page** - Overview and instructions
2. **Score Resumes** - Main interface (use this!)
3. **View Results** - See saved results

## Using the UI

1. Click **"Score Resumes"** in the sidebar
2. **Enter Job Description:**
   - Paste text OR upload a file
   - (Optional) Enter Job Title and Requisition ID
3. **Upload Resumes:**
   - Click "Browse files"
   - Select one or more resume files (PDF, DOCX, TXT)
4. **Click "Analyze & Score Resumes"**
5. **View Results:**
   - See summary statistics
   - Browse the results table
   - Filter and sort candidates
   - Download CSV

## Troubleshooting

### "streamlit: command not found"
Install Streamlit:
```powershell
py -3.11 -m pip install streamlit
```

### "Module not found" errors
Install all dependencies:
```powershell
py -3.11 -m pip install -r requirements.txt
```

### Port 8501 already in use
The app will automatically use the next available port (8502, 8503, etc.)
Check the terminal output for the correct URL.

### Browser doesn't open
Manually navigate to: `http://localhost:8501`

## Stop the App

Press `Ctrl + C` in the terminal where the app is running.

