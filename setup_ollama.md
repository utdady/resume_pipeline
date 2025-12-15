# Setting Up Ollama for AI-Powered Job Analysis

## What is Ollama?

Ollama runs large language models (like Llama 3) locally on your computer. It's:
- ✅ Free and open-source
- ✅ Runs completely offline
- ✅ No API keys needed
- ✅ Privacy-focused (data never leaves your machine)

## Installation

### Windows
1. Download: https://ollama.com/download/windows
2. Run the installer
3. Ollama will auto-start in the background

### Mac
```bash
brew install ollama
```

Or download from: https://ollama.com/download/mac

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Setup (5 minutes)

### 1. Verify Ollama is Running
```bash
ollama --version
```

### 2. Pull the Model
```bash
ollama pull llama3.2
```
This downloads ~2GB. Takes 2-5 minutes depending on internet speed.

### 3. Test It
```bash
ollama run llama3.2 "Extract skills from: Python developer needed, 5 years experience"
```

You should see the model respond with extracted skills.

### 4. Use in Resume Matcher

**CLI:**
```bash
# With AI (default)
python jd_analyzer.py job_description.txt

# Force rule-based
python jd_analyzer.py job_description.txt --no-llm
```

**Web UI:**
- Check "🧠 AI" checkbox when analyzing job descriptions
- System automatically falls back to rules if Ollama unavailable

## Troubleshooting

### "Cannot connect to Ollama"
**Solution:** Start Ollama server:
```bash
ollama serve
```

### "Model not found"
**Solution:** Pull the model:
```bash
ollama pull llama3.2
```

### Slow Analysis (>30 seconds)
**Solutions:**
1. Use a smaller model: `ollama pull llama3.2:1b`
2. Check CPU usage (close other apps)
3. Fallback to rule-based: uncheck "AI" in UI

### High RAM Usage
**Normal:** Ollama uses 4-8GB RAM while running
**Solution:** Close Ollama when not needed:
```bash
# Mac/Linux
killall ollama

# Windows
Task Manager → End "ollama" process
```

## System Requirements

- **Minimum:** 8GB RAM, 4GB disk space
- **Recommended:** 16GB RAM for best performance
- **Works on:** Windows 10+, macOS 11+, Linux (most distros)

## Alternative Models

If llama3.2 is too slow:
```bash
# Faster, less accurate
ollama pull phi3

# Balanced
ollama pull mistral

# Most accurate, slower
ollama pull llama3.1:70b  # Requires 32GB+ RAM
```

Change model in `jd_analyzer_llm.py` line 9:
```python
MODEL = "phi3"  # or "mistral"
```
# Setup Ollama for LLM Integration

## Prerequisites
- Ensure you have Docker or a compatible environment if running Ollama in a container.

## Installation Steps
1. Download and install Ollama from [ollama.ai](https://ollama.ai).
2. Open a terminal and run: `ollama pull llama3.2` to download the llama3.2 model.
3. Start the Ollama server: `ollama serve` (it runs on http://localhost:11434 by default).
4. Verify it's running by checking the API endpoint in a browser or via curl: `curl http://localhost:11434/api/tags`.

## Troubleshooting
- If the API is not responding, ensure Ollama is running and the model is pulled.
- For errors, check Ollama logs or restart the service.
