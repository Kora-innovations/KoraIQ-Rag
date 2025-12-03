# Local Development Setup

## Python Version Requirement

The project requires **Python 3.10 or higher** because `langchain==1.0.5` requires Python 3.10+.

## Current Issue

Your local environment has Python 3.9.6, which cannot install `langchain==1.0.5`.

## Solutions

### Option 1: Install Python 3.10+ (Recommended)

**Using Homebrew (macOS):**
```bash
brew install python@3.11
```

Then create a new virtual environment:
```bash
cd ragproject
python3.11 -m venv ragvenv
source ragvenv/bin/activate
pip install -r requirements.txt
```

**Using pyenv (Alternative):**
```bash
brew install pyenv
pyenv install 3.11.0
cd ragproject
pyenv local 3.11.0
python -m venv ragvenv
source ragvenv/bin/activate
pip install -r requirements.txt
```

### Option 2: Update Render to Match

If you want to keep Python 3.9 locally, you'll need to:
1. Update `render.yaml` to use Python 3.10+
2. Update `requirements.txt` to use langchain 1.0.5 compatible versions

**Note:** Render might already be using Python 3.10+ despite the config saying 3.9.0. Check Render's build logs to confirm.

## Quick Test

After setting up Python 3.10+, test the service:
```bash
source ragvenv/bin/activate
python3 start.py
```

Or use the CLI:
```bash
python3 rag_agent_deux_cli.py serve --port 8001 --host 0.0.0.0
```

