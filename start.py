#!/usr/bin/env python3
"""
Start script for Render.com deployment.
Runs the FastAPI RAG service on port 8002 (or PORT from environment).
"""
import os
import uvicorn
from advanced_rag import app

if __name__ == "__main__":
    # Render provides PORT environment variable
    port = int(os.getenv("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)
