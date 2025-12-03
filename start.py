#!/usr/bin/env python3
"""
Startup script for Kora RAG Service
This ensures the correct RAG agent is used in production
"""

import os
import sys
import uvicorn

def main():
    """Start the RAG service"""
    print(" Starting Kora RAG Service (Supabase)...")
    
    # Check if we're in production
    port = int(os.environ.get("PORT", 8002))
    host = "0.0.0.0"
    
    print(f"Starting service on {host}:{port}")
    
    # Import and run the Supabase RAG agent
    from rag_agent_supabase import app
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main()
