# Render.com Deployment Guide

## Required Environment Variables

Set these in your Render.com service dashboard under "Environment":

### API Keys (Required)
```bash
OPENAI_API_KEY=sk-...                    # OpenAI API key for LLM and embeddings
PINECONE_API_KEY=pcsk_...                # Pinecone API key
PINECONE_INDEX_NAME=people-team          # Pinecone index name
TAVILY_API_KEY=tvly-...                  # Tavily API key for web search (optional but recommended)
```

### Optional Configuration
```bash
LANGSMITH_API_KEY=...                    # LangSmith tracing (optional)
LANGSMITH_PROJECT=kora-rag-service       # LangSmith project name (optional)
MAX_HISTORY_TOKENS=1500                  # Max tokens for chat history (default: 1500)
TOKEN_MODEL_FOR_COUNT=gpt-4o-mini        # Model for token counting (default: gpt-4o-mini)
```

### Render-Specific
```bash
PORT=8002                                # Port (Render sets this automatically, but you can override)
```

## Backend Configuration

Your **backend** (Node.js) needs to know the Render.com URL. Set this in your backend's environment variables:

### Staging Backend
```bash
RAG_SERVER_URL=https://your-rag-service-name.onrender.com
```

### Production Backend
```bash
RAG_SERVER_URL=https://your-rag-service-name.onrender.com
```

**Important**: Replace `your-rag-service-name` with your actual Render.com service name.

## Build & Start Commands

In Render.com service settings:

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python start.py`

## Health Check

After deployment, test the health endpoint:
```bash
curl https://your-rag-service-name.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "kora-rag-service",
  "version": "2.0.0"
}
```

## Troubleshooting

### Service Not Starting
1. Check Render logs for import errors
2. Verify all environment variables are set
3. Ensure `requirements.txt` includes all dependencies

### Backend Can't Connect
1. Verify `RAG_SERVER_URL` in backend environment variables
2. Check CORS is enabled (already added to `advanced_rag.py`)
3. Test health endpoint from backend server

### Cold Start Issues
- Render free tier services "sleep" after inactivity
- First request may take 30-60 seconds to wake up
- Backend has 120-second timeout configured for this

## Service URLs

Update these in your backend's environment variables based on your Render service names:

- **Staging RAG Service**: `https://your-staging-rag-service.onrender.com`
- **Production RAG Service**: `https://your-production-rag-service.onrender.com`
