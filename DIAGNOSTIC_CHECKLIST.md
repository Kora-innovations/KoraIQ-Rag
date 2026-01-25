# Staging/Production Diagnostic Checklist

## Issue: 500 Error from Backend `/api/chat` Endpoint

The error shows the backend (`koraiq-4bkh.onrender.com`) is returning 500 when trying to call the RAG service.

## Step 1: Verify RAG Service is Deployed and Running

### Check RAG Service Health
```bash
# Replace YOUR_RAG_SERVICE_URL with your actual Render.com RAG service URL
curl https://YOUR_RAG_SERVICE_URL.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "kora-rag-service",
  "version": "2.0.0"
}
```

If this fails:
- ✅ Check Render.com RAG service logs for import errors
- ✅ Verify all environment variables are set in Render
- ✅ Ensure the latest code with CORS fixes is deployed

## Step 2: Verify Backend Environment Variables

In your **backend** Render.com service (`koraiq-4bkh`), check these environment variables:

### Required:
```bash
RAG_SERVER_URL=https://your-rag-service-name.onrender.com
```

**Important**: 
- Replace `your-rag-service-name` with your actual RAG service name on Render
- Must be the full HTTPS URL (not `http://localhost:8001`)
- No trailing slash

### Verify in Backend Logs
When the backend starts, you should see:
```
[RAG Integration] Initialized with RAG service
RAG Server URL: https://your-rag-service-name.onrender.com
Environment: staging
```

If you see `http://localhost:8001`, then `RAG_SERVER_URL` is NOT set in the backend's environment variables.

## Step 3: Test Backend → RAG Connection

### From Backend Server (SSH or Render Shell)
```bash
# Test if backend can reach RAG service
curl https://your-rag-service-name.onrender.com/health
```

### Check Backend Logs
Look for these error patterns in backend logs:

1. **Connection Refused / ECONNREFUSED**
   - RAG service URL is wrong or service is down
   - Fix: Set correct `RAG_SERVER_URL`

2. **Timeout / ECONNABORTED**
   - RAG service is sleeping (Render free tier)
   - First request may take 30-60 seconds
   - This is normal for free tier

3. **500 from RAG Service**
   - RAG service has an error (check RAG service logs)
   - Likely missing environment variables or import errors

4. **CORS Error**
   - Should be fixed with CORS middleware we added
   - If still happening, check CORS config in `advanced_rag.py`

## Step 4: Common Issues & Fixes

### Issue: Backend shows `RAG Server URL: http://localhost:8001`
**Fix**: Set `RAG_SERVER_URL` in backend's Render environment variables

### Issue: RAG service returns 500
**Possible causes**:
1. Missing environment variables (OPENAI_API_KEY, PINECONE_API_KEY, etc.)
2. Import errors (should be fixed, but verify latest code is deployed)
3. CORS issues (should be fixed with middleware)

**Fix**: 
- Check RAG service logs on Render
- Verify all environment variables are set
- Redeploy RAG service with latest code

### Issue: Connection timeout
**Fix**: 
- Normal for Render free tier (cold starts)
- Backend has 120-second timeout configured
- Wait for first request to wake up the service

## Step 5: Verify Latest Code is Deployed

### RAG Service (`ragproject/`)
Ensure these fixes are deployed:
- ✅ CORS middleware added
- ✅ Import errors fixed (`langchain_text_splitters`, etc.)
- ✅ `user_id` field added to `ChatRequest`

### Backend (`kora.IQ/`)
Ensure:
- ✅ `RAG_SERVER_URL` environment variable is set
- ✅ Backend is using `ragIntegrationService.sendChatMessage()`

## Quick Test Commands

### 1. Test RAG Service Directly
```bash
curl -X POST https://your-rag-service.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test123", "message": "hello"}'
```

### 2. Test Backend Health
```bash
curl https://koraiq-4bkh.onrender.com/api/rag/health
```

### 3. Test Full Flow (requires auth token)
```bash
curl -X POST https://koraiq-4bkh.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"message": "hello", "session_id": "test123"}'
```

## Next Steps

1. ✅ Check RAG service health endpoint
2. ✅ Verify `RAG_SERVER_URL` in backend environment variables
3. ✅ Check backend logs for connection errors
4. ✅ Check RAG service logs for runtime errors
5. ✅ Test direct connection from backend to RAG service
