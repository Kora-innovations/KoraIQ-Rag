# RAG Service Requirements Checklist

## ✅ What You Need for Your RAG to Work

### 1. **Supabase Database Setup** (CRITICAL)

Your Supabase database must have:

#### a) Table: `hmo_documents`
- **Columns required:**
  - `id` (bigint, primary key)
  - `content` (text) - the document text
  - `metadata` (jsonb) - metadata about the document
  - `embedding` (vector) - the embedding vector (must have pgvector extension)

#### b) RPC Function: `match_documents`
- This function performs vector similarity search
- Must accept parameters:
  - `query_embedding` (vector)
  - `match_threshold` (float, default 0.5)
  - `match_count` (int, default 4)
- Must return rows with `content` and `metadata` fields

#### c) Data in the Table
- **Documents must be ingested** with embeddings
- Empty table = no results from RAG
- Use the ingestion script to upload your PDFs/Excel files

**How to verify:**
```bash
# Run the verification script (if you have it)
python verify_supabase_setup.py
```

Or check in Supabase Dashboard:
- Go to Table Editor → `hmo_documents` → Check row count
- Go to SQL Editor → Test `match_documents` function

---

### 2. **Environment Variables on Render** (CRITICAL)

All these must be set in your Render service environment:

#### Required (Service won't start without these):
- ✅ `SUPABASE_URL` - Format: `https://[project-ref].supabase.co`
- ✅ `SUPABASE_SERVICE_KEY` - Service role key (not anon key)
- ✅ `POSTGRES_URL` - Format: `postgresql+psycopg://user:pass@host:port/dbname`
- ✅ `OPENAI_API_KEY` - Your OpenAI API key

#### Optional but Recommended:
- `SUPABASE_TABLE` - Defaults to `hmo_documents` if not set
- `SUPABASE_QUERY_NAME` - Defaults to `match_documents` if not set
- `OPENAI_MODEL` - Defaults to `gpt-5` if not set
- `LANGSMITH_API_KEY` - For tracing (optional)
- `LANGSMITH_PROJECT` - Defaults to `health-insurance-qa`
- `TAVILY_API_KEY` - For web search (optional)

**How to verify:**
- Check Render logs on startup - you should see:
  ```
  [Config] Supabase client initialized
  [Config] ✓ Connection test successful - table is accessible
  ```

---

### 3. **Code Dependencies** (Already Set)

Your `requirements.txt` has all needed packages:
- ✅ `langchain==1.0.5` (requires Python 3.10+)
- ✅ `supabase>=2.3.0`
- ✅ `langchain-community>=0.4.0`
- ✅ All other dependencies

**Note:** Render must use Python 3.10+ to install `langchain==1.0.5`

---

### 4. **Common Issues & Solutions**

#### Issue: "No documents retrieved"
**Possible causes:**
1. ❌ Table is empty → **Solution:** Run ingestion script to upload documents
2. ❌ `match_threshold` too high → **Solution:** Already lowered to 0.5 in code
3. ❌ RPC function not working → **Solution:** Check SQL function in Supabase

#### Issue: "PostgreSQL connection failed"
**Possible causes:**
1. ❌ `POSTGRES_URL` format wrong → **Solution:** Use `postgresql+psycopg://` format
2. ❌ Wrong credentials → **Solution:** Verify in Supabase Dashboard

#### Issue: "DNS error" or "Name or service not known"
**Possible causes:**
1. ❌ `SUPABASE_URL` incorrect → **Solution:** Must be `https://[project-ref].supabase.co`
2. ❌ Network issue from Render → **Solution:** Check Render's network access

#### Issue: "SyncRPCFilterRequestBuilder" error
**Status:** ✅ **FIXED** - Code now uses compatibility wrapper with fallback

---

### 5. **Testing Your RAG**

#### Step 1: Check Connection
Look for these in Render logs:
```
[Config] ✓ Connection test successful - table is accessible
```

#### Step 2: Test a Query
Send a query like: "what is the physiotherapy limit for a bronze plan?"

Check logs for:
```
[RAG Tool] Called with query: '...'
[VectorStore] similarity_search called with query: '...'
[VectorStore] Calling RPC function 'match_documents'...
[VectorStore] ✓ RPC call successful
[VectorStore] RPC returned X results
[RAG Tool] Retrieved X source documents
```

#### Step 3: Verify Results
- If you see `RPC returned 0 results` → Table might be empty or threshold too high
- If you see `RPC call successful` with results → RAG is working!

---

### 6. **Quick Diagnostic Commands**

#### Check if table has data:
```sql
-- Run in Supabase SQL Editor
SELECT COUNT(*) FROM hmo_documents;
```

#### Test the RPC function:
```sql
-- Run in Supabase SQL Editor (this is a simplified test)
-- You'll need to generate an embedding first
SELECT * FROM match_documents(
  '[0.1, 0.2, ...]'::vector,  -- Replace with actual embedding
  0.5,  -- threshold
  4     -- match_count
);
```

---

## 🎯 Current Status Checklist

Based on your setup, verify:

- [ ] Supabase table `hmo_documents` exists and has columns: `id`, `content`, `metadata`, `embedding`
- [ ] RPC function `match_documents` exists and works
- [ ] Table has documents with embeddings (not empty)
- [ ] `SUPABASE_URL` in Render is correct format: `https://[project-ref].supabase.co`
- [ ] `SUPABASE_SERVICE_KEY` in Render is the service role key (not anon key)
- [ ] `POSTGRES_URL` in Render is correct format
- [ ] Render is using Python 3.10+ (check build logs)
- [ ] All dependencies installed successfully (check build logs)
- [ ] Connection test passes on startup (check Render logs)

---

## 🚨 Most Common Issue

**"RAG not returning answers"** is usually because:
1. **Table is empty** - Documents haven't been ingested
2. **RPC function missing** - `match_documents` function not created
3. **Wrong Supabase URL** - DNS/connection errors in logs

**Fix:** Check Render logs for the detailed error messages we added!

