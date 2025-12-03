#!/usr/bin/env python3
"""
Verification script to check Supabase setup
This script verifies:
1. Connection to Supabase
2. Table existence and structure
3. RPC function existence
4. Sample data in the table
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from supabase import create_client, Client
from typing import Dict, Any

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "hmo_documents")
SUPABASE_QUERY_NAME = os.getenv("SUPABASE_QUERY_NAME", "match_documents")
POSTGRES_URL = os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL")

def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def check_environment_variables():
    """Check if all required environment variables are set"""
    print_section("1. Environment Variables Check")
    
    checks = {
        "SUPABASE_URL": SUPABASE_URL,
        "SUPABASE_SERVICE_KEY": SUPABASE_SERVICE_KEY,
        "SUPABASE_TABLE": SUPABASE_TABLE,
        "SUPABASE_QUERY_NAME": SUPABASE_QUERY_NAME,
        "POSTGRES_URL": POSTGRES_URL
    }
    
    all_set = True
    for key, value in checks.items():
        if value:
            # Mask sensitive values
            if "KEY" in key or "URL" in key:
                display_value = value[:50] + "..." if len(value) > 50 else value
            else:
                display_value = value
            print(f"   ✓ {key}: {display_value}")
        else:
            print(f"   ✗ {key}: NOT SET")
            all_set = False
    
    return all_set

def check_supabase_connection():
    """Check if we can connect to Supabase"""
    print_section("2. Supabase Connection Test")
    
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print(f"   ✓ Supabase client created successfully")
        print(f"   ✓ URL: {SUPABASE_URL}")
        return supabase, True
    except Exception as e:
        print(f"   ✗ Failed to create Supabase client: {e}")
        return None, False

def check_table_exists(supabase: Client):
    """Check if the table exists and get its structure"""
    print_section("3. Table Existence Check")
    
    try:
        # Try to query the table (limit 0 to just check if it exists)
        response = supabase.table(SUPABASE_TABLE).select("*").limit(0).execute()
        print(f"   ✓ Table '{SUPABASE_TABLE}' exists")
        
        # Try to get a sample row to see structure
        try:
            sample = supabase.table(SUPABASE_TABLE).select("*").limit(1).execute()
            if hasattr(sample, 'data') and sample.data:
                print(f"   ✓ Table has data ({len(sample.data)} sample row(s))")
                print(f"   ✓ Sample row columns: {list(sample.data[0].keys())}")
                
                # Check for required columns
                required_columns = ['content', 'embedding']
                row = sample.data[0]
                for col in required_columns:
                    if col in row:
                        if col == 'embedding':
                            embedding_type = type(row[col]).__name__
                            embedding_len = len(row[col]) if hasattr(row[col], '__len__') else 'N/A'
                            print(f"   ✓ Column '{col}' exists (type: {embedding_type}, length: {embedding_len})")
                        else:
                            print(f"   ✓ Column '{col}' exists")
                    else:
                        print(f"   ⚠️  Column '{col}' NOT FOUND in table")
            else:
                print(f"   ⚠️  Table exists but is EMPTY - you need to ingest documents")
        except Exception as e:
            print(f"   ⚠️  Could not read sample data: {e}")
            print(f"   ⚠️  Table might be empty or have permission issues")
        
        return True
    except Exception as e:
        error_str = str(e)
        if "relation" in error_str.lower() or "does not exist" in error_str.lower():
            print(f"   ✗ Table '{SUPABASE_TABLE}' does NOT exist")
            print(f"   → You need to create this table in Supabase")
        else:
            print(f"   ✗ Error checking table: {e}")
        return False

def check_rpc_function(supabase: Client):
    """Check if the RPC function exists"""
    print_section("4. RPC Function Check")
    
    try:
        # Try calling the function with a dummy embedding
        # Use a small test embedding (1536 dimensions for text-embedding-3-small)
        test_embedding = [0.0] * 1536
        
        try:
            response = supabase.rpc(
                SUPABASE_QUERY_NAME,
                {
                    "query_embedding": test_embedding,
                    "match_threshold": 0.7,
                    "match_count": 1
                }
            ).execute()
            print(f"   ✓ RPC function '{SUPABASE_QUERY_NAME}' exists and is callable")
            return True
        except Exception as rpc_error:
            error_str = str(rpc_error)
            if "function" in error_str.lower() and "does not exist" in error_str.lower():
                print(f"   ✗ RPC function '{SUPABASE_QUERY_NAME}' does NOT exist")
                print(f"   → You need to create this function in Supabase")
            elif "Name or service not known" in error_str or "Errno -2" in error_str:
                print(f"   ⚠️  DNS/Connection error when calling RPC")
                print(f"   → This might be a network issue or incorrect SUPABASE_URL")
            else:
                print(f"   ⚠️  RPC function call failed: {rpc_error}")
                print(f"   → Function might exist but have different parameters")
            return False
    except Exception as e:
        print(f"   ✗ Error checking RPC function: {e}")
        return False

def check_table_data_count(supabase: Client):
    """Check how many documents are in the table"""
    print_section("5. Data Count Check")
    
    try:
        # Count documents (Supabase doesn't have a direct count, so we'll estimate)
        response = supabase.table(SUPABASE_TABLE).select("id", count="exact").limit(1).execute()
        
        if hasattr(response, 'count') and response.count is not None:
            print(f"   ✓ Table contains {response.count} document(s)")
            if response.count == 0:
                print(f"   ⚠️  WARNING: Table is empty!")
                print(f"   → You need to run 'python ingest_to_supabase.py' to upload documents")
            return response.count
        else:
            # Fallback: try to get all and count
            try:
                all_data = supabase.table(SUPABASE_TABLE).select("*").execute()
                count = len(all_data.data) if hasattr(all_data, 'data') else 0
                print(f"   ✓ Table contains approximately {count} document(s)")
                if count == 0:
                    print(f"   ⚠️  WARNING: Table is empty!")
                    print(f"   → You need to run 'python ingest_to_supabase.py' to upload documents")
                return count
            except:
                print(f"   ⚠️  Could not count documents")
                return None
    except Exception as e:
        print(f"   ✗ Error counting documents: {e}")
        return None

def test_vector_search(supabase: Client):
    """Test a simple vector search"""
    print_section("6. Vector Search Test")
    
    try:
        # Generate a test embedding (using OpenAI if available)
        try:
            from langchain_openai import OpenAIEmbeddings
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)
                test_query = "health insurance"
                query_embedding = embeddings.embed_query(test_query)
                print(f"   ✓ Generated test embedding for query: '{test_query}'")
            else:
                print(f"   ⚠️  OPENAI_API_KEY not set, skipping embedding generation")
                return False
        except Exception as e:
            print(f"   ⚠️  Could not generate embedding: {e}")
            return False
        
        # Try the RPC call
        try:
            response = supabase.rpc(
                SUPABASE_QUERY_NAME,
                {
                    "query_embedding": query_embedding,
                    "match_threshold": 0.5,  # Lower threshold for testing
                    "match_count": 3
                }
            ).execute()
            
            if hasattr(response, 'data'):
                results = response.data
            elif isinstance(response, list):
                results = response
            else:
                results = []
            
            print(f"   ✓ Vector search successful!")
            print(f"   ✓ Found {len(results)} result(s)")
            
            if results:
                print(f"   ✓ Sample result columns: {list(results[0].keys())}")
            else:
                print(f"   ⚠️  No results found (might be due to empty table or high threshold)")
            
            return True
        except Exception as e:
            error_str = str(e)
            print(f"   ✗ Vector search failed: {e}")
            if "Name or service not known" in error_str:
                print(f"   → DNS/Connection issue - check SUPABASE_URL")
            return False
    except Exception as e:
        print(f"   ✗ Error testing vector search: {e}")
        return False

def print_setup_instructions():
    """Print instructions for setting up Supabase"""
    print_section("Setup Instructions")
    
    print("""
If any checks failed, here's what you need to do:

1. CREATE TABLE (if table doesn't exist):
   Run this SQL in your Supabase SQL Editor:
   
   CREATE TABLE IF NOT EXISTS hmo_documents (
     id BIGSERIAL PRIMARY KEY,
     content TEXT NOT NULL,
     metadata JSONB DEFAULT '{}',
     embedding vector(1536)  -- Adjust dimension if using different model
   );

2. CREATE VECTOR INDEX:
   CREATE INDEX ON hmo_documents 
   USING ivfflat (embedding vector_cosine_ops)
   WITH (lists = 100);

3. CREATE RPC FUNCTION (match_documents):
   Run this SQL in your Supabase SQL Editor:
   
   CREATE OR REPLACE FUNCTION match_documents(
     query_embedding vector(1536),
     match_threshold float DEFAULT 0.7,
     match_count int DEFAULT 10
   )
   RETURNS TABLE (
     id bigint,
     content text,
     metadata jsonb,
     similarity float
   )
   LANGUAGE plpgsql
   AS $$
   BEGIN
     RETURN QUERY
     SELECT
       hmo_documents.id,
       hmo_documents.content,
       hmo_documents.metadata,
       1 - (hmo_documents.embedding <=> query_embedding) as similarity
     FROM hmo_documents
     WHERE 1 - (hmo_documents.embedding <=> query_embedding) > match_threshold
     ORDER BY hmo_documents.embedding <=> query_embedding
     LIMIT match_count;
   END;
   $$;

4. INGEST DOCUMENTS:
   Run: python ingest_to_supabase.py
    """)

def main():
    """Main verification function"""
    print("=" * 60)
    print("  Supabase Setup Verification Script")
    print("=" * 60)
    
    # Check environment variables
    if not check_environment_variables():
        print("\n❌ Missing required environment variables!")
        print("   Please set all required variables in your .env file")
        return
    
    # Check connection
    supabase, connected = check_supabase_connection()
    if not connected:
        print("\n❌ Cannot connect to Supabase!")
        print("   Please check your SUPABASE_URL and SUPABASE_SERVICE_KEY")
        return
    
    # Check table
    table_exists = check_table_exists(supabase)
    
    # Check RPC function
    rpc_exists = check_rpc_function(supabase)
    
    # Check data count
    data_count = check_table_data_count(supabase)
    
    # Test vector search
    if table_exists and rpc_exists:
        test_vector_search(supabase)
    
    # Print summary
    print_section("Summary")
    
    all_good = True
    if not table_exists:
        print("   ✗ Table does not exist")
        all_good = False
    if not rpc_exists:
        print("   ✗ RPC function does not exist")
        all_good = False
    if data_count == 0:
        print("   ⚠️  Table is empty - need to ingest documents")
        all_good = False
    
    if all_good and data_count and data_count > 0:
        print("   ✅ All checks passed! Your Supabase setup looks good.")
        print("   ✅ You can now use the RAG service.")
    else:
        print("   ⚠️  Some issues found. See instructions below.")
        print_setup_instructions()
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()

