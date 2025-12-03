import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import requests
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="KoraIQ RAG Server", description="Health Insurance RAG System")

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# MongoDB connection
MONGODB_ATLAS_URI = os.getenv("MONGODB_ATLAS_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "korahq_dev")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "ragdb")

# Initialize MongoDB client
mongo_client = MongoClient(MONGODB_ATLAS_URI)
db = mongo_client[MONGODB_DB]
collection = db[MONGODB_COLLECTION]

# Pydantic models
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str

# Health insurance knowledge base (simplified)
HEALTH_INSURANCE_KNOWLEDGE = """
You are a helpful health insurance assistant for a Nigerian company. You help employees with:

1. Health Insurance Coverage:
   - Plan benefits and coverage limits
   - Dental, optical, and maternity coverage
   - Pre-existing conditions and exclusions

2. Provider Information:
   - In-network hospitals and clinics
   - Finding doctors on "The Island" (Lagos Island: Lekki, VI, Ikoyi, Ajah, Oniru)
   - Finding doctors on "The Mainland" (Yaba, Ikeja, Surulere, Maryland, Gbagada)

3. Claims and Benefits:
   - How to file claims
   - Required documents
   - Claim processing time

4. HMO Information:
   - HMO benefits and features
   - Network providers
   - Coverage tiers

Always provide helpful, accurate information and refer employees to HR or their HMO for specific plan details.
"""

@app.get("/")
async def root():
    return {"message": "KoraIQ RAG Server is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "koraid-rag"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Create a comprehensive prompt with health insurance context
        system_prompt = f"""
{HEALTH_INSURANCE_KNOWLEDGE}

User Question: {request.message}

Please provide a helpful response about health insurance, focusing on Nigerian context and the specific question asked.
"""

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        # Extract the response
        ai_response = response.choices[0].message.content

        # Store in MongoDB for history (optional)
        try:
            collection.insert_one({
                "session_id": request.session_id,
                "message": request.message,
                "response": ai_response,
                "timestamp": os.getenv("TIMESTAMP", "now")
            })
        except Exception as e:
            print(f"MongoDB storage error: {e}")

        return ChatResponse(response=ai_response)

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/docs")
async def get_docs():
    return {"message": "API documentation available at /docs"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
