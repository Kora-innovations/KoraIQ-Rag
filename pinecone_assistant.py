from pinecone import Pinecone
import os
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

assistant = pc.assistant.create_assistant(
    assistant_name="example-assistant", 
    instructions="Answer in polite, short sentences. Use American English spelling and vocabulary.", 
    timeout=30 # Wait 30 seconds for assistant operation to complete.
)