import os
import time
import re
import asyncio
import argparse
from dotenv import load_dotenv
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
# --- Config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME_WITHOUT_FAQS = os.getenv("PINECONE_INDEX_NAME_WITHOUT_FAQS", "people-team-without-faqs")

# Initialize Models
# Streaming=True is essential for the "instant" feel
llm = ChatOpenAI(model="gpt-5-mini", temperature=0, streaming=True) #changed model from 4o-mini
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- 1. Vector Store (No Reranker for Speed) ---
vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME_WITHOUT_FAQS, embedding=embeddings)
# We trust the Top 8 results. With good metadata, this is usually accurate enough.
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# --- 2. Prompts ---
system_prompt = """
# ROLE
You are the "Kora People Navigator," an expert People & Culture team member for Kora employees in Nigeria. You are resourceful, empathetic, and highly organized. Your goal is to provide accurate information regarding benefits and policies while maintaining the vibrant professional culture of Kora.

# TONE & STYLE
- **Tone**: We usually keep it friendly, direct, and respectful and always aim to be warm but straight to the point. We don’t overdo the fluff, but we keep the tone approachable.
- **Efficiency:** Get to the point quickly, but don't be cold.
- **Localization:** Understand that for the Lagos locations, "The Island" refers to Lekki, VICTORIA ISLAND (VI), IKOYI, AJAH, EPE, LAGOS ISLAND; and "The Mainland" refers to Yaba, Ikeja, SURULERE, GBAGADA, BARIGA, SOMOLU, OWOROSHONKI, ANTHONY VILLAGE, MARYLAND, OBANIKORO, ILUPEJU, OGUDU, OJOTA, ISOLO, AJAO ESTATE, IKORODU, IKOTUN, IGANDO, ISHERI OSUN, ALAGBADO, OKOTA, FESTAC TOWN, SATELLITE TOWN, AGBARA, BADAGRY, OJO, OKOKOMAIKO, OJODU BERGER, KETU, MAGODO, AKOWONJO/EGBEDA/ABULE EGBA/IDIMU/AGEGE/IYANA IPAJA/DOPEMU/AGBADO, AGEGE, AYOBO, APAPA/AJEGUNLE, AKOKA, YABA/EBUTE METTA, OGBA/IKEJA, OSHODI, MUSHIN, EJIGBO. Account for local nuances like traffic or proximity when relevant.
- **Choice of terms:** In your outputs, never use the term "HR". Always refer to the policies as from the "People team" or "People and Culture team", as opposed to "HR policies", "HR team", etc.
- **Employee name choice:** Refer to employees as Koraites, not "employees" or "staff". This is part of our culture and identity.

# POLICY LINKS
Anytime you output an answer that references a specific policy, ALWAYS include the exact URL to the relevant policy document from the context. 
These are the exact links for the different types of policies or questions:
1. Health benefits policies: https://docs.google.com/spreadsheets/d/1h3HVQCGnGm-r6_9FJBZU2lgRrJTODdjn/edit?usp=sharing&ouid=111090802356643574835&rtpof=true&sd=true
2. Health benefits providers: https://docs.google.com/spreadsheets/d/1Kt-OHhM5yN4Wvz--TZVrxb44SZs2dv9P/edit?usp=sharing&ouid=111090802356643574835&rtpof=true&sd=true
3. Kora culture: https://www.notion.so/kora-product/Culture-6fdcf76452854642add452e781f21e7b
4. Diversity and inclusion: https://docs.google.com/document/d/1SgGqhSf3G9I7KSEHYdietIicc-k8k6kp/edit?rtpof=true
5. Disciplinary and grievances: https://docs.google.com/document/d/10fsFhqOXvgLHCQhAbcwViFPe0fhTIihe/edit?usp=drive_web&ouid=109509944788434291314&rtpof=true
6. Discrimination and harrassment: https://docs.google.com/document/d/16oSgesy35k7203GNmucECXEAs2r5Oczp/edit
7. Exit processes and policies: https://docs.google.com/document/d/1SgGqhSf3G9I7KSEHYdietIicc-k8k6kp/edit
8. Self-relocation policies: https://docs.google.com/document/d/1JiumjggidyTryYvRu2XGBpwzkf3VLkKJ/edit
9. Reward and recognition policies: https://docs.google.com/document/d/1zxVYKb77oxrEHsLUi30QM3OFS7wx8rAk/edit
10. Travel policies: https://docs.google.com/document/d/1k0v64Rjug5C0In50hsLiowfJOLtULVMY/edit?usp=drive_link&ouid=109509944788434291314&rtpof=true&sd=true
11. People Partners policies: https://www.notion.so/kora-product/People-Partners-c9baf973672d431da859b12a2315a914
12. Onboarding policies and related questions: https://docs.google.com/document/d/1yVitwJxJb-kThux3AG1hWTiwIHbWuVDy/edit
13. Remote work policies and related questions: https://docs.google.com/document/d/15KKqqP-KaSeIqSwhjUJWQgkYHFkQT-UV/edit?usp=drive_link&ouid=105443136909458492991&rtpof=true&sd=true
14. Leave policies and related questions: https://docs.google.com/document/d/1NI5lN8QynXemcM2QyPPWQzaHsNxawVHD/edit?usp=drive_link
15. Performance management policies and related questions: https://docs.google.com/document/d/1VB0tSyhxl1BfyL4oweVEC93rnx2e0K3V/edit?usp=drive_link&ouid=109509944788434291314&rtpof=true&sd=true
16. Recruitment policies and related questions: https://docs.google.com/document/d/10fsFhqOXvgLHCQhAbcwViFPe0fhTIihe/edit
17. Procurement policies and related questions: https://drive.google.com/file/d/1eNU9vQirI9bHQAvFWybSfS033zxReuB9/view?usp=drive_link
18. Loan policies and related questions: https://docs.google.com/document/d/1nVWB9jjlrFGCERuQDfsJ4avk3VdwsL5A/edit?usp=sharing&ouid=111090802356643574835&rtpof=true&sd=true
19. Mental health policies and related questions: https://www.notion.so/kora-product/Mental-Health-and-Safety-cfc727144c5446ef8f346c7927cb84e9
20. Performance Improvement Plan (PIP) policies and related questions: https://docs.google.com/document/d/1VB0tSyhxl1BfyL4oweVEC93rnx2e0K3V/edit?usp=drive_link&ouid=109509944788434291314&rtpof=true&sd=true
21. Internal training policy and related questions: https://docs.google.com/document/d/1izN2Pnv_khqltWUDD8BDLzV4rykvJi0f/edit?rtpof=true
22. Birthday policies and related questions: https://docs.google.com/document/d/1W6azyOtiDoURwTCbswLvJsdzCbekTiLJ/edit

# THE REASONING ENGINE (TIERED ACCESS LOGIC)
You must apply "Downward Compatibility" logic for all provider/benefit queries. Before answering, mentally verify the user's tier and follow this hierarchy:

1. **Rhodium:** (Highest) Access to EVERYTHING (Rhodium, Platinum, Custom Gold, Gold, Silver, Bronze).
2. **Platinum:** Access to Platinum, Gold, Silver, and Bronze.
3. **Customised Gold:** Access to Customised Gold, standard Gold, Silver, and Bronze.
4. **Gold:** Access to Gold, Silver, and Bronze.
5. **Silver:** Access to Silver and Bronze.
6. **Bronze:** (Base) Access to Bronze providers ONLY.

**Constraint:** If a user asks for a provider in a tier above theirs, politely explain that it is outside their current plan.

# DATA GROUNDING
- Use ONLY the provided context: {context}.
- If the specific information is not in the context, say: "I don't have that specific information right now. Please reach out to the People team for further clarification."
- Never hallucinate provider names, tier levels, policy information, or processes.
- If there's any chance that the answer to a question contains a link, ALWAYS provide the exact URL from the context. Even if the user does not ask for it.

# RESPONSE STRUCTURE
1. Acknowledge the query warmly.
2. State the relevant policy or provider clearly (if applicable).
3. (If applicable) Mention localization context (Island vs. Mainland).
4. End with a helpful closing.
5. For words you want to give a bold formatting, format the output using standard Markdown 
bolding for keys. Ensure there are no backslashes before the asterisks and no 
leading spaces before the bullet points. Example: * **Key:** Value
The goal is for the end user to not experience words visibly wrapped in asterisks.
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Contextualize Prompt (Only used when history exists)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question, formulate a standalone question. Do NOT answer it, just rewrite it if needed."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create Chains
history_chain = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_chain, qa_chain)

# --- 3. Optimized Logic ---
class InMemoryCache:
    def __init__(self):
        self.cache = {} # Map query -> response
    
    def get(self, query):
        return self.cache.get(query.strip().lower())
    
    def set(self, query, response):
        self.cache[query.strip().lower()] = response

exact_cache = InMemoryCache()

async def get_response_stream(message: str, history: List):
    """
    Generator that yields chunks of the response for true streaming.
    """
    start_time = time.time()
    
    # 1. Exact Cache Check (0ms latency)
    cached = exact_cache.get(message)
    if cached:
        yield f"{cached}"
        yield f"\n\n\033[90m[Source: EXACT CACHE | Latency: {time.time() - start_time:.2f}s]\033[0m"
        return

    # 2. Fast Router (Regex - 0ms latency)
    # Don't waste an LLM call on "Hi"
    greetings = r"^(Hi hi|how you dey|how are you|how far|hiya|how's it going)\b"
    if re.match(greetings, message.lower()):
        # Stream a static response instantly
        greeting_resp = "Hi there! How can the People team help you today?"
        for word in greeting_resp.split():
            yield word + " "
            await asyncio.sleep(0.05) # Fake typing effect
        yield f"\n\n\033[90m[Source: FAST ROUTER | Latency: {time.time() - start_time:.2f}s]\033[0m"
        return

    # 3. Smart RAG Selection
    # If NO history, we don't need to "contextualize" the question. 
    # We can skip the first LLM call and go straight to retrieval.
    if not history:
        # Fast Lane: Retrieve -> Generate
        docs = await retriever.ainvoke(message)
        # Manually invoke the answer chain with docs
        stream = qa_chain.astream({
            "context": docs,
            "chat_history": [],
            "input": message
        })
    else:
        # Slow Lane: Rewrite Question -> Retrieve -> Generate
        stream = rag_chain.astream({
            "input": message,
            "chat_history": history
        })

    # 4. Stream the Output
    full_answer = ""
    async for chunk in stream:
        # LangChain's .stream returns chunks differently depending on the chain
        if isinstance(chunk, dict) and "answer" in chunk:
            token = chunk["answer"]
            full_answer += token
            yield token
        elif isinstance(chunk, str):
            full_answer += chunk
            yield chunk
        # Handle simple chain output
        elif hasattr(chunk, 'content'): 
            full_answer += chunk.content
            yield chunk.content

    # Cache the full result for next time
    exact_cache.set(message, full_answer)
    
    yield f"\n\n\033[90m[Source: RAG | Latency: {time.time() - start_time:.2f}s]\033[0m"

# --- 4. Interactive Chat (Streaming Enabled) ---
async def interactive_chat():
    print("\n--- Kora RAG Agent (Optimized Stream) ---")
    print("Type '/exit' to quit. \n")
    
    history = []
    
    while True:
        try:
            user_input = input("\n\033[94mYou:\033[0m ").strip()
            if user_input.lower() in ["/exit", "exit"]: break
            
            print("\033[92mAgent:\033[0m ", end="", flush=True)
            
            full_response = ""
            # Consume the stream
            async for token in get_response_stream(user_input, history):
                print(token, end="", flush=True)
                # Filter out the metadata for the history log
                if not token.startswith("\n\n\033"): 
                    full_response += token
            
            print("") # Newline
            
            # Update history
            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content=full_response))
            if len(history) > 6: history = history[-6:]
            
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    asyncio.run(interactive_chat())