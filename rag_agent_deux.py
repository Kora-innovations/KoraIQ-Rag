import os
import argparse
import contextvars
from typing import List, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain.schema import Document
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain_tavily import TavilySearch

from pymongo import MongoClient

# --- Globals ---
SESSIONS: Dict[str, ChatMessageHistory] = {}
current_session_id = contextvars.ContextVar("current_session_id")

# --- Mongo Connection ---
MONGODB_ATLAS_URI = os.getenv("MONGODB_ATLAS_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "ragdb")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "rag_docs")
MONGODB_INDEX_NAME = os.getenv("MONGODB_INDEX_NAME", "vector_index")

client = MongoClient(MONGODB_ATLAS_URI)
collection = client[MONGODB_DB][MONGODB_COLLECTION]

# --- Embeddings ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- VectorStore ---
vectorstore = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name=MONGODB_INDEX_NAME,
    text_key="text",
    embedding_key="embedding"
)

# --- History-Aware RAG Chain ---
from langchain.chains import ConversationalRetrievalChain

def build_history_aware_rag():
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
        You are a helpful, accurate, and professional QnA assistant designed to support employees of a Nigerian company with health insurance-related and general HR questions.
        Context: {context}
        Question: {question}
    """)

    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False
    )

qa_chain = build_history_aware_rag()

# --- Tools ---
search = TavilySearch()

def search_web(query: str) -> str:
    prefix = ("Nigerian context. When users say 'the Island', interpret as Lagos Island axis "
              "(Lekki, VI, Ikoyi, Ajah, Oniru). ")
    return search.run(prefix + query)

def rag_tool_func(query: str) -> str:
    sid = current_session_id.get(None)
    history = SESSIONS.get(sid, ChatMessageHistory())
    result = qa_chain.invoke({"question": query, "chat_history": history.messages})
    return result["answer"] if "answer" in result else result.get("result", "")

rag_tool = Tool(
    name="search_internal_docs",
    func=rag_tool_func,
    description="Use for questions answerable with internal company docs (insurance coverage, provider lists, internal FAQs, etc)."
)

web_tool = Tool(
    name="search_web",
    func=search_web,
    description="Search the public web for up-to-date info: hospital reviews, rankings, addresses, recent news."
)

# --- Agent ---
nigerian_system_prompt = """
You are a helpful, accurate, and professional QnA assistant supporting employees of a Nigerian fintech company with health insurance, HR, and provider-related questions.
Localization rules:
- 'The Island' means the Lagos Island axis (Lekki, VI, Ikoyi, Ajah, Oniru).
- 'The Mainland' means places like Yaba, Ikeja, Surulere, Maryland.
- Default all context to Nigerian usage and culture.
When users ask for hospital or clinic recommendations, interpret requests in the Nigerian context, using recent data where possible. Never reveal private information about another employee.
Always explain the reasoning for your recommendations. If you don't have enough information, ask clarifying questions using Nigerian examples.
"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(nigerian_system_prompt),
    MessagesPlaceholder("chat_history"),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

agent = create_openai_functions_agent(
    llm=llm,
    tools=[rag_tool, web_tool],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[rag_tool, web_tool],
    verbose=True,
    handle_parsing_errors=True
)

# --- Ingestion ---
def ingest_files(files: List[str]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_docs: List[Document] = []

    for file in files:
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file)
        elif file.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(file)
        else:
            raise ValueError(f"Unsupported file type: {file}")
        docs = loader.load()
        splits = splitter.split_documents(docs)
        all_docs.extend(splits)

    vectorstore.add_documents(all_docs)
    print(f"Ingested {len(all_docs)} chunks into MongoDB Atlas")

# --- FastAPI ---
app = FastAPI()

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    sid = req.session_id
    current_session_id.set(sid)
    if sid not in SESSIONS:
        SESSIONS[sid] = ChatMessageHistory()

    history = SESSIONS[sid]
    result = agent_executor.invoke({
        "input": req.message,
        "chat_history": history.messages
    })

    history.add_user_message(req.message)
    history.add_ai_message(result["output"])

    return {"response": result["output"]}

# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--files", nargs="+", required=True)

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "ingest":
        ingest_files(args.files)
    elif args.command == "serve":
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        parser.print_help()
