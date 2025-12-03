import os
import logging
from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient

logger = logging.getLogger("ingest")

# --- ENV (assumes you’ve already loaded .env earlier) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_ATLAS_URI = os.getenv("MONGODB_ATLAS_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "korahq_dev")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "ragdb")
MONGODB_INDEX_NAME = os.getenv("MONGODB_INDEX_NAME", "rag_vector_index_3")  # Atlas Vector Search index name

assert OPENAI_API_KEY, "OPENAI_API_KEY not set"
assert MONGODB_ATLAS_URI, "MONGODB_ATLAS_URI not set"

# --- Mongo collection + VectorStore factory ---
_client = MongoClient(MONGODB_ATLAS_URI)
_collection = _client[MONGODB_DB][MONGODB_COLLECTION]
_embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)  # default model: text-embedding-3-small (1536 dims)

def get_vectorstore() -> MongoDBAtlasVectorSearch:
    return MongoDBAtlasVectorSearch(
        collection=_collection,
        embedding=_embeddings,
        index_name=MONGODB_INDEX_NAME,
        text_key="text",
        embedding_key="embedding"
        #metadata_key="metadata",
    )

# --- Loaders ---
def load_pdfs(pdf_paths: List[str]) -> List[Document]:
    all_pages: List[Document] = []
    for pdf_path in pdf_paths:
        try:
            pages = PyPDFLoader(pdf_path).load()
            all_pages.extend(pages)
            logger.info(f"Loaded {len(pages)} pages from PDF: {pdf_path}")
        except Exception as e:
            logger.error(f"Failed to load PDF ({pdf_path}): {e}")
    print(f"Loaded {len(all_pages)} pages from PDFs.")
    logger.info(f"Loaded {len(all_pages)} PDF pages total.")
    return all_pages

def load_xlsx_langchain(xlsx_paths: List[str]) -> List[Document]:
    """Uses UnstructuredExcelLoader → one row == one Document with metadata(sheet, row, etc.)."""
    all_docs: List[Document] = []
    for xlsx_path in xlsx_paths:
        try:
            docs = UnstructuredExcelLoader(xlsx_path).load()
            all_docs.extend(docs)
            logger.info(f"Loaded {len(docs)} docs from {xlsx_path}")
        except Exception as e:
            logger.error(f"Failed to load XLSX ({xlsx_path}): {e}")
    print(f"Total loaded from XLSX files: {len(all_docs)}")
    logger.info(f"Total loaded from XLSX files: {len(all_docs)}")
    return all_docs

# --- Splitting (PDF only) ---
def split_pdf_docs(pdf_docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(pdf_docs)
    print(f"PDF split into {len(chunks)} chunks.")
    logger.info(f"Split {len(pdf_docs)} PDF docs into {len(chunks)} chunks.")
    return chunks

# --- Clean metadata for Atlas (no arrays/objects that Chroma complained about; Atlas is flexible but keep it tidy) ---
def prepare_docs_for_embedding(splits: List[Document]) -> List[Document]:
    clean_splits = []
    for idx, s in enumerate(splits):
        if isinstance(s, Document):
            s.metadata = filter_complex_metadata([s])[0].metadata
            if s.page_content and len(s.page_content.strip()) > 20:
                clean_splits.append(s)
        else:
            print(f"Warning: splits[{idx}] is not a Document (is {type(s)}), skipping. Value: {repr(s)[:80]}")
    print(f"{len(clean_splits)} splits ready for embedding.")
    return clean_splits
    # clean: List[Document] = []
    # for idx, d in enumerate(docs):
    #     if not isinstance(d, Document):
    #         logger.warning(f"Skipping non-Document at index {idx}: {type(d)}")
    #         continue
    #     if not d.page_content or len(d.page_content.strip()) < 20:
    #         continue
    #     # filter_complex_metadata expects a list; returns sanitized copy
    #     d.metadata = filter_complex_metadata([d])[0].metadata
    #     clean.append(d)
    # logger.info(f"{len(clean)} docs ready for embedding.")
    # return clean

# --- Embed + persist to MongoDB Atlas Vector Search ---
def embed_and_persist(clean_docs: List[Document]) -> MongoDBAtlasVectorSearch:
    if not clean_docs:
        raise ValueError("No documents to embed.")
    vs = get_vectorstore()
    # This computes embeddings with OpenAI and upserts documents (text + embedding + metadata) into Atlas
    vs.add_documents(clean_docs)
    print("Vector store created and persisted.")
    logger.info(f"Persisted {len(clean_docs)} documents into MongoDB Atlas Vector Search.")
    return vs

# --- High-level ingest helper (drop-in replacement for the notebook flow) ---
def ingest_to_mongo(pdf_paths: List[str], xlsx_paths: List[str]) -> MongoDBAtlasVectorSearch:
    pdf_docs = load_pdfs(pdf_paths)
    xlsx_docs = load_xlsx_langchain(xlsx_paths)

    # Only split PDFs
    pdf_chunks = split_pdf_docs(pdf_docs) if pdf_docs else []

    # Concatenate: PDF chunks + Excel row-docs
    all_docs_for_embedding = pdf_chunks + xlsx_docs
    print(f"Total documents for embedding: {len(all_docs_for_embedding)}")
    # Clean & embed
    clean_splits = prepare_docs_for_embedding(all_docs_for_embedding)
    vectorstore = embed_and_persist(clean_splits)
    return vectorstore

# ======== Example usage (comment out in production) ========
if __name__ == "__main__":
    PDF_PATHS = ["general_faqs.pdf", "general_faqs_2.pdf"]  # replace with real paths
    XLSX_PATHS = ["axa_may_2025_provider_list.xlsx", "customised_plan_and_benefits_2025.xlsx"]
    ingest_to_mongo(PDF_PATHS, XLSX_PATHS)
    # def find_by_source(src_name: str, limit=3):
    #     for d in _collection.find({"metadata.source": src_name}, {"_id":0,"text":1,"metadata":1}).limit(limit):
    #         print("\n---")
    #         print(d["text"][:400], "...")
    #         print("META:", d["metadata"])
    # find_by_source("customised_plan_and_benefits_2025.xlsx", limit=3)

    print("Ingestion complete. Your MongoDB Atlas collection now holds your vectors.")
