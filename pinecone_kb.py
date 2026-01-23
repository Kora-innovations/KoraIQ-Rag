import os
import time
import logging
import pandas as pd
from typing import List
from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_pinecone import PineconeVectorStore
from langchain_experimental.text_splitter import SemanticChunker
from pinecone import Pinecone, ServerlessSpec

# logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest")

# --- env variables ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pinecone-test-index")

assert OPENAI_API_KEY, "OPENAI_API_KEY not set"
assert PINECONE_API_KEY, "PINECONE_API_KEY not set"

# --- Initialize Pinecone Client ---
pc = Pinecone(api_key=PINECONE_API_KEY)

# --- Embeddings & Chunker ---
# using text-embedding-3-small for speed and cost efficiency
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY, chunk_size=200)

def setup_pinecone_index():
    """Checks if index exists, creates a Serverless index if not."""
    existing_indexes = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        logger.info(f"Creating Pinecone Index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,  # text-embedding-3-small default
            metric="cosine", 
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1" # might change to EU region
            )
        )
        # Wait for index to be ready
        while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
            time.sleep(1)
    else:
        logger.info(f"Index {PINECONE_INDEX_NAME} already exists.")

def get_vectorstore() -> PineconeVectorStore:
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )

# --- Loaders ---
def load_pdfs(pdf_paths: List[str]) -> List[Document]:
    all_pages: List[Document] = []
    for pdf_path in pdf_paths:
        try:
            pages = PyPDFLoader(pdf_path).load()
            # Clean metadata to avoid vector store errors
            for page in pages:
                page.metadata['source'] = os.path.basename(pdf_path)
            all_pages.extend(pages)
            logger.info(f"Loaded {len(pages)} pages from PDF: {pdf_path}")
        except Exception as e:
            logger.error(f"Failed to load PDF ({pdf_path}): {e}")
    return all_pages

# excel loader with metadata
def load_csv_with_metadata(csv_paths: List[str]) -> List[Document]:
    all_docs = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            df = df.fillna("")
            for _, row in df.iterrows():
                # 1. Construct the Searchable Text (Page Content)
                # Combine all fields into a natural language string. 
                # This helps the vector search match queries like "hospitals in Ikeja".
                content = (
                    f"Hospital Code: {row.get('Code', '')}. "
                    f"Hospital Name: {row.get('Name', '')}. "
                    f"Location: {row.get('Address', '')}, {row.get('City', '')}, {row.get('State', '')}. "
                    f"Plan Type: {row.get('Hospital Class', '')}. "
                    f"Services: {row.get('ServiceType', 'General')}. "  # Adjust column names as needed
                    f"Email: {row.get('Email', '')}. "
                    f"Phone Number: {row.get('Phone', '')}. "
                )
                # 2. Construct Metadata (The Fix for your issue)
                # This allows for precise filtering later if you decide to implement it,
                # and helps the Reranker (Flashrank) see the structured data clearly.
                metadata = {
                    "source": os.path.basename(path),
                    "row_index": _,
                    "hospital_name": str(row.get('Name', '')),
                    "city": str(row.get('City', '')), # might need to add for 'address' later on
                    "state": str(row.get('State', '')),
                    "plan": str(row.get('Hospital Class', '')),
                    "service_type": str(row.get('ServiceType', '')),
                    "email": str(row.get('Email', '')),
                    "phone": str(row.get('Phone', ''))
                }
                # create Document
                doc = Document(page_content=content, metadata=metadata)
                all_docs.append(doc)                
            logger.info(f"Loaded {len(df)} rows from {path} with rich metadata.")
            
        except Exception as e:
            logger.error(f"Failed to process {path}: {e}")
            
    return all_docs

# --- Semantic Chunking: added to boost accuracy ---
def split_documents_semantically(docs: List[Document]) -> List[Document]:
    """
    Uses OpenAI embeddings to determine where to split text based on 
    semantic similarity breakpoints rather than arbitrary characters.
    """
    if not docs:
        return []
    
    logger.info("Starting semantic chunking...")
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile" # Splits when difference is in top percentile
    )
    
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Split {len(docs)} raw docs into {len(chunks)} semantic chunks.")
    return chunks

# --- Ingest Flow ---
def ingest_to_pinecone(pdf_paths: List[str], csv_paths: List[str]):
    setup_pinecone_index()
    vectorstore = get_vectorstore()
    
    # 1. process PDFs
    raw_pdf_docs = load_pdfs(pdf_paths)
    if raw_pdf_docs:
        logger.info("Processing PDFs...")
        pdf_chunks = split_documents_semantically(raw_pdf_docs)
        if pdf_chunks:
            logger.info(f"Upserting {len(pdf_chunks)} PDF chunks...")
            vectorstore.add_documents(pdf_chunks)
    
    # 2. process Excel: No chunking needed as each row is already a perfect "chunk".
    if csv_paths:
        logger.info("Processing CSV files with Pandas...")
        csv_docs = load_csv_with_metadata(csv_paths)
        if csv_docs:
            logger.info(f"Upserting {len(csv_docs)} CSV rows...")
            # Batch upsert to prevent timeouts
            batch_size = 100
            for i in range(0, len(csv_docs), batch_size):
                batch = csv_docs[i:i+batch_size]
                vectorstore.add_documents(batch)
                logger.info(f"Upserted batch {i} to {i+len(batch)}")

    logger.info("Ingestion complete.")

if __name__ == "__main__": #add updated performance management before upserting
    PDFS = ["people_policies/DEI_FAQs.pdf", "people_policies/DEI_Policy.pdf",
            "people_policies/Disciplinary_Grievance_Policy.pdf", "people_policies/Disciplinary_Grievances_FAQs.pdf",
            "people_policies/Discrimination_Harassment_Bullying_FAQs.pdf", "people_policies/Discrimination_Harassment_Bullying_Policy.pdf",
            "people_policies/Exit_FAQs.pdf", "people_policies/Exit_Policy.pdf",
            "people_policies/Health_Insurance_FAQs.pdf",
            "people_policies/Internal_Training_FAQs.pdf", "people_policies/Internal_Training_Policy.pdf",
            "people_policies/Leave_FAQs.pdf", "people_policies/Leave_Policy.pdf",
            "people_policies/Loan_FAQs.pdf", "people_policies/Loan_Policy.pdf",
            "people_policies/Onboarding_FAQs.pdf", "people_policies/Onboarding_Policy.pdf",
            "people_policies/Procurement_Policy.pdf", "people_policies/Performance_Management_Policy.pdf",
            "people_policies/Recruitment FAQs.pdf", "people_policies/Recruitment_Policy.pdf",
            "people_policies/Remote Work FAQs.pdf", "people_policies/Remote_Work.pdf",
            "people_policies/Reward_Recognition_Policy.pdf", "people_policies/Rewards_Benefits_FAQs.pdf",
            "people_policies/Self_Relocation_FAQs.pdf", "people_policies/Self_Relocation_Policy.pdf",
            "people_policies/Travel_FAQs.pdf", "people_policies/Travel_Policy.pdf"] 
    CSV = ["people_policies/axa_may_2025_all_sheets.xlsx", "people_policies/axa_may_2025_dental.csv",
           "people_policies/axa_may_2025_diagnostics.csv", "people_policies/axa_may_2025_general.csv",
           "people_policies/axa_may_2025_optical.csv", "people_policies/axa_may_2025_paediatrics.csv",
           "people_policies/axa_may_2025_physiotherapy.csv", "people_policies/axa_may_2025_referrals.csv"]
    ingest_to_pinecone(PDFS, CSV)
    print("Knowledgebase successfully ingested.")


    #DONE THESE: PDFs: general_faqs.pdf, general_faqs_2.pdf, Corporate Health Plan Benefits & Limits 2025.pdf
    #csvs: axa_may_2025_dental.csv, axa_may_2025_diagnostics.csv, axa_may_2025_general.csv, axa_may_2025_optical.csv, 
    # axa_may_2025_paediatrics.csv, axa_may_2025_physiotherapy.csv, axa_may_2025_referrals.csv