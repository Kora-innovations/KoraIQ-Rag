import os
import sys
from dotenv import load_dotenv, find_dotenv
import pandas as pd

# --- ENVIRONMENT ---
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")

assert OPENAI_API_KEY, "OPENAI_API_KEY not set"
assert LANGSMITH_API_KEY, "LANGSMITH_API_KEY not set"
assert LANGSMITH_PROJECT, "LANGSMITH_PROJECT not set"

# --- LANGCHAIN & LANGSMITH SETUP ---
from langsmith import Client as LangSmithClient, traceable
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

# # Optional: set up LangSmith for tracing
# import langsmith
# langsmith_client = LangSmithClient(api_key=LANGSMITH_API_KEY) #removed project_name parameter
# langsmith.enable_tracing() #come back to this later

# --- DOCUMENT LOADING ---
def load_pdfs(pdf_paths):
    """
    Load and return all pages from multiple PDF files.
    """
    all_pages = []
    for pdf_path in pdf_paths:
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            all_pages.extend(pages)
        except Exception as e:
            print(f"Failed to load PDF ({pdf_path}): {e}")
    return all_pages

def load_xlxs(xlsx_paths):
    """
    Load and return all elements from multiple XLSX files.
    """
    all_docs = []
    for xlsx_path in xlsx_paths:
        try:
            loader = UnstructuredExcelLoader(xlsx_path, mode="elements")
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            print(f"Failed to load XLSX ({xlsx_path}): {e}")
    return all_docs

def load_xlsx_as_documents(xlsx_paths):
    """
    Loads data from specified Excel files, treating each row as a separate document.

    This optimized function:
    1. Reads each sheet in an Excel file.
    2. Cleans the data by removing empty rows based on a key column.
    3. Converts each row into a structured string format.
    4. Attaches critical metadata (source file and sheet name) for RAG.
    """
    all_docs = []
    for xlsx_path in xlsx_paths:
        try:
            # Load the entire Excel file to get sheet names
            xls = pd.ExcelFile(xlsx_path)
            for sheet_name in xls.sheet_names:
                # Read the specific sheet into a DataFrame
                df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
                
                # --- OPTIMIZATION: Data Cleaning ---
                # Identify the primary column that should not be empty. 
                # From your data, the second column holds the hospital name.
                if df.shape[1] < 2:
                    print(f"Warning: Sheet '{sheet_name}' in '{xlsx_path}' has fewer than 2 columns. Skipping.")
                    continue
                
                # Drop rows where the hospital name is missing to avoid creating empty documents.
                key_column = df.columns[1] 
                df.dropna(subset=[key_column], inplace=True)
                df.reset_index(drop=True, inplace=True)

                # Iterate through cleaned rows to create Document objects
                for _, row in df.iterrows():
                    # Create a clean, readable string from the row's data
                    content = "\n".join(
                        f"{col}: {row[col]}" 
                        for col in df.columns 
                        if pd.notna(row[col]) and str(row[col]).strip() != ''
                    )
                    
                    # Skip creating a document if the content is empty after cleaning
                    if not content:
                        continue
                        
                    metadata = {
                        "source": xlsx_path,
                        "sheet": sheet_name
                    }
                    doc = Document(page_content=content, metadata=metadata)
                    all_docs.append(doc)
        except Exception as e:
            print(f"Failed to process XLSX file ({xlsx_path}): {e}")
            
    return all_docs

def load_all_docs(pdf_paths, xlsx_paths):
    all_docs = []
    all_docs += load_pdfs(pdf_paths)
    all_docs += load_xlsx_as_documents(xlsx_paths)
    return all_docs


# --- SPLITTING (Context Aware) ---
def split_docs(docs, chunk_size=800, chunk_overlap=80):
    # Use RecursiveCharacterTextSplitter for context-aware splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = []
    for doc in docs:
        try:
            splits += splitter.split_documents([doc])
        except Exception as e:
            print(f"Splitting failed for doc: {e}")
            continue
    return splits

from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
# --- EMBEDDING & VECTORSTORE ---
def embed_docs(splits, persist_directory="chroma_db"):
    clean_splits = []
    for idx, s in enumerate(splits):
        if isinstance(s, Document):
            # Defensive metadata filter
            s.metadata = filter_complex_metadata(s.metadata)
            # Filter out empty/short
            if s.page_content and len(s.page_content.strip()) > 20:
                clean_splits.append(s)
        else:
            print(f"Warning: splits[{idx}] is not a Document object. It is {type(s)}. Value: {repr(s)[:80]}... Skipping.")

    if not clean_splits:
        raise Exception("No valid Document objects found after filtering splits. Check your loaders/splitter.")

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(clean_splits, embeddings, persist_directory=persist_directory)
    return vectorstore

# --- FAILURE MODES ---
def safe_load_and_embed(pdf_path, xlsx_paths, persist_directory="chroma_db"):
    docs = load_all_docs(pdf_path, xlsx_paths)
    if not docs:
        raise Exception("No documents loaded from inputs! Check your files.")
    splits = split_docs(docs)
    if not splits:
        raise Exception("Document splitting failed! Check the content/format of your docs.")
    try:
        vectorstore = embed_docs(splits, persist_directory)
    except Exception as e:
        raise Exception(f"Embedding failed: {e}")
    return vectorstore

# --- RETRIEVER, QA, AND CHAT ---
@traceable
def setup_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful, accurate, and professional QnA RAG (Retrieval Augmented Generation) bot designed
    to assist employees of a company with their health insurance-related questions. 
    Your primary goal is to provide clear, concise, and relevant answers based on the information 
    available in your knowledge bases.

    Knowledge Bases:
    You have access to information from the following sources:
    - One PDF document (general policy information).
    - customised_plan_and_benefits.xlsx (Excel spreadsheet containing detailed plan benefits and coverage limits).
    - provider_list.xlsx (Excel spreadsheet containing the list of hospitals and clinics in the network).

    Instructions for Answering Specific Question Types:
    For questions related to coverage limits, benefits, or detailed plan breakdowns (e.g., "What is my coverage limit under this plan?", "What are the maternity benefits?", "Does my plan cover dental?”), refer to: customised_plan_and_benefits.xlsx.
    Structure your answer similarly to this example: "Coverage limits depend on your specific plan (e.g. Gold, Platinum, Customized Gold, Customized Platinum ). They vary by category, such as inpatient, outpatient, maternity, dental, optical, and wellness. Please see customised_plan_and_benefits.xlsx for full details of the comprehensive breakdown."

    For questions related to the network of hospitals, clinics, or healthcare providers (e.g., "How can I find the list of hospitals or clinics covered by the plan?", "Is [Hospital Name] covered?”), refer to: provider_list.xlsx.
    Structure your answer similarly to this example: "You can access the current network list provider_list.xlsx or through the MY AXA Plus mobile app. The list may differ slightly based on your plan type, so we recommend checking the exact provider list under your specific plan."

    General Instructions:
    - Prioritize Knowledge Bases: Always strive to answer questions using information from the provided knowledge bases.
    - Accuracy: Ensure all information provided is accurate and directly supported by the retrieved context.
    - Conciseness: Provide direct answers without unnecessary jargon or lengthy explanations.
    - Clarity: Use clear and easy-to-understand language.
    - Handling Unanswered Questions: If a question cannot be answered definitively from the provided knowledge bases, politely state that the information is not available or suggest where the employee might find it (e.g., "I don't have that specific detail in my current knowledge base. You might want to contact the People team for further assistance.").
    - Maintain Professionalism: Always maintain a polite and helpful tone.


    Context: {context}
    Question: {question}
    """)
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # You can switch to map_reduce or refine for large docs
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain

def answer_question(qa_chain, question):
    try:
        response = qa_chain(question)
        answer = response['result']
        sources = response.get('source_documents', [])
        print(f"Answer: {answer}")
        if sources:
            print("Sources:")
            for doc in sources:
                print(f"- {getattr(doc, 'metadata', {}).get('source', 'unknown source')}")
        return answer
    except Exception as e:
        print(f"Error answering question: {e}")
        return "An error occurred while answering the question."

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    PDF_PATH = ["general_faqs.pdf"]
    XLSX_PATHS = ["axa_may_2025_provider_list.xlsx", "customised_plan_and_benefits_2025.xlsx"]

    # Edge-case: Verify all files exist
    for f in PDF_PATH + XLSX_PATHS:
        if not os.path.exists(f):
            print(f"Missing file: {f}")
            sys.exit(1)

    print("Loading and embedding documents...")
    vectorstore = safe_load_and_embed(PDF_PATH, XLSX_PATHS)

    print("Setting up Q&A chain...")
    qa_chain = setup_qa_chain(vectorstore)

    # --- Simple CLI Chat Loop ---
    print("Health Insurance QA Chatbot. Type 'exit' to quit.")
    while True:
        question = input("Ask a question: ")
        if question.lower() in ['exit', 'quit']:
            break
        answer_question(qa_chain, question)

