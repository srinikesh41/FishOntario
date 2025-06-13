import os
import sys
from dotenv import load_dotenv
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import json
import numpy as np
import requests
from pathlib import Path
from functools import lru_cache

# Load environment variables
load_dotenv()

# Constants
PDF_URL = os.environ.get('PDF_URL', 'https://files.ontario.ca/mnr-2025-fishing-regulations-summary-en-2024-12-09_0.pdf')
EMBEDDINGS_DIR = os.path.join(os.getcwd(), 'tmp')
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings.json")

# --- In-memory cache for the data ---
_embedding_data = {}

@lru_cache(maxsize=None)
def _load_embedding_data():
    """Loads the pre-computed embeddings and texts from the JSON file."""
    global _embedding_data
    if not _embedding_data:
        if not os.path.exists(EMBEDDINGS_PATH):
            print(f"FATAL: Embeddings file not found at {EMBEDDINGS_PATH}.")
            print("Please run the 'generate_embeddings.py' script first.")
            sys.exit(1)
        try:
            with open(EMBEDDINGS_PATH, 'r') as f:
                data = json.load(f)
            
            _embedding_data['embeddings'] = np.array(data['embeddings'])
            _embedding_data['documents'] = [Document(page_content=text) for text in data['texts']]
            
            print(f"Successfully loaded {_embedding_data['embeddings'].shape[0]} documents from cache.")
        except (IOError, json.JSONDecodeError) as e:
            print(f"FATAL: Could not read or parse embeddings file: {e}")
            sys.exit(1)
    return _embedding_data

def ensure_temp_dir():
    """Ensure temporary directory exists"""
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

def download_pdf():
    """Download PDF from URL"""
    try:
        response = requests.get(PDF_URL)
        response.raise_for_status()
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
        
        # Save PDF temporarily
        pdf_path = os.path.join(EMBEDDINGS_DIR, 'regulations.pdf')
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        return pdf_path
    except Exception as e:
        print(f"Error downloading PDF: {str(e)}")
        sys.exit(1)

def check_openai_api_key():
    """Check if OpenAI API key is set"""
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

def load_and_split_document() -> List:
    """Load and split the PDF document into chunks"""
    pdf_path = download_pdf()
    
    # Load the document using PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Clean up temporary PDF file
    try:
        os.remove(pdf_path)
    except:
        pass
    
    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\nZone ", "\nSeason: ", "\nLimits: ", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    split_docs = text_splitter.split_documents(documents)
    return split_docs

_embeddings = None
_documents = None

def load_and_embed_document():
    """Load the document, split it and create embeddings"""
    global _embeddings, _documents
    check_openai_api_key()
    
    # Ensure temp directory exists
    ensure_temp_dir()
    
    # Try to load existing embeddings
    if os.path.exists(EMBEDDINGS_PATH):
        try:
            with open(EMBEDDINGS_PATH, 'r') as f:
                data = json.load(f)
                stored_texts = data['texts']
                _documents = [Document(page_content=text) for text in stored_texts]
                _embeddings = OpenAIEmbeddings()
                return
        except:
            pass
    
    # Load and split the document
    split_docs = load_and_split_document()
    _documents = split_docs
    
    # Create embeddings
    embeddings = OpenAIEmbeddings()
    _embeddings = embeddings
    
    # Generate embeddings for all documents
    doc_embeddings = []
    for doc in split_docs:
        emb = embeddings.embed_query(doc.page_content)
        doc_embeddings.append(emb)
    
    # Save embeddings to file
    with open(EMBEDDINGS_PATH, 'w') as f:
        json.dump({
            'embeddings': doc_embeddings,
            'texts': [doc.page_content for doc in split_docs]
        }, f)

def get_relevant_documents(query: str, k: int = 5) -> List[Document]:
    """
    Gets the most relevant documents for a query using pre-computed embeddings.
    """
    data = _load_embedding_data()
    
    query_embedding = OpenAIEmbeddings().embed_query(query)
    
    # Calculate dot product similarity
    similarities = np.dot(data['embeddings'], query_embedding)
    
    # Get top k indices
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    return [data['documents'][i] for i in top_k_indices] 