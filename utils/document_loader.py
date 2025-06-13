import os
import sys
from dotenv import load_dotenv
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import json
import numpy as np

# Load environment variables
load_dotenv()

# Constants
REGULATIONS_PDF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                    "data", 
                                    "mnr-2025-fishing-regulations-summary-en-2024-12-09_0.pdf")
EMBEDDINGS_PATH = "/tmp/embeddings.json"

def check_openai_api_key():
    """Check if OpenAI API key is set"""
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it in your .env file or environment variables.")
        sys.exit(1)

def load_and_split_document() -> List:
    """Load and split the PDF document into chunks"""
    if not os.path.exists(REGULATIONS_PDF_PATH):
        print(f"Error: PDF document not found at: {REGULATIONS_PDF_PATH}")
        sys.exit(1)
        
    # Load the document using PyPDFLoader
    loader = PyPDFLoader(REGULATIONS_PDF_PATH)
    documents = loader.load()
    
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

def get_relevant_documents(query: str, k: int = 5):
    """Get the most relevant documents for a query"""
    global _embeddings, _documents
    
    # Load embeddings if not loaded
    if _embeddings is None or _documents is None:
        if os.path.exists(EMBEDDINGS_PATH):
            with open(EMBEDDINGS_PATH, 'r') as f:
                data = json.load(f)
                stored_embeddings = data['embeddings']
                stored_texts = data['texts']
                _embeddings = OpenAIEmbeddings()
                _documents = [Document(page_content=text) for text in stored_texts]
        else:
            load_and_embed_document()
    
    # Get query embedding
    query_embedding = _embeddings.embed_query(query)
    
    # Calculate similarities
    similarities = []
    with open(EMBEDDINGS_PATH, 'r') as f:
        data = json.load(f)
        stored_embeddings = data['embeddings']
        
    for doc_embedding in stored_embeddings:
        similarity = np.dot(query_embedding, doc_embedding)
        similarities.append(similarity)
    
    # Get top k most similar documents
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    return [_documents[i] for i in top_k_indices] 