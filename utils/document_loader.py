import os
import sys
from dotenv import load_dotenv
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Constants
REGULATIONS_PDF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                    "data", 
                                    "mnr-2025-fishing-regulations-summary-en-2024-12-09_0.pdf")
CHROMA_PERSIST_DIRECTORY = os.environ.get("CHROMA_PERSIST_DIRECTORY", "./vector_db")

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
        separators=[
            "\n\n",  # Double line breaks often separate sections
            "\nZone ", # Zone headers
            "\nSeason: ", # Season information
            "\nLimits: ", # Limit information
            "\n", # Single line breaks
            ". ",  # Sentences
            " ",   # Words
            ""     # Characters
        ],
        length_function=len,
    )
    
    split_docs = text_splitter.split_documents(documents)
    return split_docs

def load_and_embed_document():
    """Load the document, split it and create a vector store"""
    check_openai_api_key()
    
    # Check if vector store exists
    if os.path.exists(CHROMA_PERSIST_DIRECTORY) and len(os.listdir(CHROMA_PERSIST_DIRECTORY)) > 0:
        print(f"Using existing vector store in {CHROMA_PERSIST_DIRECTORY}")
        # We don't need to recreate it, just return
        return
    
    print("Creating new vector store...")
    
    # Load and split the document
    split_docs = load_and_split_document()
    
    # Create embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create vector store
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY
    )
    
    print(f"Vector store created successfully with {len(split_docs)} document chunks")
    return vector_store

def get_vector_store():
    """Get the vector store for querying"""
    check_openai_api_key()
    
    # Create embeddings
    embeddings = OpenAIEmbeddings()
    
    # Load the existing vector store
    if not os.path.exists(CHROMA_PERSIST_DIRECTORY):
        # If it doesn't exist yet, create it
        return load_and_embed_document()
    
    vector_store = Chroma(
        persist_directory=CHROMA_PERSIST_DIRECTORY, 
        embedding_function=embeddings
    )
    return vector_store 