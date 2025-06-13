"""
This script downloads the regulations PDF, generates embeddings for its content,
and saves them to a JSON file for the main application to use.

This is a one-time setup script that should be run locally.
"""
import os
import sys
import json
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
PDF_URL = "https://files.ontario.ca/mnr-2025-fishing-regulations-summary-en-2024-12-09_0.pdf"
LOCAL_PDF_PATH = os.path.join("data", "regulations.pdf")
EMBEDDINGS_PATH = os.path.join("data", "embeddings.json")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def check_openai_api_key():
    """Checks if the OpenAI API key is set in the environment variables."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please create a .env file and add your OPENAI_API_KEY to it.")
        sys.exit(1)

def download_pdf():
    """Downloads the PDF from the specified URL if it doesn't already exist."""
    if os.path.exists(LOCAL_PDF_PATH):
        print(f"PDF already exists at {LOCAL_PDF_PATH}. Skipping download.")
        return

    print(f"Downloading PDF from {PDF_URL}...")
    try:
        # Add a user-agent header to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(PDF_URL, stream=True, headers=headers)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(LOCAL_PDF_PATH), exist_ok=True)
        
        with open(LOCAL_PDF_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("PDF downloaded successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF: {e}")
        sys.exit(1)

def generate_and_save_embeddings():
    """Loads the document, splits it, generates embeddings, and saves them to a file."""
    if not os.path.exists(LOCAL_PDF_PATH):
        print(f"Error: PDF file not found at {LOCAL_PDF_PATH}")
        print("Please download the PDF manually and place it in the 'data' directory.")
        sys.exit(1)

    print("Loading and splitting the document...")
    loader = PyPDFLoader(LOCAL_PDF_PATH)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\nZone ", "\nSeason: ", "\nLimits: ", "\n", ". ", " ", ""],
        length_function=len,
    )
    split_docs = text_splitter.split_documents(documents)
    
    if not split_docs:
        print("Error: No documents were split. Check the PDF content.")
        sys.exit(1)

    print(f"Document split into {len(split_docs)} chunks.")
    print("Generating embeddings with OpenAI... (This may take a while)")
    
    try:
        embeddings_model = OpenAIEmbeddings()
        
        doc_embeddings = embeddings_model.embed_documents([doc.page_content for doc in split_docs])
        
        print("Embeddings generated successfully.")
        
        # Save embeddings and the corresponding text chunks
        with open(EMBEDDINGS_PATH, 'w') as f:
            json.dump({
                'embeddings': doc_embeddings,
                'texts': [doc.page_content for doc in split_docs]
            }, f)
            
        print(f"Embeddings saved to {EMBEDDINGS_PATH}")
    except Exception as e:
        print(f"An error occurred while generating embeddings: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_openai_api_key()
    # download_pdf() # Download is now a manual step
    generate_and_save_embeddings()
    print("\nSetup complete. You can now commit the 'data/embeddings.json' file.") 