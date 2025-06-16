# Ontario Fishing Regulations QA System

A full-stack application that allows users to ask natural language questions about Ontario's 2025 Fishing Regulations. The system consists of a FastAPI backend with RAG (Retrieval Augmented Generation) capabilities and a React frontend, providing accurate answers with source citations from the official regulations document.

## Features

### Backend (FastAPI)
- **Natural language Q&A** about Ontario's 2025 Fishing Regulations
- **RAG system** using LangChain, OpenAI embeddings, and ChromaDB
- **Persistent vector store** for efficient document retrieval
- **Structured responses** with answers and source document excerpts
- **CORS enabled** for frontend integration
- **Environment variable configuration** for security

### Frontend (React + Vite)
- **Modern React interface** built with Vite and TypeScript
- **Tailwind CSS** for styling with shadcn/ui components
- **Real-time API integration** with axios
- **Source document display** with collapsible sections
- **Responsive design** for desktop and mobile

### QA System Improvements
- **Enhanced prompt engineering** for better answer accuracy
- **Daily vs possession limits** properly distinguished
- **Seasonal regulations** correctly identified
- **Location-specific rules** accurately extracted
- **Multiple license types** (Sport vs Conservation) handled

## Architecture

```
┌─────────────────┐    HTTP/JSON    ┌──────────────────┐
│   React Frontend│ ──────────────► │  FastAPI Backend │
│   (Port 8081)   │                 │   (Port 8000)    │
└─────────────────┘                 └──────────────────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │   ChromaDB       │
                                    │  Vector Store    │
                                    │  + OpenAI        │
                                    │  Embeddings      │
                                    └──────────────────┘
```

## Prerequisites

- **Python 3.8+**
- **Node.js 16+** and npm
- **OpenAI API key**
- **Fishing Regulations PDF** file (`mnr-2025-fishing-regulations-summary-en-2024-12-09_0.pdf`)

## Setup

### 1. Clone the repository
```bash
git clone <repository-url>
cd fishing_api
```

### 2. Backend Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Edit .env with your OpenAI API key: OPENAI_API_KEY=your_key_here
```

### 3. Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install
```

### 4. Add Regulations PDF
```bash
# Create data directory (if not exists)
mkdir -p data

# Download the Ontario 2025 Fishing Regulations PDF and save as:
# data/mnr-2025-fishing-regulations-summary-en-2024-12-09_0.pdf
```

## Running the Application

### Development Mode

**Terminal 1 - Backend:**
```bash
# From project root
python3 -m uvicorn main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
# From project root
cd frontend
npm run dev -- --port 8081
```

Access the application:
- **Frontend**: http://localhost:8081
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Production Build

**Frontend:**
```bash
cd frontend
npm run build
```

## API Endpoints

### POST /ask
Ask a question about Ontario's 2025 Fishing Regulations.

**Request:**
```json
{
  "question": "What are the bass limits on Lake Simcoe under a Sport Fishing Licence?"
}
```

**Response:**
```json
{
  "answer": "The Largemouth and Smallmouth Bass combined limits for Lake Simcoe under a Sport Fishing Licence are S-2 for the daily catch and retain limit and S-6 for the possession limit. The size limit is less than 35 cm from January 1 to June 30 and December 1 to December 31, and no size limit from July 1 to November 30.",
  "sources": [
    "•no more than 6 Largemouth and SmallmouthBass (combined) are held at any one time forfish caught under a Sport Fishing Licence.",
    "• Largemouth and Smallmouth Bass combined - S-2 and C-1; must be less than 35 cm from January 1 to June 30..."
  ],
  "error": null
}
```

### GET /health
Health check endpoint.

## Deployment

### Frontend (Vercel)
The frontend is configured for Vercel deployment with `vercel.json`.

**Environment Variables:**
- `VITE_API_URL`: Backend API URL (e.g., `https://your-backend.com`)

### Backend (Any Cloud Provider)
The backend can be deployed to Railway, Render, AWS, etc.

**Environment Variables:**
- `OPENAI_API_KEY`: Your OpenAI API key
- `ALLOWED_ORIGINS`: Comma-separated list of allowed origins for CORS

## Example Questions

Try asking questions like:
- "What are the bass limits on Lake Simcoe?"
- "When is the walleye season open in Zone 12?"
- "What are the size limits for northern pike?"
- "What's the difference between Sport and Conservation fishing licence limits?"
- "Can I keep bass in January?"

## Technical Details

### QA System
- **Embeddings**: OpenAI text-embedding-ada-002
- **Vector Store**: ChromaDB with similarity search
- **LLM**: GPT-3.5-turbo with custom prompts
- **Retrieval**: Top 5 most relevant document chunks
- **Response**: Structured JSON with answer and sources

### Frontend Stack
- **Framework**: React 18 with Vite
- **Language**: TypeScript
- **Styling**: Tailwind CSS + shadcn/ui
- **HTTP Client**: Axios
- **Build Tool**: Vite

### Backend Stack
- **Framework**: FastAPI
- **Vector DB**: ChromaDB
- **Embeddings**: LangChain + OpenAI
- **PDF Processing**: PyPDF2
- **CORS**: FastAPI CORS middleware

## Notes

- The PDF is embedded once during first startup and saved to `vector_db/`
- Vector store persists between restarts for faster startup
- Answers are based solely on the official Ontario 2025 Fishing Regulations
- The system distinguishes between daily limits and possession limits
- Source documents are provided with each answer for verification 