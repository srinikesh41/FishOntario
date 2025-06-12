# Ontario Fishing Regulations QA API

A FastAPI backend that allows users to ask natural language questions about Ontario's 2025 Fishing Regulations. The API loads and embeds a PDF document of the regulations and uses LangChain with OpenAI and ChromaDB to provide accurate answers.

## Features

- Natural language Q&A about Ontario's 2025 Fishing Regulations
- Persistent vector store (ChromaDB) for efficient document retrieval
- RESTful API endpoint for asking questions
- Automatic embedding of the regulations PDF at startup
- Environment variable configuration for security

## Prerequisites

- Python 3.8+
- OpenAI API key
- Fishing Regulations PDF file (`mnr-2025-fishing-regulations-summary-en-2024-12-09_0.pdf`)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd fishing_api
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your OpenAI API key:
```bash
cp .env.example .env
# Edit .env with your actual API key
```

5. Download the Ontario 2025 Fishing Regulations PDF and place it in the `data` directory:
```bash
mkdir -p data
# Download the PDF manually or use wget/curl and save as:
# data/mnr-2025-fishing-regulations-summary-en-2024-12-09_0.pdf
```

## Running the API

Start the API server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

## Running with Docker

Build and run the containerized API:
```bash
docker build -t fishing-api .
docker run -p 8000:8000 --env-file .env -v $(pwd)/vector_db:/app/vector_db -v $(pwd)/data:/app/data fishing-api
```

## API Endpoints

### POST /ask

Ask a question about Ontario's 2025 Fishing Regulations.

**Request Body:**
```json
{
  "question": "What are the walleye catch limits for Zone 12?"
}
```

**Response:**
```json
{
  "answer": "In Zone 12, for walleye (and sauger), the sport fishing license limit is 4 (S) and the conservation license limit is 2 (C).",
  "error": null
}
```

### GET /health

Check if the API is running.

**Response:**
```json
{
  "status": "healthy"
}
```

## Optional: Frontend & Deployment

- You can connect this API to a frontend (e.g., React, Next.js) and deploy the frontend via Vercel.
- The backend can be deployed to any cloud provider or container platform.

## Notes

- The PDF is embedded only once during the first startup and saved to a persistent vector store (`vector_db`).
- The API requires both the PDF document and an OpenAI API key to function properly.
- Answers are based solely on the contents of the PDF document.
- See `.env.example` for required environment variables. 