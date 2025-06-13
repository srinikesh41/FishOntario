import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
from utils.document_loader import load_and_embed_document
from utils.qa_chain import get_answer
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Get allowed origins from environment variable or use default
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "https://fishontario-frontend.vercel.app,http://localhost:3000").split(",")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        load_and_embed_document()
    except Exception as e:
        print(f"Error during startup: {str(e)}")
    yield
    # Shutdown
    pass

app = FastAPI(
    title="Ontario Fishing Regulations QA API",
    description="API for answering questions about Ontario's 2025 Fishing Regulations",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware with environment-based configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question about Ontario fishing regulations")

# Response model
class QuestionResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = None
    error: Optional[str] = None

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    if not request.question.strip():
        return QuestionResponse(answer="", error="Question cannot be empty.")
    
    try:
        result = get_answer(request.question)
        if "error" in result:
            return QuestionResponse(answer="", error=result["error"])
        return QuestionResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        # Log the error (in production, use a proper logging system)
        print(f"Error processing question: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="An error occurred while processing your question. Please try again later."
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "Welcome to the Ontario Fishing Regulations API"}

# Only run the server when running locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 