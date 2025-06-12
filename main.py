import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from utils.document_loader import load_and_embed_document
from utils.qa_chain import get_answer_for_query
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
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
    error: Optional[str] = None

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    if not request.question.strip():
        return QuestionResponse(answer="", error="Question cannot be empty.")
    
    try:
        answer = get_answer_for_query(request.question)
        return QuestionResponse(answer=answer)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 