from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Ontario Fishing Regulations QA API",
    description="API for answering questions about Ontario's 2025 Fishing Regulations",
    version="1.0.0"
)

# Configure CORS
origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    sources: List[str] = []
    error: Optional[str] = None

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    try:
        from utils.qa_chain import get_answer
        
        # Get answer using the QA chain
        result = get_answer(request.question)
        
        if "error" in result:
            return QuestionResponse(
                answer="I apologize, but I encountered an error processing your question. Please try again.",
                error=result["error"]
            )
        
        return QuestionResponse(
            answer=result["answer"],
            sources=result.get("sources", [])
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        return QuestionResponse(
            answer="I apologize, but I encountered an error processing your question. Please try again.",
            error=str(e)
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to the Ontario Fishing Regulations QA API",
        "docs": "/docs",
        "health": "/health"
    }

# For Vercel deployment
def handler(request):
    return app

# This block is for local development only
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 