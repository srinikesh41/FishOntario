from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    error: Optional[str] = None

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    try:
        # Initialize ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0
        )
        
        # Create a prompt that includes context about Ontario fishing regulations
        prompt = f"""You are an expert on Ontario's fishing regulations. 
        Answer the following question about fishing in Ontario:
        
        Question: {request.question}
        
        If you're not sure about the specific regulation, say so and recommend checking the official regulations."""
        
        # Get response from OpenAI
        response = llm.invoke(prompt)
        
        return QuestionResponse(answer=response.content)
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

# This block is for local development only
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 