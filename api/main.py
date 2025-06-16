from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
import os
from openai import OpenAI

# Initialize FastAPI app
app = FastAPI(
    title="Ontario Fishing Regulations QA API",
    description="API for answering questions about Ontario's 2025 Fishing Regulations",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Create a comprehensive prompt with fishing regulations context
        prompt = f"""You are an expert on Ontario's 2025 Fishing Regulations. Answer the following question about fishing in Ontario based on the official regulations.

For bass fishing specifically:
- Largemouth and Smallmouth Bass combined limits: Sport Fishing Licence allows up to 6 fish possession limit, with daily limits varying by season and location
- Lake Simcoe: S-2 and C-1 daily limits, with size restrictions (less than 35 cm from Jan 1-June 30 and Dec 1-31)
- General seasons: Early season catch and release from January 1 to May 10, regular seasons vary by species and location

For other species, provide accurate information about:
- Daily catch limits (S-X for Sport, C-X for Conservation licenses)
- Possession limits
- Size restrictions
- Seasonal closures
- Location-specific rules

Question: {request.question}

Provide a detailed, accurate answer based on Ontario's 2025 Fishing Regulations. If you're not completely certain about specific details, recommend checking the official regulations."""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        return QuestionResponse(
            answer=response.choices[0].message.content,
            sources=["Ontario 2025 Fishing Regulations Summary"]
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        return QuestionResponse(
            answer="I apologize, but I encountered an error processing your question. Please try again.",
            error=str(e)
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Ontario Fishing Regulations QA API",
        "docs": "/docs",
        "health": "/health"
    } 