from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from utils.document_loader import get_vector_store
import os

# Define the prompt template
QA_PROMPT = """You are an expert assistant for answering questions about Ontario's 2025 Fishing Regulations.

Use only the information from the provided context to answer the question. If the information is not in the context, 
say "I don't have enough information about that in the Ontario 2025 Fishing Regulations." Do not make up information.

When answering questions about specific locations or time periods:
1. First check if the location/zone is mentioned in the context
2. Then check if there are any time-specific regulations (seasons, dates)
3. Finally check for any special conditions or exceptions

For catch limits:
- Always specify both sport and conservation license limits
- Include size limits if mentioned
- Note any special conditions or exceptions
- If a location is mentioned, only provide limits specific to that location

Context:
{context}

Question: {question}

Answer (be concise and specific):"""

def get_qa_chain():
    """Create a QA chain using the vector store"""
    # Get the vector store
    vector_store = get_vector_store()
    
    # Create a retriever from the vector store
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Create the prompt template
    PROMPT = PromptTemplate(
        template=QA_PROMPT,
        input_variables=["context", "question"]
    )
    
    # Create the chain
    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0
        ),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": PROMPT,
        },
        return_source_documents=True
    )
    
    return chain

def get_answer(question: str) -> dict:
    """Get an answer to a question about fishing regulations"""
    try:
        # Get the QA chain
        chain = get_qa_chain()
        
        # Get the answer
        result = chain.invoke({"query": question})
        
        # Extract the answer and sources
        answer = result["result"]
        sources = [doc.page_content for doc in result["source_documents"]]
        
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        print(f"Error in QA chain: {str(e)}")
        return {
            "error": str(e)
        } 