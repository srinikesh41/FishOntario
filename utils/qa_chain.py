from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from utils.document_loader import get_relevant_documents

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

def get_answer(question: str) -> dict:
    """Get an answer to a question about fishing regulations"""
    try:
        # Get relevant documents
        relevant_docs = get_relevant_documents(question, k=5)
        
        # Combine the relevant documents into context
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create the prompt template
        PROMPT = PromptTemplate(
            template=QA_PROMPT,
            input_variables=["context", "question"]
        )
        
        # Create the chain
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0
        )
        
        chain = LLMChain(
            llm=llm,
            prompt=PROMPT
        )
        
        # Get the answer
        result = chain.invoke({
            "context": context,
            "question": question
        })
        
        return {
            "answer": result["text"],
            "sources": [doc.page_content for doc in relevant_docs]
        }
    except Exception as e:
        print(f"Error in QA chain: {str(e)}")
        return {
            "error": str(e)
        } 