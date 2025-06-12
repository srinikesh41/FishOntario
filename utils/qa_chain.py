from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from utils.document_loader import get_vector_store

# Define the prompt template
QA_PROMPT = """You are an expert assistant for answering questions about Ontario's 2025 Fishing Regulations.

Use only the information from the provided context to answer the question. If the information is not in the context, 
say "I don't have enough information about that in the Ontario 2025 Fishing Regulations." Do not make up information.

When relevant, always provide both sport and conservation license regulations and limits in your answer.

Context:
{context}

Question: {question}

Answer (be concise, clear and helpful):"""

def get_qa_chain():
    """Create a QA chain for answering questions about fishing regulations"""
    # Get the vector store
    vector_store = get_vector_store()
    
    # Create a retriever from the vector store
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Fetch 5 most relevant documents
    )
    
    # Create the prompt template
    prompt = PromptTemplate(
        template=QA_PROMPT,
        input_variables=["context", "question"]
    )
    
    # Create the LLM
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,  # No creativity needed, just facts
    )
    
    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Simple approach: just stuff all documents into the prompt
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

def get_answer_for_query(question: str) -> str:
    """Get an answer for a question about Ontario fishing regulations"""
    qa_chain = get_qa_chain()
    
    try:
        result = qa_chain.invoke(question)
        return result['result']
    except Exception as e:
        # Log the error for debugging
        print(f"Error in QA chain: {str(e)}")
        # Return a user-friendly message
        return "I'm sorry, I couldn't process your question. Please try asking in a different way or check back later." 