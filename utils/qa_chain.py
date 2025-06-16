from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from utils.document_loader import get_relevant_documents

# Define the prompt template
QA_PROMPT = """You are an expert assistant for answering questions about Ontario's 2025 Fishing Regulations.

Use the information from the provided context to answer the question. When you find relevant information in the context, provide a clear and specific answer based on that information.

Guidelines:
1. If you find specific information that answers the question, provide that answer
2. If there are multiple regulations for different areas or conditions, explain the differences
3. When mentioning limits, always clarify what S-X and C-X mean (Sport licence daily limit and Conservation licence daily limit)
4. Pay attention to both daily limits AND possession limits - they may be different
5. If the context contains conflicting information, present the most specific or relevant information
6. Only say you don't have enough information if the context truly contains no relevant information about the topic

For catch limits:
- S-X means Sport Fishing Licence daily catch and retain limit
- C-X means Conservation Fishing Licence daily catch and retain limit
- Always specify both when available
- Include size limits and seasonal restrictions if mentioned
- If multiple regulations exist for different areas, specify which applies where
- Look for possession limits which may be different from daily limits

Context:
{context}

Question: {question}

Answer (be specific and cite the relevant regulation, including both daily and possession limits if different):"""

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