import time
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


def get_answer(context, extra_text, question):
    """
    Get answer with retry logic and fallback to Mistral if Gemini fails.
    """
    # Use LangChain ChatPromptTemplate to construct the system and human messages
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an assistant. Use the following extracted text and user input "
            "to answer the question clearly, concisely, and accurately.\n"
            "Keep your response concise (maximum 3 sentences or a brief paragraph) unless the user explicitly asks for more detail.\n\n"
            "--- Extracted Text ---\n"
            "{context}\n\n"
            "--- Additional Input ---\n"
            "{extra_text}"
        )),
        ("human", "{question}"),
    ])

    # First try Gemini
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0, api_key=os.environ.get("GOOGLE_API_KEY"))
        chain = prompt | llm
        response = chain.invoke({
            "context": context,
            "extra_text": extra_text,
            "question": question
        })
        return response.content
    except Exception as e:
        print(f"Gemini failed: {str(e)}. Falling back to Mistral...")
        
    # If Gemini fails, fallback to Mistral
    try:
        mistral_api_key = os.environ.get("MISTRAL_API_KEY")
        if not mistral_api_key:
            raise ValueError("Mistral API key not found in environment.")
            
        mistral_llm = ChatMistralAI(model="mistral-large-latest", temperature=0, mistral_api_key=mistral_api_key)
        mistral_chain = prompt | mistral_llm
        mistral_response = mistral_chain.invoke({
            "context": context,
            "extra_text": extra_text,
            "question": question
        })
        return mistral_response.content
    except Exception as e:
        print(f"Mistral fallback also failed: {str(e)}")
        return generate_fallback_answer(context, question)


async def get_answer_stream(context, extra_text, question):
    """
    Get answer stream with retry logic and fallback to Mistral if Gemini fails.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an assistant. Use the following extracted text and user input "
            "to answer the question clearly, concisely, and accurately.\n"
            "Keep your response concise (maximum 3 sentences or a brief paragraph) unless the user explicitly asks for more detail.\n\n"
            "--- Extracted Text ---\n"
            "{context}\n\n"
            "--- Additional Input ---\n"
            "{extra_text}"
        )),
        ("human", "{question}"),
    ])

    # First try Gemini stream
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0, api_key=os.environ.get("GOOGLE_API_KEY"))
        chain = prompt | llm
        async for chunk in chain.astream({
            "context": context,
            "extra_text": extra_text,
            "question": question
        }):
            yield chunk.content
        return
    except Exception as e:
        print(f"Gemini streaming failed: {str(e)}. Falling back to Mistral streaming...")

    # If Gemini fails, fallback to Mistral stream
    try:
        mistral_api_key = os.environ.get("MISTRAL_API_KEY")
        if not mistral_api_key:
            raise ValueError("Mistral API key not found in environment.")
            
        mistral_llm = ChatMistralAI(model="mistral-large-latest", temperature=0, mistral_api_key=mistral_api_key)
        mistral_chain = prompt | mistral_llm
        async for chunk in mistral_chain.astream({
            "context": context,
            "extra_text": extra_text,
            "question": question
        }):
            yield chunk.content
        return
    except Exception as e:
        print(f"Mistral fallback streaming failed: {str(e)}")
        # Yield the static fallback response
        yield generate_fallback_answer(context, question)


def generate_fallback_answer(context, question):
    """
    Fallback answer generator when API is rate limited.
    Uses simple text matching from the context.
    """
    if not context or not context.strip():
        return "I apologize, but I cannot answer this question at the moment. The system has reached its API rate limit. Please try again later."
    
    # Simple fallback: extract relevant sentences from context
    sentences = [s.strip() for s in context.split('.') if s.strip()]
    
    if not sentences:
        return "Unable to generate an answer from the available context at this time due to rate limiting."
    
    # Return first 2 relevant sentences as fallback
    fallback_response = ". ".join(sentences[:2]) + "."
    
    return f"[Limited Response] {fallback_response}\n\nNote: The system is currently rate-limited. A more detailed answer will be available after a short wait."

