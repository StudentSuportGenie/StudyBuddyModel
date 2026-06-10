import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


def get_answer(context, extra_text, question):
    """
    Get answer with retry logic and rate limit handling.
    Retries up to 3 times with exponential backoff.
    """
    max_retries = 3
    retry_delay = 2  # Start with 2 seconds
    
    for attempt in range(max_retries):
        try:
            # Initialize LangChain's ChatGoogleGenerativeAI component using gemini-1.5-flash
            llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

            # Use LangChain ChatPromptTemplate to construct the system and human messages
            prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are an assistant. Use the following extracted text and user input "
                    "to answer the question clearly and accurately.\n\n"
                    "--- Extracted Text ---\n"
                    "{context}\n\n"
                    "--- Additional Input ---\n"
                    "{extra_text}"
                )),
                ("human", "{question}"),
            ])

            # Chain definition
            chain = prompt | llm

            # Invoke the chain
            response = chain.invoke({
                "context": context,
                "extra_text": extra_text,
                "question": question
            })

            return response.content
            
        except Exception as e:
            error_str = str(e)
            
            # Check if it's a rate limit error (429)
            if "429" in error_str or "quota" in error_str.lower():
                if attempt < max_retries - 1:
                    print(f"Rate limit hit (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    # Last attempt failed - return graceful fallback
                    print("Rate limit exceeded. Returning fallback response.")
                    return generate_fallback_answer(context, question)
            else:
                # Non-rate-limit error, raise it
                raise


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
