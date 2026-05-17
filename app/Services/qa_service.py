import google.generativeai as genai


def get_answer(context, extra_text, question):
    model = genai.GenerativeModel("gemini-flash-latest")

    prompt = f"""
You are an assistant. Use the following extracted text and user input to answer the question clearly and accurately.

--- Extracted Text ---
{context}

--- Additional Input ---
{extra_text}

--- Question ---
{question}
"""

    response = model.generate_content(prompt)
    return response.text
