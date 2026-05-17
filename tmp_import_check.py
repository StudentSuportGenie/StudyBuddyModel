import os
import inspect

os.environ['GEMINI_API'] = 'DUMMY'

print('start imported modules')
import app.main
import app.routes.api as api
import app.Services.qa_service as qa_service
import app.Services.pdf_service as pdf_service
import app.Services.imageRead as imageRead
import app.Services.vector_service as vector_service
import google.generativeai as genai

print('app imported')
print('qa', inspect.signature(qa_service.get_answer))
print('pdf', inspect.signature(pdf_service.extract_text_from_pdf_url))
print('img', inspect.signature(imageRead.extract_text_from_image_url))
print('store', inspect.signature(vector_service.store_text_to_chroma))
print('load', inspect.signature(vector_service.load_user_vector_db))
model = genai.GenerativeModel('gemini-2.0-flash')
print('generate_content', inspect.signature(model.generate_content))
