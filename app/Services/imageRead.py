# import cv2
# import numpy as np
# import easyocr  # Replaced PaddleOCR with EasyOCR
# import requests
# import base64
# import os


# #  SET IMAGE PATH HERE

# IMAGE_PATH = r"D:\Personal Projects\StudyBuddyModel\Test image\Test_01.png"


# # Preprocess Image (OpenCV)

# def preprocess_image(image_path):
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"❌ Image not found: {image_path}")

#     img = cv2.imread(image_path)

#     if img is None:
#         raise ValueError("❌ Failed to load image. Check file format or path.")

#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Noise removal
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Adaptive threshold
#     thresh = cv2.adaptiveThreshold(
#         blur, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         11, 2
#     )

#     return thresh


# # Text OCR (EasyOCR)

# reader = easyocr.Reader(['en'])  # Initialize EasyOCR

# def extract_text(image):
#     results = reader.readtext(image)
#     # Extract text only
#     text = " ".join([res[1] for res in results])
#     return text


# # Formula OCR (Mathpix API)

# def extract_formula_mathpix(image_path):
#     APP_ID = "YOUR_APP_ID"
#     APP_KEY = "YOUR_APP_KEY"

#     if APP_ID == "YOUR_APP_ID":
#         return "⚠️ No formula (Mathpix API not configured)"

#     with open(image_path, "rb") as img_file:
#         image_data = base64.b64encode(img_file.read()).decode()

#     try:
#         response = requests.post(
#             "https://api.mathpix.com/v3/text",
#             json={
#                 "src": f"data:image/png;base64,{image_data}",
#                 "formats": ["latex_styled"],
#             },
#             headers={
#                 "app_id": APP_ID,
#                 "app_key": APP_KEY,
#                 "Content-type": "application/json",
#             },
#             timeout=10
#         )

#         result = response.json()
#         return result.get("latex_styled", "No formula detected")

#     except Exception as e:
#         return f"Mathpix error: {str(e)}"


# #  Merge Results

# def merge_results(text, formula):
#     return f"""
# ===== EXTRACTED TEXT =====
# {text}

# ===== EXTRACTED FORMULA (LaTeX) =====
# {formula}
# """


# #  Run Pipeline

# def process_image():
#     print("🔄 Preprocessing...")
#     processed_img = preprocess_image(IMAGE_PATH)

#     print("🔄 Extracting text...")
#     text = extract_text(processed_img)

#     print("🔄 Extracting formulas...")
#     formula = extract_formula_mathpix(IMAGE_PATH)

#     print("🔄 Merging results...")
#     final = merge_results(text, formula)

#     return final


# #  RUN

# if __name__ == "__main__":
#     try:
#         result = process_image()
#         print(result)

#     except Exception as e:
#         print(f"\n🔥 ERROR: {e}")