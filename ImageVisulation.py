from PIL import Image
import pytesseract


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Load image
img = Image.open('5.jpg')

# Extract text
text = pytesseract.image_to_string(img)

print(text)
