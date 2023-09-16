import os
from google.cloud import vision_v1
from google.oauth2 import service_account

# Replace with your own Google Cloud credentials JSON file path
credentials_path = 'path/to/your/credentials.json'

# Initialize the client
credentials = service_account.Credentials.from_service_account_file(credentials_path)
client = vision_v1.ImageAnnotatorClient(credentials=credentials)

# Replace with the path to your image file
image_path = 'ocr_test_1.jpg'

# Read the image
with open(image_path, 'rb') as image_file:
    content = image_file.read()

# Perform OCR on the image
image = vision_v1.Image(content=content)
response = client.text_detection(image=image)

# Extract and print the detected text
texts = response.text_annotations
if texts:
    detected_text = texts[0].description
    print("Detected text:")
    print(detected_text)
else:
    print("No text detected in the image.")
