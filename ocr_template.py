import os
from google.cloud import vision
from openpyxl import Workbook

client = vision.ImageAnnotatorClient.from_service_account_json('path/to/your/service-account-file.json')

# create excel file
wb = Workbook()
ws = wb.active
ws.append(["Folder Name", "File Name", "OCR Result"])


folder_name = "example_folder"
with open(f"{folder_name}/{file_name}", "rb") as image_file:
    content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
                texts = response.text_annotations

                if texts:
                    ocr_result = texts[0].description
                else:
                    ocr_result = "No text detected"


def write_excel(result):
    for file_name in os.listdir(folder_name):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        #OCR


            ws.append([folder_name, file_name, ocr_result])


wb.save("OCR_Results.xlsx")

print(f"OCR results for all images in {folder_name} have been saved to OCR_Results.xlsx")
