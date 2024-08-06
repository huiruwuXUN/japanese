# This has all py code required for workflow in branch 1.
from transformers import AutoTokenizer, VisionEncoderDecoderModel
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.transforms import Resize, ToTensor
from tqdm import tqdm
from openpyxl import Workbook
import pandas as pd
import glob
import os
from google.cloud import vision_v1
from google.oauth2 import service_account
from tqdm import tqdm
import openpyxl
import shutil
# model from manga-ocr-base, utilize it as japanese OCR
tokenizer = AutoTokenizer.from_pretrained("kha-white/manga-ocr-base")
model = VisionEncoderDecoderModel.from_pretrained("kha-white/manga-ocr-base")

transform = torch.nn.Sequential(
    Resize((224, 224)),
    ToTensor()
)
def jap_ocr(image_path):
    image = Image.open(image_path)
    resize = Resize((224, 224))
    image = resize(image)
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    generated_ids = model.generate(image_tensor)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def concate_files(folder_path, merged_name='merged.csv'):
    # Loop through all files and read them into a dataframe, then append to one csv
    all_files = glob.glob(folder_path + "/*.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    # Concatenate all dataframes in the list into one dataframe
    frame = pd.concat(li, axis=0, ignore_index=True)

    # Optionally, you can save the merged dataframe to a new CSV file
    frame.to_csv(merged_name, index=False)
    return merged_name


def google_ocr(image_path, credentials_path):
    # Initialize the client
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    client = vision_v1.ImageAnnotatorClient(credentials=credentials)

    # Read the image
    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision_v1.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations  # Extract and store the detected text in a list
    if texts:
        texts = texts[-1].description.split('\n')
    return texts

def merged_ocr(image_folder_path, credentials_path, type_classifier = "merged.csv"):

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL Image to tensor
        # Add any additional transformations as needed
    ])
    label_map = pd.read_csv(type_classifier, index_col=0).to_dict()['Class']
    result = []
    for img_dir_name in tqdm(os.listdir(image_folder_path)):
        for filename in os.listdir(os.path.join(image_folder_path, img_dir_name)):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(image_folder_path, img_dir_name, filename)

                label = label_map.get(filename)
                if label == 'Kana':
                    text = jap_ocr(image_path)
                elif label == 'Kanji':
                    text = google_ocr(image_path, credentials_path) # this is poor ocr in traditional chinese...
                    if text == []:
                        text = jap_ocr(image_path)
                else:
                    text = "unknown label"
                result.append((img_dir_name, filename, text))

    return result
from openpyxl import Workbook

def save_to_excel(results, excel_path):
    workbook = Workbook()
    sheet = workbook.active
    for (img_dir_name, filename, text) in results:
        if text != []:
            combined_text = ' '.join(text) if isinstance(text, list) else text
            sheet.append((img_dir_name, filename, combined_text))
        else:
            sheet.append((img_dir_name, filename, ""))
    workbook.save(excel_path)
    workbook.close()


def rename_images_with_excel_mapping(excel_file, image_folder):# THIS RENAME YOUR IMAGES!!!
    # Read the Excel file
    mapping_df = pd.read_excel(excel_file)

    # Iterate through each row in the Excel file
    for index, row in mapping_df.iterrows():
        folder_name = row[0]
        current_name = row[1]
        new_name = row[2]

        # Get the full path of the folder containing images
        folder_path = os.path.join(image_folder, folder_name)

        # Check if the folder exists
        if os.path.exists(folder_path):
            # Traverse through files in the folder
            for file in os.listdir(folder_path):
                if file == current_name:
                    # Get the full path of the current image file
                    current_file_path = os.path.join(folder_path, file)

                    # Construct the new file name with an index and ensure it ends with '.jpg'
                    _, ext = os.path.splitext(file)
                    new_file_name = f"{new_name}__{current_name}"

                    # Construct the new file path with the replaced name
                    new_file_path = os.path.join(folder_path, new_file_name)

                    try:
                        # Rename the image file
                        os.rename(current_file_path, new_file_path)

                    except Exception as e:
                        print(f"Error: {e}. Skipping {current_name}")
                    break  # Break once the file is renamed to avoid unnecessary iterations
        else:
            print(f"Folder {folder_name} not found in the specified image folder.")

def cluster_common(df, commonkaha, commonkanji):
    # Create sets for commonkaha and commonkanji
    kaha_sets = {char: set() for char in commonkaha}
    kanji_sets = {char: set() for char in commonkanji}

    for i in range(len(df)):
        # df.iloc[i].ocr
        group = df.iloc[i].group
        first = df.iloc[i].ocr
        filename = df.iloc[i].filename
        if first in commonkaha:
            kaha_sets[first].add(group + "/" + filename)
        # Check if the first character is in commonkanji
        elif first in commonkanji:
            kanji_sets[first].add(group + "/" + filename)

    # sample output: kaha_set: [は: {'B/B_3_47.jpg', 'D/D_4_5.jpg'},...]
    return kaha_sets,kanji_sets

# model from manga-ocr-base, utilize it as japanese OCR
tokenizer = AutoTokenizer.from_pretrained("kha-white/manga-ocr-base")
model = VisionEncoderDecoderModel.from_pretrained("kha-white/manga-ocr-base") # for kana ocr

#OCR process:
# Specify the path to CSV files that claim which character is kanji which are kana
folder_path = "../../pilot data/classifypilot"
concate_files(folder_path)
credentials_path = '../../credentials.json' # for kanji ocr
image_folder_path = '../../pilot data/data/'  # dir/dir/image
type_classifier = "merged.csv"
excel_file = 'new_pilotdata_ocr1.xlsx'
result = merged_ocr(image_folder_path,credentials_path)
save_to_excel(result,excel_file)
image_folder = '../../pilot data/data'# Change this to target image folder
# rename_images_with_excel_mapping(excel_file,image_folder) # THIS RENAME YOUR IMAGES!!!

#Classifying process
image_folder_path = '../../pilot data/data/'  # dir/image inside
df = pd.read_excel(excel_file, engine='openpyxl')
commonkaha = ['は', 'か', 'へ', 'で', 'す', 'あ', 'お', 'の', 'に', 'を', 'る', 'く', 'し', 'な', 'よ', 'ス', 'ル']
commonkanji = ['日', '事', '人', '一', '見', '本', '子', '出', '年', '大', '言', '学', '分', '中', '記', '会', '新',
               '月', '時', '行', '本', '立', '気', '報', '思', '上', '語', '自', '者', '生', '文', '明', '情', '国',
               '朝', '用', '書', '私', '手', '間', '小', '合']
cluster_common
