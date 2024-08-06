from typing import Tuple, List

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
from google.cloud import vision_v1
from google.oauth2 import service_account
import cv2
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def jap_ocr(image_path: str) -> str:
    """
    Perform OCR on a Japanese image using a pre-trained model.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    str: Extracted text from the image.
    """
    image = Image.open(image_path)
    resize = Resize((224, 224))
    image = resize(image)
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    generated_ids = model.generate(image_tensor)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def concate_files(folder_path: str, merged_name: str = 'merged.csv') -> str:
    """
    Concatenate multiple CSV files in a folder into a single CSV file.

    Parameters:
    folder_path (str): Path to the folder containing CSV files.
    merged_name (str): Name of the merged CSV file. Default is 'merged.csv'.

    Returns:
    str: Path to the merged CSV file.
    """
    all_files = glob.glob(folder_path + "/*.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame.to_csv(merged_name, index=False)
    return merged_name


def google_ocr(image_path: str, credentials_path: str) -> list:
    """
    Perform OCR on an image using Google Cloud Vision API.

    Parameters:
    image_path (str): Path to the image file.
    credentials_path (str): Path to the Google Cloud service account credentials file.

    Returns:
    list: List of detected text lines from the image.
    """
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    client = vision_v1.ImageAnnotatorClient(credentials=credentials)

    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision_v1.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if texts:
        texts = texts[-1].description.split('\n')
    return texts


def merged_ocr(image_folder_path: str, credentials_path: str, type_classifier: str = "merged.csv") -> list:
    """
    Perform OCR on images in a folder, using different OCR methods based on image classification.

    Parameters:
    image_folder_path (str): Path to the folder containing images.
    credentials_path (str): Path to the Google Cloud service account credentials file.
    type_classifier (str): Path to the CSV file containing image classification information.

    Returns:
    list: List of tuples containing group, filename, and OCR text.
    """
    workbook = Workbook()
    sheet = workbook.active
    transform = transforms.Compose([
        transforms.ToTensor(),
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
                    text = google_ocr(image_path, credentials_path)
                    if text == []:
                        text = jap_ocr(image_path)
                else:
                    text = "unknown label"
                result.append((img_dir_name, filename, text))

    return result


def save_to_excel(results: list, excel_path: str) -> str:
    """
    Save OCR results to an Excel file.

    Parameters:
    results (list): List of tuples containing group, filename, and OCR text.
    excel_path (str): Path to the Excel file.

    Returns:
    str: Path to the saved Excel file.
    """
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["group", "filename", "ocr"])
    for (img_dir_name, filename, text) in results:
        if text != []:
            combined_text = ' '.join(text) if isinstance(text, list) else text
            sheet.append((img_dir_name, filename, combined_text))
        else:
            sheet.append((img_dir_name, filename, ""))
    workbook.save(excel_path)
    df = pd.read_excel(excel_path)
    df.rename(columns={df.columns[0]: 'group', df.columns[1]: 'filename', df.columns[2]: 'ocr'}, inplace=True)
    df.to_excel(excel_path, index=False)
    return excel_path


def cluster_common(df: pd.DataFrame, commonkaha: list, commonkanji: list) -> Tuple[dict, dict]:
    """
    Cluster images based on common Kana and Kanji characters.

    Parameters:
    df (pd.DataFrame): DataFrame containing OCR results.
    commonkaha (list): List of common Kana characters.
    commonkanji (list): List of common Kanji characters.

    Returns:
    Tuple[dict, dict]: Dictionaries containing sets of image names for each common character.
    """
    kaha_sets = {char: set() for char in commonkaha}
    kanji_sets = {char: set() for char in commonkanji}
    for i in range(len(df)):
        group = df.iloc[i].group
        first = df.iloc[i].ocr
        filename = df.iloc[i].filename
        if first in commonkaha:
            kaha_sets[first].add(group + "/" + filename)
        elif first in commonkanji:
            kanji_sets[first].add(group + "/" + filename)

    return kaha_sets, kanji_sets


def create_common_appear_group(kaha_sets: dict, kanji_sets: dict):
    """
    Create CSV files for each common character containing images where they appear.

    Parameters:
    kaha_sets (dict): Dictionary of sets containing image names for common Kana characters.
    kanji_sets (dict): Dictionary of sets containing image names for common Kanji characters.
    """
    directory = 'cluster'
    if not os.path.exists(directory):
        os.makedirs(directory)
    count_csv = 0
    for key, images in kaha_sets.items():
        df = pd.DataFrame(list(images), columns=['Image_name'])
        file_path = os.path.join(directory, f'{key}.csv')
        df.to_csv(file_path, index=False)
        count_csv += 1
    for key, images in kanji_sets.items():
        df = pd.DataFrame(list(images), columns=['Image_name'])
        file_path = os.path.join(directory, f'{key}.csv')
        df.to_csv(file_path, index=False)
        count_csv += 1
    print(f"{count_csv} CSV files have been created.")


def non_max_suppression(cornresult: np.ndarray, threshold: float = 0.01, K_size: int = 3) -> List[Tuple[int, int]]:
    """
    Perform non-maximum suppression on Harris corner detection results.

    Parameters:
    cornresult (np.ndarray): 2D array obtained from Harris corner detection.
    threshold (float): Threshold to determine if a local maximum is considered a corner.
    K_size (int): Size of the kernel used for non-maximum suppression.

    Returns:
    List[Tuple[int, int]]: List of tuples (row, column) indices of detected corners.
    """
    m, n = cornresult.shape
    supresult = []
    k_half = K_size // 2
    padded_result = np.pad(cornresult, pad_width=k_half, mode='constant', constant_values=0)
    threshold = threshold * np.max(cornresult)
    for r in range(m):
        for c in range(n):
            center_value = cornresult[r, c]
            if center_value == 0:
                continue
            local_patch = padded_result[r + k_half - k_half: r + k_half + k_half + 1,
                                        c + k_half - k_half: c + k_half + k_half + 1]
            if (r - k_half == 0 or r + k_half == m + k_half - 1
                    or c - k_half == 0 or c + k_half == n + k_half - 1):
                continue
            if center_value == np.max(local_patch) and center_value > threshold:
                supresult.append((r, c))

    return supresult


def pad_images_to_size(img: np.ndarray, target_size: Tuple[int, int] = (80, 80)) -> np.ndarray:
    """
    Pad or resize an image to a target size.

    Parameters:
    img (np.ndarray): Input image.
    target_size (Tuple[int, int]): Target size for the image.

    Returns:
    np.ndarray: Padded or resized image.
    """
    if img is not None:
        h, w = img.shape[:2]
        if h > target_size[1] or w > target_size[0]:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        elif h < target_size[1] or w < target_size[0]:
            top = (target_size[1] - h) // 2
            bottom = target_size[1] - h - top
            left = (target_size[0] - w) // 2
            right = target_size[0] - w - left
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return img


def dot_feature(image_folder_path: str, source_path: str) -> Tuple[List[list], List[int], int, int]:
    """
    Extract dot features from images.

    Parameters:
    image_folder_path (str): Path to the folder containing images.
    source_path (str): Path to the CSV file containing image names.

    Returns:
    Tuple[List[list], List[int], int, int]: Dot features, dot features length, average height, and average width.
    """
    df = pd.read_csv(source_path)
    sum_m = 0
    sum_n = 0
    for i in range(len(df)):
        image_path = os.path.join(image_folder_path, df.iloc[i]['Image_name'])
        Img = cv2.imread(image_path)
        m, n = Img.shape[0], Img.shape[1]
        sum_m += m
        sum_n += n
    sum_m = sum_m // len(df)
    sum_n = sum_n // len(df)

    feature_dots_len = []
    feature_dots = []
    plt.figure(figsize=(15, 5))
    for i in range(len(df)):
        image_path = os.path.join(image_folder_path, df.iloc[i]['Image_name'])
        Img = cv2.imread(image_path)
        Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
        Img = pad_images_to_size(Img, (round(sum_m), round(0.7 * sum_n)))
        gray_Img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
        in_build_harris = cv2.cornerHarris(gray_Img, 2, 3, 0.04)
        in_build_corner = non_max_suppression(in_build_harris)
        feature_dots.append(in_build_corner)
        feature_dots_len.append(len(in_build_corner))
    return feature_dots, feature_dots_len, sum_m, sum_n


def compute_corner_density(corners: list, image_width: int, image_height: int) -> float:
    """
    Compute the density of corners in an image.

    Parameters:
    corners (list): List of detected corners.
    image_width (int): Width of the image.
    image_height (int): Height of the image.

    Returns:
    float: Density of corners in the image.
    """
    num_corners = len(corners)
    area = image_width * image_height
    density = num_corners / area
    return density


def compute_corner_stats(corners: list) -> dict:
    """
    Compute statistical features of corners.

    Parameters:
    corners (list): List of detected corners.

    Returns:
    dict: Statistical features of corners.
    """
    x_coords = [coord[0] for coord in corners]
    y_coords = [coord[1] for coord in corners]

    stats = {
        'mean_x': np.mean(x_coords),
        'mean_y': np.mean(y_coords),
        'var_x': np.var(x_coords),
        'var_y': np.var(y_coords)
    }
    return stats


def compute_bounding_box_size(corners: list) -> Tuple[int, int]:
    """
    Compute the bounding box size of corners.

    Parameters:
    corners (list): List of detected corners.

    Returns:
    Tuple[int, int]: Width and height of the bounding box.
    """
    x_coords = [coord[0] for coord in corners]
    y_coords = [coord[1] for coord in corners]

    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    width = max_x - min_x
    height = max_y - min_y
    return width, height


def load_local_image(file_path: str) -> np.ndarray:
    """
    Load a local image.

    Parameters:
    file_path (str): Path to the image file.

    Returns:
    np.ndarray: Loaded image.
    """
    if os.path.exists(file_path):
        image = cv2.imread(file_path)
        return image
    else:
        print(f"{file_path} doesn't exist")
        return None


def preprocess_image(image: np.ndarray, margin: int = 10, fixed_size: Tuple[int, int] = (200, 200)) -> np.ndarray:
    """
    Preprocess an image by converting it to grayscale and applying binary thresholding.

    Parameters:
    image (np.ndarray): Input image.
    margin (int): Margin for preprocessing. Default is 10.
    fixed_size (Tuple[int, int]): Fixed size for the image. Default is (200, 200).

    Returns:
    np.ndarray: Preprocessed binary image.
    """
    if image is None:
        print("Nothing to preprocess")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    return binary


def extract_features(binary_image: np.ndarray) -> dict:
    """
    Extract features from a binary image.

    Parameters:
    binary_image (np.ndarray): Binary image.

    Returns:
    dict: Extracted features.
    """
    features = {
        'white_pixels_ratio': np.sum(binary_image > 128) / binary_image.size
    }
    return features


def process_image_set(image_set: list, dir_path: str) -> list:
    """
    Process a set of images and extract features.

    Parameters:
    image_set (list): List of image filenames.
    dir_path (str): Directory path containing the images.

    Returns:
    list: List of extracted features for each image.
    """
    features_list = []
    for file_name in image_set:
        file_path = os.path.join(dir_path, file_name)
        image = cv2.imread(file_path)
        if image is not None:
            preprocessed_image = preprocess_image(image)
            if preprocessed_image is not None:
                features = extract_features(preprocessed_image)
                if features is not None:
                    features_list.append(features)
            else:
                print(f"Preprocessing {file_name} failed")
        else:
            print(f"Loading {file_name} failed")
    return features_list


def color_feature(image_folder_path: str, source_path: str) -> list:
    """
    Extract color features from images.

    Parameters:
    image_folder_path (str): Path to the folder containing images.
    source_path (str): Path to the CSV file containing image names.

    Returns:
    list: List of tuples containing image names and extracted features.
    """
    feature_list = []
    df = pd.read_csv(source_path)
    for i in range(len(df)):
        image_path = os.path.join(image_folder_path, df.iloc[i]['Image_name'])
        image = load_local_image(image_path)
        if image is not None:
            preprocessed_image = preprocess_image(image)
            if preprocessed_image is not None:
                features = extract_features(preprocessed_image)
                if features is not None:
                    feature_list.append((df.iloc[i]['Image_name'], features))
    return feature_list


def feature_mixed(feature_dots: list, feature_dots_len: list, sum_m: int, sum_n: int, feature_list: list) -> list:
    """
    Combine various features extracted from images.

    Parameters:
    feature_dots (list): List of dot features.
    feature_dots_len (list): List of dot features length.
    sum_m (int): Average height of images.
    sum_n (int): Average width of images.
    feature_list (list): List of features extracted from images.

    Returns:
    list: Combined features list.
    """
    for index, (img, feature) in enumerate(feature_list):
        corners = feature_dots[index]
        dots_len = feature_dots_len[index]
        density = compute_corner_density(corners, sum_m, round(sum_n * 0.7))
        stats = compute_corner_stats(corners)
        s_width, s_height = compute_bounding_box_size(corners)
        feature['corner_num'] = dots_len
        feature['corner_density'] = density
        for k, v in stats.items():
            feature[k] = v
        feature['smallest_width'] = s_width
        feature['smallest_height'] = s_height
    return feature_list


def cluster_one_image(feature_list: list) -> dict:
    """
    Cluster images based on extracted features.

    Parameters:
    feature_list (list): List of features extracted from images.

    Returns:
    dict: Dictionary of cluster assignments.
    """
    num_cluster = 5
    os.environ['OMP_NUM_THREADS'] = '1'
    data = []
    for file_name, features in feature_list:
        entry = [
            features['white_pixels_ratio'],
            features['corner_num'],
            features['corner_density'],
            features['mean_x'],
            features['mean_y'],
            features['var_x'],
            features['var_y'],
            features['smallest_width'],
            features['smallest_height']
        ]
        data.append(entry)
    data = np.array(data)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=num_cluster)
    kmeans.fit(data_scaled)
    clusters = kmeans.labels_
    cluster_dict = defaultdict(list)
    for index, cluster_label in enumerate(clusters):
        image_name = feature_list[index][0]
        cluster_dict[cluster_label].append(image_name)
    return cluster_dict


def visualize_cluster(image_folder_path: str, cluster_dict: dict):
    """
    Visualize clusters of images.

    Parameters:
    image_folder_path (str): Path to the folder containing images.
    cluster_dict (dict): Dictionary of cluster assignments.
    """
    for cluster, images in cluster_dict.items():
        print(f"Cluster {cluster}: {images}")
        plt.figure(figsize=(3 * len(images), 3))
        for index, img_name in enumerate(images):
            image_path = os.path.join(image_folder_path, img_name)
            image = plt.imread(image_path)
            ax = plt.subplot(1, len(images), index + 1)
            ax.imshow(image)
            ax.set_title(img_name)
            ax.axis('off')
        plt.tight_layout()
        plt.show()


transform = torch.nn.Sequential(
    Resize((224, 224)),
    ToTensor()
)

tokenizer = AutoTokenizer.from_pretrained("kha-white/manga-ocr-base")
model = VisionEncoderDecoderModel.from_pretrained("kha-white/manga-ocr-base")

# OCR process
folder_path = "../../pilot data/classifypilot"
concate_files(folder_path)
credentials_path = '../../credentials.json'
image_folder_path = '../../pilot data/data/'
type_classifier = "merged.csv"
excel_file = 'new_pilotdata_ocr1.xlsx'
result = merged_ocr(image_folder_path, credentials_path)
save_to_excel(result, excel_file)

image_folder_path = '../../pilot data/data/'
df = pd.read_excel(excel_file, engine='openpyxl')
commonkaha = ['は', 'か', 'へ', 'で', 'す', 'あ', 'お', 'の', 'に', 'を', 'る', 'く', 'し', 'な', 'よ', 'ス', 'ル']
commonkanji = ['日', '事', '人', '一', '見', '本', '子', '出', '年', '大', '言', '学', '分', '中', '記', '会', '新',
               '月', '時', '行', '本', '立', '気', '報', '思', '上', '語', '自', '者', '生', '文', '明', '情', '国',
               '朝', '用', '書', '私', '手', '間', '小', '合']
kaha_sets, kanji_sets = cluster_common(df, commonkaha, commonkanji)
create_common_appear_group(kaha_sets, kanji_sets)

source_path = 'cluster/あ.csv'

# Feature extraction for single image
feature_dots, feature_dots_len, sum_m, sum_n = dot_feature(image_folder_path, source_path)
feature_list = color_feature(image_folder_path, source_path)
feature_list = feature_mixed(feature_dots, feature_dots_len, sum_m, sum_n, feature_list)
cluster_dict = cluster_one_image(feature_list)
visualize_cluster(image_folder_path, cluster_dict)
