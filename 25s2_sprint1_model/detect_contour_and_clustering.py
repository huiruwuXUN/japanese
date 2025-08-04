import os
import re
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


folder_path = r"M:\cv_mini_project\pdfs\segmented_chars_img\RC05128\page_1"

def compute_contour_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"unable to read image:{image_path}")
    binary = 255 - img
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    total_length = sum(cv2.arcLength(c, True) for c in contours)
    return [num_contours, total_length]
def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def cluster_folder(folder_path):
    features = []
    filenames = []

    for fname in sorted(os.listdir(folder_path), key=natural_key):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            fpath = os.path.join(folder_path, fname)
            try:
                feat = compute_contour_features(fpath)
                features.append(feat)
                filenames.append(fname)
            except Exception as e:
                print(f"skip{fname}:{e}")

    if not features:
        print("unable to find valid img")
        return

    features = np.array(features)
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(features)

    df = pd.DataFrame({'filename': filenames, 'cluster': labels})
    out_path = os.path.join(folder_path, 'contour_complexity.csv')
    df.to_csv(out_path, index=False)
    print(f"output file:{out_path}")


cluster_folder(folder_path)

