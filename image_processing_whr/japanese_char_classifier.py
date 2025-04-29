import os
import cv2
import numpy as np
import pandas as pd
import shutil
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

LABEL_CSV = "labels.csv"              # 主标签表
OUTPUT_FOLDER = "classified_output_char"   # 输出总文件夹
IMAGE_SIZE = 64
TRAIN_SIZE = 0.8

def extract_hog(img_path):
    image = cv2.imread(img_path, 
                       cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, 
                       (IMAGE_SIZE, IMAGE_SIZE))
    _, image = cv2.threshold(image, 0, 255, 
                             cv2.THRESH_BINARY_INV + 
                             cv2.THRESH_OTSU)
    features = hog(image, 
                   pixels_per_cell=(8, 8), 
                   cells_per_block=(2, 2))
    return features

def load_labeled_data(label_csv):
    label_df = pd.read_csv(label_csv)
    X, y = [], []
    for _, row in label_df.iterrows():
        img_path = row['filename']
        if os.path.exists(img_path):
            feat = extract_hog(img_path)
            X.append(feat)
            y.append(row['label'])
        else:
            print(f"Warning: {img_path} not found, skipping.")
    return np.array(X), np.array(y)

def train_svm_classifier(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, train_size=TRAIN_SIZE, random_state=42)
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    return clf, le

def find_all_png_files_page_dirs(root_dir="."):
    all_imgs = []
    for folder in os.listdir(root_dir):
        if folder.startswith("page_") and os.path.isdir(folder):
            page_folder = os.path.join(root_dir, folder)
            for fname in os.listdir(page_folder):
                if fname.endswith(".png"):
                    all_imgs.append(os.path.join(folder, fname))
    return all_imgs

def predict_all_images_kanji_only(img_paths, clf, le):
    results = []
    for img_path in img_paths:
        if os.path.exists(img_path):
            feat = extract_hog(img_path)
            pred = clf.predict([feat])[0]
            label = le.inverse_transform([pred])[0]
            out_label = "kanji" if label == "kanji" else "non_kanji"
            results.append((img_path, out_label))
    return results

def move_images_by_prediction(results, output_root="."):
    for img_path, label in results:
        src_path = img_path
        page_dir = os.path.dirname(img_path)
        dst_folder = os.path.join(output_root, label, page_dir)
        os.makedirs(dst_folder, exist_ok=True)
        fname = os.path.basename(img_path)
        dst_path = os.path.join(dst_folder, fname)
        if not os.path.abspath(src_path) == os.path.abspath(dst_path):
            shutil.copy2(src_path, dst_path)
    print(f"✅ 已按类别+page子目录输出到 {output_root}/kanji/page_x 和 {output_root}/non_kanji/page_x 等目录。")

def main():
    print("=== 加载训练数据 ===")
    X, y = load_labeled_data(LABEL_CSV)
    print(f"加载标注样本数: {len(X)}")
    print("=== 训练SVM分类器 ===")
    clf, le = train_svm_classifier(X, y)
    print("=== 批量分类所有图片（只区分汉字/非汉字，且只遍历page_文件夹）===")
    all_img_paths = find_all_png_files_page_dirs(".")
    results = predict_all_images_kanji_only(all_img_paths, clf, le)
    # 输出csv
    outcsv = os.path.join(OUTPUT_FOLDER, "predicted_labels_kanji_only.csv")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    pd.DataFrame(results, columns=["filename", "predicted_label"]).to_csv(outcsv, index=False)
    print(f"✅ 分类结果已输出到 {outcsv}")
    # 按类别+page子文件夹存放
    move_images_by_prediction(results, OUTPUT_FOLDER)

if __name__ == "__main__":
    main()
