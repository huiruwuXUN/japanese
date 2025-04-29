import os
import cv2
import numpy as np
import pandas as pd

IMG_SIZE = 64

def load_images_and_labels(labels_csv, img_size=IMG_SIZE):
    df = pd.read_csv(labels_csv)
    images, labels, paths = [], [], []
    for idx, row in df.iterrows():
        path, label = row['filename'], row['label']
        if not isinstance(label, str) or label.strip() == "": continue  # 跳过无标签项
        if not os.path.exists(path): continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype(np.float32) / 255.0
        images.append(img[..., None])  # 增加通道维
        labels.append(label)
        paths.append(path)
    return np.array(images), np.array(labels), np.array(paths)
import tensorflow as tf
from tensorflow.keras import layers, models

def build_mini_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=3):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

images, labels, paths = load_images_and_labels('labels.csv')
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
X_train, X_val, y_train, y_val = train_test_split(images, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

model = build_mini_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=len(le.classes_))
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
def batch_predict_and_save(model, le, img_size=IMG_SIZE, output_folder='classified_output'):
    for folder in os.listdir('.'):
        if folder.startswith('page_') and os.path.isdir(folder):
            for fname in os.listdir(folder):
                if not fname.endswith('.png'): continue
                img_path = os.path.join(folder, fname)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                img = cv2.resize(img, (img_size, img_size)).astype(np.float32)/255.0
                img = np.expand_dims(img, axis=(0,-1))  # (1, H, W, 1)
                pred_idx = np.argmax(model.predict(img))
                pred_label = le.inverse_transform([pred_idx])[0]
                if pred_label == "kanji":
                    cat = "kanji"
                else:
                    cat = "non_kanji"
                # 保留原page结构
                page_dir = folder
                dst_dir = os.path.join(output_folder, cat, page_dir)
                os.makedirs(dst_dir, exist_ok=True)
                dst_path = os.path.join(dst_dir, fname)
                if not os.path.abspath(img_path) == os.path.abspath(dst_path):
                    import shutil
                    shutil.copy2(img_path, dst_path)
    print("分类已完成，输出目录为 classified_output/")
if __name__ == '__main__':
    # 1. 训练
    images, labels, paths = load_images_and_labels('labels.csv')
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    X_train, X_val, y_train, y_val = train_test_split(images, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)
    model = build_mini_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=len(le.classes_))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    # 2. 批量预测+输出
    batch_predict_and_save(model, le)
