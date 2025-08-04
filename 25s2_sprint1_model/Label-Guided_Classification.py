import os
import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ---------- Feature----------
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    stroke_density = np.count_nonzero(gray) / gray.size

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_length = sum(cv2.arcLength(c, True) for c in contours)

    gray_f = np.float32(gray)
    dst = cv2.cornerHarris(gray_f, 2, 3, 0.04)
    corner_count = np.sum(dst > 0.01 * dst.max())

    return [stroke_density, contour_length, corner_count]

# ---------- load data ----------
def load_data(base_dir, label_file):
    df = pd.read_csv(label_file)
    features, labels = [], []
    for _, row in df.iterrows():
        path = os.path.join(base_dir, row["filepath"])
        if not os.path.exists(path): continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        feat = extract_features(img)
        features.append(feat)
        labels.append(int(row["label"]))
    return np.array(features), np.array(labels)

# ---------- main ----------
if __name__ == "__main__":
    base_dir = r"C:\Users\Lenovo\Downloads\resized_to_100x100"
    label_100_file = "labels_100.csv"     
    label_full_file = "labels_full.csv"   

    # Step 1: use 100  picture to train
    X_train, y_train = load_data(base_dir, label_100_file)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Step 2: load all RC04844 picture to predict
    X_all, y_true = load_data(base_dir, label_full_file)
    y_pred = model.predict(X_all)

    # Step 3: output acc and report
    acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ Accuracy on full RC04844 dataset: {acc*100:.2f}%\n")
    print("📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Simple", "Complex"]))

# confusion matrix
cm = confusion_matrix(y_true, y_pred)
labels = ["Simple", "Complex"]


plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix on RC04844")
plt.tight_layout()
plt.savefig("confusion_matrix_RC04844.png", dpi=300)  
plt.show()