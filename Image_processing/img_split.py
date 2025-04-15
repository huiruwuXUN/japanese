import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# Set paths
input_dir = "./source"
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# List all image files in the input directory
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

# Loop over all images
for image_file in image_files:
    image_path = os.path.join(input_dir, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_file}")
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary inverse thresholding
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Morphological closing to reduce noise
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter valid character regions
    filtered_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area > 30:
            filtered_boxes.append((x, y, w, h))

    # Sort by column group (x // 50) then y
    filtered_boxes = sorted(filtered_boxes, key=lambda b: (b[0] // 50, b[1]))

    # Draw bounding boxes
    output_image = image.copy()
    for x, y, w, h in filtered_boxes:
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save annotated image
    base_name = os.path.splitext(image_file)[0]
    image_output_path = os.path.join(output_dir, f"{base_name}_detected.png")
    cv2.imwrite(image_output_path, output_image)

    # Create subfolder for character crops
    char_dir = os.path.join(output_dir, f"{base_name}_chars")
    os.makedirs(char_dir, exist_ok=True)

    # Save character crops
    for i, (x, y, w, h) in enumerate(filtered_boxes):
        char_img = image[y:y+h, x:x+w]
        char_path = os.path.join(char_dir, f"char_{i}.png")
        cv2.imwrite(char_path, char_img)

    print(f"Processed {image_file}, saved {len(filtered_boxes)} characters.")

# Optional: Display last processed result
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Example result from last image")
plt.show()
