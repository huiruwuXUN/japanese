import os
import cv2
import numpy as np

# Define the input and output folder paths
input_folder = "data/yongsong_notLean"  # Replace with your input folder path
output_folder = "data/yongsong_notLean_preprocess"  # Replace with your output folder path

# Ensure the output folder exists, create it if it doesn't
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define the threshold value
threshold_value = 128

# Define the convolution kernel
kernel = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]])

# Process each image in the folder
for filename in os.listdir(input_folder):
    # Load the image
    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)

    # Resize the image to 64x64
    resized_image = image

    # Apply thresholding
    _, thresholded_image = cv2.threshold(resized_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Apply convolution to the image
    convolved_image = cv2.filter2D(thresholded_image, -1, kernel)

    # Save the processed image
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, convolved_image)