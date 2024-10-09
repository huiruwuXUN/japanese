import os
import numpy as np
from PIL import Image

def preprocess_images(input_dir, output_file, img_size=(32, 32)):
    """
    Preprocess images from the input directory, resize them, and store them with corresponding labels.

    Parameters:
    input_dir (str): The directory containing subfolders for each writer, where each folder contains the images.
    output_file (str): The file path where the preprocessed data and labels will be saved as a .npy file.
    img_size (tuple): The size to which each image will be resized (default is 32x32).
    
    Returns:
    None: The function saves the processed data and labels as a .npy file.
    """
    data = []  # List to store the preprocessed images
    labels = []  # List to store the corresponding writer IDs (labels)

    # Iterate over each folder (representing a writer) in the input directory
    for writer_id in os.listdir(input_dir):
        writer_folder = os.path.join(input_dir, writer_id)
        if os.path.isdir(writer_folder):
            # Iterate over each image file in the writer's folder
            for image_file in os.listdir(writer_folder):
                if image_file.endswith(('.jpg', '.png', '.jpeg')):
                    image_path = os.path.join(writer_folder, image_file)
                    
                    # Open the image, convert to grayscale, and resize
                    image = Image.open(image_path).convert('L').resize(img_size)
                    
                    # Convert image to numpy array and store it
                    image_np = np.array(image)
                    data.append(image_np)
                    labels.append(writer_id)

    # Convert the lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Save the data and labels to a .npy file
    np.save(output_file, {"data": data, "labels": labels})

# Example usage:
# preprocess_images("E:/Data/JHA/CASIA_char_imgs/Gnt1.0TrainPart1", "E:/Data/JHA/CASIA_char_imgs_preprocessed.npy")
