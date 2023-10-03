import cv2
import os
import numpy as np


def enhance_contrast_image(img_path, output_folder):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Unable to read image from {img_path}")
        return

    img_float = img.astype('float32')
    min_val = np.min(img_float)
    max_val = np.max(img_float)

    enhanced_img = 255.0 * (img_float - min_val) / (max_val - min_val)
    enhanced_img = np.clip(enhanced_img, 0, 255).astype('uint8')

    # Construct the output path and save the enhanced image
    output_path = os.path.join(output_folder, os.path.basename(img_path))
    cv2.imwrite(output_path, enhanced_img)


if __name__ == "__main__":
    root_folder = r"D:\8715_project\japanese-handwriting-analysis\pilot_seg"
    output_root_folder = r"D:\8715_project\japanese-handwriting-analysis\pilot_enhence"

    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            # Construct full image path
            input_img_path = os.path.join(subdir, file)

            # Construct the corresponding output folder
            relative_subdir = os.path.relpath(subdir, root_folder)
            output_subfolder = os.path.join(output_root_folder, relative_subdir)
            os.makedirs(output_subfolder, exist_ok=True)  # create the output subfolder if it doesn't exist

            # Enhance and save the image
            enhance_contrast_image(input_img_path, output_subfolder)
