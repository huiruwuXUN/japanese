import cv2 as cv
import os
from PIL import Image
from roboflow import Roboflow

# This module is used to cut regions of interest (ROIs) from an image based on provided predictions
# Usage of this function requires the Roboflow API key

def cut_image(image_path, file_name, output_directory, predictions):
    """
    Cuts regions of interest (ROIs) from an image based on provided predictions
    and saves them into a specified directory.

    Parameters:
    image_path (str): Path to the directory containing the image.
    file_name (str): Name of the image file to be processed.
    output_directory (str): Path to the directory where the cropped images will be saved.
    predictions (list): A list of dictionaries containing prediction details,
                        each with keys 'x', 'y', 'width', and 'height'.

    Example:
    predictions = [
        {'x': 50, 'y': 50, 'width': 100, 'height': 200},
        {'x': 150, 'y': 150, 'width': 80, 'height': 120}
    ]
    """

    # Construct the full image path
    full_image_path = os.path.join(image_path, file_name)

    # Load the image using OpenCV
    image = cv.imread(full_image_path)
    if image is None:
        raise ValueError(f"Image not found at {full_image_path}")

    # Extract the base file name (without extension) for creating output folder
    base_file_name = os.path.splitext(file_name)[0]
    save_path = os.path.join(output_directory, base_file_name)

    # Create the output directory if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Iterate over the predictions and crop the image accordingly
    for index, prediction in enumerate(predictions):
        # Calculate the coordinates and size of the ROI
        roi_x = int(prediction['x'] - prediction['width'] / 2)
        roi_y = int(prediction['y'] - prediction['height'] / 2)
        roi_width = int(prediction['width'])
        roi_height = int(prediction['height'])

        # Define the file name for the cropped image
        roi_file_name = os.path.join(save_path, f"crop_{index}.jpg")

        # Crop the ROI from the image
        roi = image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        # Save the cropped image to the specified path
        cv.imwrite(roi_file_name, roi)


def get_prediction_json(image_path, api_key, project_name="japan-croxp", version_number=5, confidence=20, overlap=30):
    """
    Retrieves prediction results in JSON format for the provided image using a specified Roboflow model.

    Parameters:
    image_path (str): Path to the image file to be used for prediction.
    api_key (str): API key for accessing the Roboflow service.
    project_name (str): The name of the project in Roboflow. Default is 'japan-croxp'.
    version_number (int): The version number of the model to use. Default is 5.
    confidence (float): Minimum confidence level for predictions. Default is 20.
    overlap (float): Maximum allowable overlap for predictions. Default is 30.

    Returns:
    dict: A JSON object containing the prediction results.
    """

    # Initialize the Roboflow object with the provided API key
    rf = Roboflow(api_key)

    # Access the specified project within the workspace
    project = rf.workspace().project(project_name)

    # Get the specified version of the model
    model = project.version(version_number).model

    # Run the prediction on the provided image and get the results in JSON format
    prediction_json = model.predict(image_path, confidence=confidence, overlap=overlap).json()

    return prediction_json


def file_lookup(directory_path, api_key, project_name="japan-croxp", version_number=5, confidence=20, overlap=30):
    """
    Looks up files in a specified directory, processes each image file, and performs
    image prediction and cropping operations based on the predictions.

    Parameters:
    directory_path (str): The directory path where image files are located.
    api_key (str): API key for accessing the Roboflow service.
    project_name (str): The name of the project in Roboflow. Default is 'japan-croxp'.
    version_number (int): The version number of the model to use. Default is 5.
    confidence (float): Minimum confidence level for predictions. Default is 20.
    overlap (float): Maximum allowable overlap for predictions. Default is 30.

    Returns:
    None
    """

    # Get the list of files in the specified directory
    file_list = os.listdir(directory_path)
    file_list.sort(key=lambda x: str(x[:-4]))  # Sort the files based on their names without extensions

    for file_name in file_list:
        # Construct the full path for the image file
        image_path = os.path.join(directory_path, file_name)

        # Perform prediction on the image
        prediction_json = get_prediction_json(image_path, api_key, project_name, version_number, confidence, overlap)

        # Perform image cropping based on the prediction results
        cut_image(directory_path, file_name, directory_path, prediction_json)


if __name__ == '__main__':

    file_lookup()
