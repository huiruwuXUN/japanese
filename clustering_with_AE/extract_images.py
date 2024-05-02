import pandas as pd
import os

def image_extraction_from_file(class_dir, images_dir, character_type):
    """
    Extract the file paths of images corresponding to a specific character type from a directory specified by a CSV file.
    
    This function reads a CSV file where each row contains information about images, specifically a class identifier and 
    an image file name. It filters these images by the specified character type and constructs a list of full file paths
    to these images, which are stored in a specified directory.
    
    Parameters:
    - class_dir (str): The file path to the CSV file containing the class and image file name information.
    - images_dir (str): The directory path where the images are stored.
    - character_type (str): The character class/type to filter the images by.

    Returns:
    - list: A list of strings where each string is the full file path to an image of the specified character type.
    
    Example:
    >>> image_paths = image_extraction_from_file("data/classes.csv", "/images", "TypeA")
    >>> print(image_paths)
    ['/images/img001.jpg', '/images/img002.jpg', ...]
    """
    images_dir_list = []
    # Load data from CSV file
    data = pd.read_csv(class_dir, sep=',')
    # Iterate through DataFrame rows
    for index, row in data.iterrows():
        if row['Class'] == character_type:
            # Append the full path of the image to the list
            images_dir_list.append(os.path.join(images_dir, row['Image_name']))
    return images_dir_list

if __name__ == '__main__':
    os.chdir('character_classifying_cnn\outputs')
    idl = image_extraction_from_file('class_B.csv', 'images\B', 'Kanji')
    print(idl)