import cv2 as cv
import os
from PIL import Image
from roboflow import Roboflow


def cut_img(path, file, json):
    img_path = os.path.join(path, file)
    img = cv.imread(img_path)
    saved_parent = 'D:\\8715_project\\japanese-handwriting-analysis\\pilot_seg'

    file = file.split('.')
    saved_folder = file[0]
    saved_path = os.path.join(saved_parent, saved_folder)

    # Create the directory if it does not exist
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    counter = 0
    for prediction in json['predictions']:
        # Calculate the coordinates of the ROI
        roi_x = int(prediction['x'] - prediction['width'] / 2)
        roi_y = int(prediction['y'] - prediction['height'] / 2)
        roi_width = int(prediction['width'])
        roi_height = int(prediction['height'])

        filename = os.path.join(saved_path, str(counter) + ".jpg")
        # Crop the ROI from the image
        roi = img[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
        #print("file name ", filename)
        cv.imwrite(filename, roi)
        counter += 1

def get_prediction_json(img):
    rf = Roboflow(api_key="KMZmYcucQKFOHB6wfLl7")
    project = rf.workspace().project("japan-croxp")
    model = project.version(5).model
    json=model.predict(img, confidence=20, overlap=30).json()
    return json
def file_lookup():
    path='D:\8715_project\japanese-handwriting-analysis\pilot'
    path_list=os.listdir(path)
    path_list.sort(key=lambda x:str(x[:-4]))

    for file in path_list:

        #print("file ",file)
        img_path=os.path.join(path,file)
        #print(img_path)
        # infer on a local image

        json_0=get_prediction_json(img_path)
        cut_img(path,file,json_0)



if __name__ == '__main__':

    file_lookup()
