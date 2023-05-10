import cv2
import numpy as np
import os

def extedn_channel():
    dir_path='single'
    file_names = os.listdir(dir_path)

    if not os.path.exists('single_out'):
        os.makedirs('single_out', mode=0o777)
    out_path = "./single_out\\"

    for file in file_names:
        image_path = os.path.join(dir_path, file)
        img = cv2.imread(image_path, -1)
        img_bgr = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        img_bgr[:, :, 0] = img
        img_bgr[:, :, 1] = img
        img_bgr[:, :, 2] = img
        out_file=os.path.join(out_path,file)
        print(out_file)

        cv2.imwrite(out_file,img_bgr)


if __name__ == '__main__':
    extedn_channel()