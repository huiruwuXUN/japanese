import cv2
import os

def binarization_thresholding(stand_img):
    ret, thresh = cv2.threshold(stand_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return thresh

def process_images(root_folder, output_folder):
    #
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                #
                full_path = os.path.join(subdir, file)
                print("Processing:", full_path)

                #
                img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                bin_img = binarization_thresholding(img)

                #
                relative_subdir = os.path.relpath(subdir, root_folder)
                output_subdir = os.path.join(output_folder, relative_subdir)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                #
                output_path = os.path.join(output_subdir, file)
                cv2.imwrite(output_path, bin_img)

if __name__ == "__main__":
    root_folder = "D:\\8715_project\\japanese-handwriting-analysis\\seg_letter"
    output_folder = "D:\\8715_project\\japanese-handwriting-analysis\\bin_seg"
    process_images(root_folder, output_folder)
