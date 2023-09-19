# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os

def print_hi():
    # Use a breakpoint in the code line below to debug your script.
    parent_path='D:\8715_project\japanese-handwriting-analysis\seg_letter'
    img_foler='D:\8715_project\japanese-handwriting-analysis\high_res_png'
    img_path=os.listdir(img_foler)
    for file in img_path:

        folder_name=file.split('.')[0]
        #print(folder_name)
        dir_name=os.path.join(parent_path,folder_name)
        os.makedirs(dir_name)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
