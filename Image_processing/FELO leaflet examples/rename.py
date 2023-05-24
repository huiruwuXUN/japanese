import os
import glob
folder_dir = '/Users/neoh/Desktop/COMP8715/japanese-handwriting-analysis/Image_processing/FELO leaflet examples'

for group_dir in os.listdir(folder_dir):
    path = os.path.join(folder_dir, group_dir)
    for filename in glob.iglob(os.path.join(path, '*.jfif')):
        os.rename(filename, filename[:-4] + '.png')


