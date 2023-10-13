import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances


def read_features_from_json(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return data


def compute_average_cosine_distance(features, image_paths):
    if len(image_paths) < 2:  # Avoid computing distance for folders with only one image
        return None
    vectors = [features[path] for path in image_paths]
    distances = cosine_distances(vectors)
    upper_triangle_indices = np.triu_indices(len(vectors), k=1)
    avg_distance = np.mean(distances[upper_triangle_indices])
    return avg_distance

if __name__ == "__main__":
    # read the json file
    json_file = r"D:\8715_project\japanese-handwriting-analysis\seg_enhence_json\seg_letter_enhance.json"
    features = read_features_from_json(json_file)

    # compute cosine sim
    subdir_to_avg_distance = {}
    for image_path in features.keys():
        subdir = os.path.dirname(image_path)
        #print("sub_dir ",subdir)
        if subdir not in subdir_to_avg_distance:
            subdir_images = [path for path in features.keys() if os.path.dirname(path) == subdir]

            avg_distance = compute_average_cosine_distance(features, subdir_images)
            subdir_to_avg_distance[subdir] = avg_distance

    # save to excel file
    df = pd.DataFrame(list(subdir_to_avg_distance.items()), columns=["Subfolder", "Average Cosine Distance"])
    excel_path = "D:\\8715_project\\japanese-handwriting-analysis\\average_cosine_distances_enhence.xlsx"
    df.to_excel(excel_path, index=False)
    print(f"result saved to {excel_path}")
