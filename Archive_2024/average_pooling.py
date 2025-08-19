import json
import numpy as np
import os


def average_pooling(vectors):
    # compute the mean
    return np.mean(vectors, axis=0).tolist()


def process_json(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    # 结果数据初始化
    result_data = {}

    # key is the path of the img, values are the vectors
    dir_vectors = {}
    for img_path, vector in data.items():

        dir_name = os.path.basename(os.path.dirname(img_path))

        if dir_name not in dir_vectors:
            dir_vectors[dir_name] = []

        dir_vectors[dir_name].append(vector)

    for dir_name, vectors in dir_vectors.items():
        # select 30 vectors
        selected_vectors = vectors[:30]


        avg_vector = average_pooling(selected_vectors)


        result_data[dir_name] = avg_vector

    # save to json format
    with open(output_path, 'w') as f:
        json.dump(result_data, f)



input_path = 'D:\8715_project\japanese-handwriting-analysis\json\pilot_json\pilot_enhence.json'
output_path = 'json/averagepooling3.json'
process_json(input_path, output_path)
