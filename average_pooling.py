import json
import numpy as np
import os


def average_pooling(vectors):
    # 使用numpy库来计算向量列表的平均值
    return np.mean(vectors, axis=0).tolist()


def process_json(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    # 结果数据初始化
    result_data = {}

    # 创建一个字典，以子文件夹名为键，该子文件夹中的向量列表为值
    dir_vectors = {}
    for img_path, vector in data.items():
        # 使用os库来处理路径和获取子文件夹名
        dir_name = os.path.basename(os.path.dirname(img_path))

        if dir_name not in dir_vectors:
            dir_vectors[dir_name] = []

        dir_vectors[dir_name].append(vector)

    for dir_name, vectors in dir_vectors.items():
        # 从每个子文件夹中选择30个向量
        selected_vectors = vectors[:30]

        # 对选择的向量进行average pooling
        avg_vector = average_pooling(selected_vectors)

        # 保存结果到结果数据中
        result_data[dir_name] = avg_vector

    # 将结果数据保存到新的JSON文件中
    with open(output_path, 'w') as f:
        json.dump(result_data, f)


# 使用函数
input_path = 'D:\8715_project\japanese-handwriting-analysis\pilot_json\pilot_enhence.json'
output_path = 'json/averagepooling.json'
process_json(input_path, output_path)
