import json
import numpy as np
import os
import random

def average_pooling(vectors):
    # 使用numpy库来计算向量列表的平均值
    return np.mean(vectors, axis=0).tolist()

def process_json(input_path, output_path, num_selected_paths):
    with open(input_path, 'r') as f:
        data = json.load(f)

    # 随机选择一定数量的img_path
    selected_paths = random.sample(list(data.keys()), num_selected_paths)
    print("selected path: ",selected_paths)
    # 结果数据初始化
    result_data = {}

    for img_path in selected_paths:
        vector = data[img_path]

        # 使用os库来处理路径和获取子文件夹名
        dir_name = os.path.basename(os.path.dirname(img_path))
        print(dir_name)
        if dir_name not in result_data:
            result_data[dir_name] = []

        result_data[dir_name].append(vector)
    #print(result_data)
    for dir_name, vectors in result_data.items():
        # 从每个文件夹中选择前30个向量
        selected_vectors = vectors[:30]

        # 对选择的向量进行average pooling
        avg_vector = average_pooling(selected_vectors)

        # 保存结果到结果数据中
        result_data[dir_name] = avg_vector

    # 将结果数据保存到新的JSON文件中
    with open(output_path, 'w') as f:
        json.dump(result_data, f)

# 使用函数，设置要随机选择的img_path数量
input_path = 'D:\8715_project\japanese-handwriting-analysis\json\pilot_json\pilot_enhence.json'
output_path = 'json/averagepooling6.json'
num_selected_paths = 7   # 设置要随机选择的img_path数量
process_json(input_path, output_path, num_selected_paths)
