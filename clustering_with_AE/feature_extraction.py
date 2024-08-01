import numpy as np
from sklearn.decomposition import PCA
import torch.nn.functional as F
import torch
import os
from auto_encoder import AutoEncoder
from dataset import get_dataloader, transform
from extract_images import image_extraction_from_file
from sklearn.cluster import KMeans
from cluster_visualization import *


def extract_features(model, dataloader, device):
    model.eval()  # 确保模型处于评估模式
    features = []
    with torch.no_grad():  # 确保不计算梯度
        for inputs in dataloader:
            inputs = inputs.to(device)
            # 只通过编码器部分
            encoded_features = model.encoder(inputs)
            # global_avg_pooled_features = F.adaptive_avg_pool2d(encoded_features, (1, 1)).view(encoded_features.size(0), -1)
            features.append(encoded_features.cpu().numpy())  # 转换为NumPy数组，如果需要处理
    return np.concatenate(features)  # 返回一个扁平化后的特征数组


def main():
    os.chdir('character_classifying_cnn\outputs')
    img_paths = []
    writers = ['B', 'C', 'D', 'Denvour', 'Grant', 'Monica']
    ground_truth = []
    i = 0
    for writer in writers:
        img_paths += image_extraction_from_file(f'class_{writer}.csv', f'images\{writer}', 'Kanji')
        ground_truth.append(i * np.ones((1, len(image_extraction_from_file(f'class_{writer}.csv', f'images\{writer}', 'Kanji')))))
        i = i + 1
    # img_paths = image_extraction_from_file('class_B.csv', 'images\B', 'Kanji')

    ground_truth = np.hstack(ground_truth)
    # print(ground_truth)
    # print(ground_truth.shape)

    dataloader = get_dataloader(image_paths=img_paths, batch_size=1, transform=transform, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoEncoder()
    model.load_state_dict(torch.load('../../clustering_with_AE/models/model_1.pth')) 
    model.to(device)
    model.eval() 
    features = extract_features(model, dataloader, device)
    features = features.reshape((features.shape[0], -1))

    
    kmeans = KMeans(n_clusters=3)  
    cluster_labels = kmeans.fit_predict(features)
    print(cluster_labels)

    # print(features.shape)

    # 比较两个数组
    matching_elements = ground_truth == cluster_labels
    # 计算相同元素的占比
    similarity_ratio = np.mean(matching_elements)
    print(similarity_ratio)

    plot_clustered_data(features, labels=ground_truth, n_clusters=6)

main()
