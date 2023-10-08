import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity

# 1. 读取JSON数据
with open(r'D:\8715_project\japanese-handwriting-analysis\averagepooling.json', 'r') as f:
    data = json.load(f)

# 2. 合并所有向量为一个数组
all_vectors = []
for key, vectors in data.items():
    all_vectors.append(vectors)

vec_array = np.array(all_vectors)
print(vec_array.shape)

# 3. 计算余弦相似度
cosine_sim = cosine_similarity(vec_array)

# 4. 绘制失真图
n_clusters_range = range(1, 20)  # 可以根据实际需要调整

distortions = []

for n_clusters in n_clusters_range:
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)
    cluster_labels = spectral_clustering.fit_predict(1 - cosine_sim)  # 使用余弦相似度的倒数

    cluster_centers = np.zeros((n_clusters, vec_array.shape[1]))

    for i in range(n_clusters):
        cluster_centers[i] = np.mean(vec_array[cluster_labels == i], axis=0)

    distortion = np.sum((vec_array - cluster_centers[cluster_labels]) ** 2)
    distortions.append(distortion)

plt.figure(figsize=(10, 5))
plt.plot(n_clusters_range, distortions, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.title('Spectral Clustering Distortion Plot')
plt.show()
