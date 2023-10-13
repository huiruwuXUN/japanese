import numpy

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
import random



# 1. read the json file that generate from average_pooling
with open(r'D:\8715_project\japanese-handwriting-analysis\json\averagepooling3.json', 'r') as f:
    data = json.load(f)

# 2. random select some vectors
num_selected_keys = 8
selected_keys = random.choices(list(data.keys()), k=num_selected_keys)
all_vectors = [data[key] for key in selected_keys]
print(selected_keys)
unique_key_count = len(set(selected_keys))
print(f"Number of Unique Keys: {unique_key_count}")
vec_array=numpy.array(all_vectors)


cosine_sim = cosine_similarity(vec_array)

n_clusters_range = range(1, num_selected_keys+1)  # 可以根据实际需要调整

distortions = []

for n_clusters in n_clusters_range:
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)
    cluster_labels = spectral_clustering.fit_predict(1-cosine_sim)  # 使用余弦相似度的倒数

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
