import json
from sklearn.cluster import KMeans
import numpy as np

# 1. read json
with open(r'D:\8715_project\japanese-handwriting-analysis\averagepooling.json', 'r') as f:
    data = json.load(f)

# 2. merge all the vector in to one array
all_vectors = []
for key, vectors in data.items():
    all_vectors.extend(vectors)

# 3. perform k-mean
kmeans = KMeans(n_clusters=5).fit(all_vectors)

# 4. calculate the distance
cluster_to_distance = {}
for i in range(len(kmeans.cluster_centers_)):
    mask = kmeans.labels_ == i
    cluster_vectors = np.array(all_vectors)[mask]
    distance = np.sum(np.linalg.norm(cluster_vectors - kmeans.cluster_centers_[i], axis=1))
    cluster_to_distance[f"Cluster_{i}"] = distance

print(cluster_to_distance)





