import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load the Olivetti Faces dataset
faces_data = fetch_olivetti_faces()
X = faces_data.data

# Perform PCA for dimensionality reduction
pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X)

# Determine the optimal number of clusters using silhouette score
best_score = -1
best_k = -1
for k in range(2, 41):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    silhouette_avg = silhouette_score(X_pca, kmeans.labels_)
    if silhouette_avg > best_score:
        best_score = silhouette_avg
        best_k = k

# Cluster the images using K-Means with the best number of clusters
print(f'best_k: -> {best_k}')
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans.fit(X_pca)

# Visualize the cluster centers (representative faces)
fig, axes = plt.subplots(nrows=1, ncols=best_k, figsize=(15, 3))
for i, ax in enumerate(axes):
    face = pca.inverse_transform(kmeans.cluster_centers_[i])
    ax.imshow(face.reshape(64, 64), cmap='gray')
    ax.axis('off')
    ax.set_title(f'Cluster {i+1}')
plt.show()
