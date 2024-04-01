import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

# Load the Olivetti Faces dataset
faces_data = fetch_olivetti_faces()
X, y = faces_data.data, faces_data.target

# Perform stratified sampling to split the dataset into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

# Perform PCA for dimensionality reduction
# This increased performance of the the clustering from 2 clusters to 39 clusters
pca = PCA(n_components=100, random_state=42)
X_train_pca = pca.fit_transform(X_train)

# Determine the optimal number of clusters using silhouette score
best_score = -1
best_k = -1

# The loop provided 39 clusters without stratified sampling
# and 38 clusters with stratified sampling
for k in range(2, 41):  # 40 classes in the Olivetti Faces dataset
    silhouette_avg = 0
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train_pca)
    silhouette_avg = silhouette_score(X_train_pca, kmeans.labels_)
    if silhouette_avg > best_score:
        best_score = silhouette_avg
        best_k = k

# Cluster the images using K-Means with the best number of clusters
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans.fit(X_train_pca)

# Visualize the cluster centers (representative faces)
fig, axes = plt.subplots(nrows=1, ncols=best_k, figsize=(15, 3))
for i, ax in enumerate(axes):
    face = pca.inverse_transform(kmeans.cluster_centers_[i])
    ax.imshow(face.reshape(64, 64), cmap='gray')
    ax.axis('off')
    ax.set_title(f'Cluster {i+1}')
plt.show()
