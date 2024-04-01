import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the Olivetti Faces dataset
faces_data = fetch_olivetti_faces()
X, y = faces_data.data, faces_data.target

# Split the dataset into training, validation, and test sets using stratified sampling
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Find a good number of clusters using silhouette score
best_score = -1
best_k = -1
for k in range(2, 41):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train)
    silhouette_avg = silhouette_score(X_train, kmeans.labels_)
    if silhouette_avg > best_score:
        best_score = silhouette_avg
        best_k = k

# Cluster the images using K-Means with the best number of clusters
print(f'best number of clusters: {best_k}')
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans.fit(X_train)
cluster_centers = kmeans.cluster_centers_

# Visualize the clusters
fig, axes = plt.subplots(nrows=1, ncols=best_k, figsize=(15, 3))
for i, ax in enumerate(axes):
    ax.imshow(cluster_centers[i].reshape(64, 64), cmap='gray')
    ax.axis('off')
    ax.set_title(f'Cluster {i+1}')
plt.show()
