import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

# Convert labels to integers
y = y.astype(int)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)

# Standardize the data (optional but often recommended for PCA)
X_train_std = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test_std = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

# Apply PCA with 95% explained variance ratio
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

# Print the number of components selected
print(f"Number of components selected: {pca.n_components_}")

# Visualize the explained variance ratio
explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(explained_variance_ratio_cumsum)
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio Cumulative Sum')
plt.title('Explained Variance Ratio Cumulative Sum')
plt.show()
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

# Train a Random Forest classifier on the reduced dataset and measure the time
start_time_pca = time.time()

rf_clf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf_pca.fit(X_train_pca, y_train)

training_time_pca = time.time() - start_time_pca
print(f"Training time on reduced dataset: {training_time_pca:.2f} seconds")

# Evaluate the model on the test set
X_test_pca = pca.transform(X_test_std)
y_pred_pca = rf_clf_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)

print(f"Accuracy on the test set with PCA: {accuracy_pca:.4f}")
