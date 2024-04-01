from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameters to search through
param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance']
}

# Create the KNeighborsClassifier
knn = KNeighborsClassifier()

# Grid search to find the best KNeighborsClassifier
grid_search = GridSearchCV(knn, param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best KNeighborsClassifier
best_knn = grid_search.best_estimator_

# Evaluate the best KNeighborsClassifier on the test set
y_pred = best_knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Best KNeighborsClassifier: {best_knn}")
print(f"Accuracy on test set: {accuracy}")
