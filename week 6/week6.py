import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Step 1: Generate Data
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

# Step 2: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Fine-tune the Decision Tree using GridSearchCV
param_grid = {'max_leaf_nodes': [None, 5, 10, 20, 30, 50], 'min_samples_split': [2, 5, 10]}
tree_clf = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(tree_clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Step 4: Display the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Step 5: Train the Decision Tree with the best hyperparameters on the full training set
best_tree_clf = grid_search.best_estimator_
best_tree_clf.fit(X_train, y_train)

# Step 6: Evaluate the model on the test set
accuracy_test = best_tree_clf.score(X_test, y_test)
print(f"Accuracy on the test set: {accuracy_test:.4f}")
