print("Importing  required packages")
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
# Load the MNIST dataset
print("loading mnist dataset")
mnist = datasets.fetch_openml('mnist_784', parser='auto')

# Split the dataset into features (X) and labels (y)
X, y = mnist['data'], mnist['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVC model
from sklearn.svm import SVC
svc = SVC()
#Tune the SVC
print("Tuning svc using GridSearchCV")
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}

from sklearn.model_selection import GridSearchCV
# Perform GridSearchCV to find the best parameters
grid_search = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best parameters found
print("Best parameters:", grid_search.best_params_)

# Evaluate the model with best parameters on the test set
best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
print("Accuracy on test set:", accuracy)

# Train the SVC model
print("training SVC model")
svc.fit(X_train,y_train)

# Predict on the test set
print("using trained svc model to predict on the dataset")
y_pred = svc.predict(X_test)

# Evaluate the SVC model using Mean Squared Error (MSE)
# Use numpy.array to convert y_test and y_pred to int arrays
import numpy as np
mse=mean_squared_error(np.array([int(test) for test in y_test]),np.array([int(test) for test in y_pred]))
print(f"Mean Squared Error: {mse}")

