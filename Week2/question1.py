from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate some data
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the hyperparameters to search through
param_grid = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10],
    'gamma': [0.1, 1, 10]
}

# Create the SVR regressor
svr = SVR()

# Grid search to find the best SVR predictor
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best SVR predictor
best_svr = grid_search.best_estimator_

# Evaluate the best SVR predictor on the test set
y_pred = best_svr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Best SVR predictor: {best_svr}")
print(f"Mean squared error on test set: {mse}")
