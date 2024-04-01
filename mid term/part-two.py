from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import time

# Load MNIST data
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

# Split data into training (50,000), validation (10,000), and test (10,000) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=50000, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=10000, random_state=42)

# Train Logistic Regression Classifier
start_time = time.time()
log_reg = LogisticRegression(max_iter=100)
log_reg.fit(X_train, y_train)
training_time_log_reg = time.time() - start_time

# Train SVM Classifier
start_time = time.time()
svm_clf = SVC(kernel='rbf')
svm_clf.fit(X_train, y_train)
training_time_svm = time.time() - start_time

# Predict on validation set
y_val_pred_log_reg = log_reg.predict(X_val)
y_val_pred_svm = svm_clf.predict(X_val)

# Calculate accuracy on validation set
accuracy_log_reg = accuracy_score(y_val, y_val_pred_log_reg)
accuracy_svm = accuracy_score(y_val, y_val_pred_svm)

# Calculate confusion matrix for Logistic Regression
conf_matrix_log_reg = confusion_matrix(y_val, y_val_pred_log_reg)

# Calculate confusion matrix for SVM
conf_matrix_svm = confusion_matrix(y_val, y_val_pred_svm)

# Error analysis for Logistic Regression
errors_log_reg = (y_val_pred_log_reg != y_val)
X_val_errors_log_reg = X_val[errors_log_reg]
y_val_errors_log_reg = y_val[errors_log_reg]

# Error analysis for SVM
errors_svm = (y_val_pred_svm != y_val)
X_val_errors_svm = X_val[errors_svm]
y_val_errors_svm = y_val[errors_svm]

# Print results
print("Logistic Regression - Accuracy:", accuracy_log_reg)
print("SVM - Accuracy:", accuracy_svm)

print("Logistic Regression - Confusion Matrix:")
print(conf_matrix_log_reg)

print("SVM - Confusion Matrix:")
print(conf_matrix_svm)

print("Logistic Regression - Errors:", len(X_val_errors_log_reg))
print("SVM - Errors:", len(X_val_errors_svm))
