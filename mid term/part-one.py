from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
import time

# Load MNIST data
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

# Convert target labels to binary classification (0 and 1)
y = (y.astype(int) == 0).astype(int)  # 0 for non-zeros, 1 for zeros

# Split data into training (50,000), validation (10,000), and test (10,000) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=50000, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=10000, random_state=42)

# Train Logistic Regression Classifier
start_time = time.time()
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
training_time_log_reg = time.time() - start_time

# Train SVM Classifier
start_time = time.time()
svm_clf = SVC(kernel='rbf', probability=True)
svm_clf.fit(X_train, y_train)
training_time_svm = time.time() - start_time

# Predict probabilities on validation set
y_val_proba_log_reg = log_reg.predict_proba(X_val)[:, 1]  # Probability of class 1
y_val_proba_svm = svm_clf.predict_proba(X_val)[:, 1]      # Probability of class 1

# Calculate accuracy on validation set
accuracy_log_reg = accuracy_score(y_val, log_reg.predict(X_val))
accuracy_svm = accuracy_score(y_val, svm_clf.predict(X_val))

# Calculate AUC on validation set
auc_log_reg = roc_auc_score(y_val, y_val_proba_log_reg)
auc_svm = roc_auc_score(y_val, y_val_proba_svm)

# Print results
print("Logistic Regression - Computation Time:", training_time_log_reg)
print("SVM - Computation Time:", training_time_svm)

print("Logistic Regression - Accuracy:", accuracy_log_reg)
print("SVM - Accuracy:", accuracy_svm)

print("Logistic Regression - AUC:", auc_log_reg)
print("SVM - AUC:", auc_svm)
