from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier

#Import train set
from preprocessing import y_train, X_train_scaled, y_test, X_test_scaled 

# Initialize models
logistic_regression = LogisticRegression()
random_forest = RandomForestClassifier()
svm = SVC()
knn = KNeighborsClassifier()
ridge = RidgeClassifier()

# Train the models
logistic_regression.fit(X_train_scaled, y_train)
random_forest.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)
knn.fit(X_train_scaled, y_train)
ridge.fit(X_train_scaled, y_train)


from sklearn.metrics import classification_report

# Predictions
logistic_regression_preds = logistic_regression.predict(X_test_scaled)
random_forest_preds = random_forest.predict(X_test_scaled)
svm_preds = svm.predict(X_test_scaled)
knn_preds = knn.predict(X_test_scaled)
ridge_preds = ridge.predict(X_test_scaled)


# Evaluate performance
print("Logistic Regression:")
print(classification_report(y_test, logistic_regression_preds))

print("Random Forest:")
print(classification_report(y_test, random_forest_preds))

print("SVM:")
print(classification_report(y_test, svm_preds))

print("KNeighbors:")
print(classification_report(y_test, knn_preds))

print("RidgeClassifier:")
print(classification_report(y_test, ridge_preds))

# Model tuning

def evaluate_hyperparams(model,xtrain,ytrain,xtest,ytest):
    from sklearn.metrics import classification_report
    model.fit(xtrain, ytrain)
    best_preds = model.predict(xtest) 
    return classification_report(ytest, best_preds)

# Perform grid search with cross-validation
# Tune svc
print('Tuning SVC')
from tuning import tune_svc
svm_grid_search = tune_svc(X_train_scaled, y_train) 
# Get best hyperparameters
best_params = svm_grid_search.best_params_
print("Best Hyperparameters:", best_params)#Best Hyperparameters: {'C': 100, 'gamma': 0.0001, 'kernel': 'rbf'}
# Re-evaluate SVM with best hyperparameters
svm_best = SVC(**best_params)
print('Evaluating svc on tuned hyperparameters ')
print(evaluate_hyperparams(svm_best,X_train_scaled, y_train, X_test_scaled, y_test))


# Tune random_forest
print('Tuning RandomForestClassifier ')
from tuning import tune_randomf
randomf_grid_search = tune_randomf(X_train_scaled, y_train) 
# Get best hyperparameters
best_params = randomf_grid_search.best_params_
print("Best Hyperparameters:", best_params)#Best Hyperparameters: {'C': 100, 'gamma': 0.0001, 'kernel': 'rbf'}

# Re-evaluate Random Forest  with best hyperparameters
svm_best = RandomForestClassifier(**best_params)
print('Evaluating svc on tuned hyperparameters ')
print(evaluate_hyperparams(svm_best,X_train_scaled, y_train, X_test_scaled, y_test))

# Tune random_forest
print('Tuning Ridge classifier')
from tuning import tune_ridge
randomf_grid_search = tune_ridge(X_train_scaled, y_train) 
# Get best hyperparameters
best_params = randomf_grid_search.best_params_
print("Best Hyperparameters:", best_params)#Best Hyperparameters: {'C': 100, 'gamma': 0.0001, 'kernel': 'rbf'}

# Re-evaluate RidgeClassifier  with best hyperparameters
svm_best = RidgeClassifier(**best_params)
print('Evaluating knn on tuned hyperparameters ')
print(evaluate_hyperparams(svm_best,X_train_scaled, y_train, X_test_scaled, y_test))

# Tune logistic regressor
print('Tuning LogisticRegression ')
from tuning import tune_logr
randomf_grid_search = tune_logr(X_train_scaled, y_train) 
# Get best hyperparameters
best_params = randomf_grid_search.best_params_
print("Best Hyperparameters:", best_params)#Best Hyperparameters: {'C': 100, 'gamma': 0.0001, 'kernel': 'rbf'}

# Re-evaluate LogisticRegression  with best hyperparameters
svm_best = LogisticRegression(**best_params)
print('Evaluating LogisticRegression on tuned hyperparameters ')
print(evaluate_hyperparams(svm_best,X_train_scaled, y_train, X_test_scaled, y_test))


# Tune knn
print('Tuning KNeighborsClassifier ')
from tuning import tune_knn
randomf_grid_search = tune_knn(X_train_scaled, y_train) 
# Get best hyperparameters
best_params = randomf_grid_search.best_params_
print("Best Hyperparameters:", best_params)#Best Hyperparameters: {'C': 100, 'gamma': 0.0001, 'kernel': 'rbf'}

# Re-evaluate KNeighborsClassifier  with best hyperparameters
svm_best = KNeighborsClassifier(**best_params)
print('Evaluating KNeighborsClassifier on tuned hyperparameters ')
print(evaluate_hyperparams(svm_best,X_train_scaled, y_train, X_test_scaled, y_test))


# SVC and RandomForestClassifier performed the best
# Reaching accuracy of 0.75 after tuning 
