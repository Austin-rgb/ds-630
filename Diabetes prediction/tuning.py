def search(model,grid,X,y):
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(X, y)
    return grid_result

def tune_logr(X,y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l1','l2','elasticnet']
    c_values = [100, 10, 1.0, 0.1, 0.01]
    # define grid search
    grid = dict(solver=solvers,penalty=penalty,C=c_values)
    return search(model, grid,X,y)

def tune_ridge(X,y):
    from sklearn.linear_model import RidgeClassifier
    model = RidgeClassifier()
    alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # define grid search
    grid = dict(alpha=alpha)
    return search(model, grid,X,y)

def tune_knn(X,y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    n_neighbors = range(1, 21, 2)
    weights = ['uniform', 'distance']
    metric = ['euclidean', 'manhattan', 'minkowski']
    # define grid search
    grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
    return search(model, grid,X,y)

def tune_svc(X,y):
    from sklearn.svm import SVC
    model = SVC()
    kernel = ['linear','poly', 'rbf', 'sigmoid']
    C = [100, 10, 1.0, 0.1, 0.01]
    gamma = [0.1, 0.01, 0.001, 0.0001]
    # define grid search
    grid = dict(kernel=kernel,C=C,gamma=gamma)
    return search(model, grid,X,y)

def tune_bagc(X,y):
    from sklearn.ensemble import BaggingClassifier
    model = BaggingClassifier()
    n_estimators = [10, 100, 1000]
    # define grid search
    grid = dict(n_estimators=n_estimators)
    return search(model, grid,X,y)

def tune_randomf(X,y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    n_estimators = [10, 100, 1000]
    max_features = ['sqrt', 'log2']
    # define grid search
    grid = dict(n_estimators=n_estimators,max_features=max_features)
    return search(model, grid,X,y)

def tune_gbc(X,y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier()
    n_estimators = [10, 100, 1000]
    learning_rate = [0.001, 0.01, 0.1]
    subsample = [0.5, 0.7, 1.0]
    max_depth = [3, 7, 9]
    # define grid search
    grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
    return search(model, grid,X,y)