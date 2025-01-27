from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
import numpy as np
import pandas as pd

param_grid = {
    'max_depth': [2, 3, 5],
    'min_samples_split': [2, 3, 5 ],
    'min_samples_leaf': [2, 3, 5],
    'min_impurity_decrease': [0.01, 0.001],
    'ccp_alpha': [0.01, 0.001, 0.0001]
}
def ensemble(X_train,y_train,X_test,y_test, models):
    #prepare the dataframe
    X_train_dtree = {}
    X_test_dtree = {}
    for key, model in models.items():
        X_train_dtree[key] = model.predict(X_train)
        X_test_dtree[key] = model.predict(X_test)
        if key == 'nn':
            # change the sigmoid output to 0 or 1; reshape the array
            X_train_dtree[key] = np.where(X_train_dtree[key] > 0.5, 1, 0).ravel()
            X_test_dtree[key] = np.where(X_test_dtree[key] > 0.5, 1, 0).ravel()
    X_train_dtree_df = pd.DataFrame(X_train_dtree)
    X_test_dtree_df = pd.DataFrame(X_test_dtree)

    print('*** Ensemble, Train Decision Tree')
    # Define hyperparameter grid
    dtree = tree.DecisionTreeClassifier(criterion="gini")
    grid_search = GridSearchCV(dtree, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_dtree_df, y_train)
    best_model = grid_search.best_estimator_

    best_model.fit(X_train_dtree_df, y_train)


    # Accuracy on train data
    train_pred_label = best_model.predict(X_train_dtree_df)
    train_accuracy = accuracy_score(train_pred_label, y_train)
    print(f"Train Data Accuracy: {train_accuracy}")

    # Accuracy on test data
    test_pred_label = best_model.predict(X_test_dtree_df)
    test_accuracy = accuracy_score(test_pred_label, y_test)
    print('test data accuracy =', test_accuracy)
    print('Best parameter:', grid_search.best_params_)
    return best_model
