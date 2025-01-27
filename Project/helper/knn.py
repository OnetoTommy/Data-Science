from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

n_neighbors_max = 25

def train_knn(X_train,y_train,X_test,y_test, best_k=None):
    print('*** Training KNN')
    feat_train, feat_validate, label_train, label_validate = train_test_split(X_train, y_train, test_size=0.1, random_state=0)
    if best_k:
        best_model = KNeighborsClassifier(n_neighbors=best_k)
        best_model.fit(feat_train, np.ravel(label_train))
    else:
        # Train KNN models with k from 3 to 20 on Train Dataset dataset with ‘Outcome’ as label and all other attributes as features.
        k_values = range(1, n_neighbors_max,2)
        best_k = 0
        best_accuracy = 0
        best_model = None
        accuracies = []
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(feat_train, np.ravel(label_train))
            label_pred = knn.predict(feat_validate)
            accuracy = accuracy_score(np.ravel(label_validate), label_pred)
            accuracies.append(accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
                best_model = knn
    '''    
    # Plot the results
    plt.plot(k_values, accuracies, marker='o')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title('Elbow Method for Optimal k')
    plt.xticks(k_values)
    plt.show()  # we can see 5 is a good option
    exit()
    '''
    print(f"Best k: {best_k}")
    print(f"Best Accuracy: {best_accuracy}")

    # Accuracy on train data
    train_pred_label = best_model.predict(X_train)
    train_accuracy = accuracy_score(train_pred_label, y_train)
    print(f"Train Data Accuracy: {train_accuracy}")


    predict = best_model.predict(X_test)
    test_accuracy = accuracy_score(predict, y_test)
    print(f"Test Data Accuracy: {test_accuracy}")
    return best_model