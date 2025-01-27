import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np

#Input Dateset
org_df = pd.read_csv("non_linear.csv")

'''
Train two SVM models, one with ‘linear’ kernel and another with ‘rbf’ kernel on dataset ( ‘label’ as label and all other attributes as features) using 75% of data as training and remaining data as test.
Calculate accuracy of both models and report the best accuracy.
'''

#Labels and Features
label_df = org_df.loc[:,org_df.columns == 'label']
feat_df = org_df.loc[:,org_df.columns != 'label']

#Split Train and Test Data
x_train, x_test, y_train, y_test = train_test_split(feat_df, label_df, test_size = 0.25)
y_train = np.ravel(y_train)

#Create SVM Model
svm_linear = SVC(kernel='linear') # kernel='rbf'
svm_linear.fit(x_train, y_train)

svm_rbf = SVC(kernel='rbf') # kernel='rbf'
svm_rbf.fit(x_train, y_train)

#Accuracy of Model
accuracy_linear = svm_linear.score(x_test,y_test)
accuracy_rbf = svm_rbf.score(x_test,y_test)

print("linear SVM accuracy:  ", accuracy_linear)
print("rbf SVM accuracy:  ", accuracy_rbf)

print("best accuracy is:  ", accuracy_rbf if accuracy_rbf > accuracy_linear else accuracy_linear )




