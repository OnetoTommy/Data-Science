import pandas as pd
from sklearn import tree
from sklearn.tree import _tree

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.model_selection import train_test_split,GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import *
import numpy as np

'''
Create a binary decision tree using Gini index impurity measure over the diabetes dataset to predict Outcome (as the label) by other attributes (as features).
Consider the following ranges of values for hyper-parameters:
max_depth = [3, 5]
min_sample_split= [5, 10]
min_samples_leaf= [3, 5]
min_impurity_decrease = [0.01, 0.001]
ccp_alpha = [0.001, 0.0001]
Spilt data into train, test, and validation (72%, 20%, 8%) and use validation data to select best hyper-parameter.
Calculate accuracy of the best Dtree on test data.
Extract three rules from the decision tree.
'''

#Input Dateset
org_df = pd.read_csv("diabetes.csv")

#Remove all the outliers from features.
features = org_df.columns.tolist()
features.remove('Outcome')
for feature in features:
    ##Detect Outliers
    q3, q1 = np.percentile(org_df[feature], [75 ,25])
    fence = 1.5 * (q3 - q1)
    upper_band = q3 + fence
    lower_band = q1 - fence
    #print( 'q1=',q1,' q3=', q3, ' IQR=', q3-q1, ' upper=', upper_band, 'lower=',lower_band)
    org_df.loc[(org_df[feature] < lower_band) |
    (org_df[feature] > upper_band), feature] = None

##MICE
imputer = IterativeImputer(max_iter=10, random_state=0)
imputed_dataset = imputer.fit_transform(org_df)
imputed_dataframe = pd.DataFrame(imputed_dataset, columns=org_df.columns)

#Define features to predict Resistance label
label_df = imputed_dataframe.loc[:,imputed_dataframe.columns == 'Outcome']   #contains the target labels (whether a person had a heart attack or not).
feat_df = imputed_dataframe.loc[:,imputed_dataframe.columns != 'Outcome']  # contains the features used to predict heart attacks.

#Seperate test， train and validation data
#HW: Spilt data into train, test, and validation (72%, 20%, 8%) and use validation data to select best hyper-parameter.

# 80% train_validate 20% test
train_validate_feat,test_feat,train_validate_label,test_label = train_test_split(feat_df,label_df,test_size=0.20, random_state=3)
#90% train 10% validate
train_feat,validate_feat,train_label,validate_label = train_test_split(train_validate_feat,train_validate_label,test_size=0.10, random_state=3)

print("train data prepare finished")

# Define the grid of hyperparameters
'''
Consider the following ranges of values for hyper-parameters:
max_depth = [3, 5]
min_sample_split= [5, 10]
min_samples_leaf= [3, 5]
min_impurity_decrease = [0.01, 0.001]
ccp_alpha = [0.001, 0.0001]
'''
param_grid = {
    'max_depth': [3,4,5],
    'min_samples_split': [5, 6, 7, 8, 9, 10],
    'min_samples_leaf': [3, 4, 5],
    'min_impurity_decrease': [0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001],
    'ccp_alpha': [0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001]
}


'''
#user loops to go over all the combination of the params to find best params
best_accuracy = 0
best_params = {}
for depth in param_grid['max_depth']:
    for split in param_grid['min_samples_split']:
        for leaf in param_grid['min_samples_leaf']:
            for impurity in param_grid['min_impurity_decrease']:
                for alpha in param_grid['ccp_alpha']:

                    params = {
                        'min_impurity_decrease': impurity,
                         'max_depth': depth,
                         'min_samples_split': split,
                         'min_samples_leaf': leaf,
                         'ccp_alpha': alpha
                    }

                    treemodel = tree.DecisionTreeClassifier( criterion = "gini", **params)
                    treemodel.fit(train_feat, train_label)

                    #Accuracy on validation data
                    validate_pred_label = treemodel.predict(validate_feat)
                    validate_accuracy = accuracy_score(validate_pred_label, validate_label)

                    #print(params)
                    #print(validate_accuracy)
                    #better score, better params
                    if validate_accuracy > best_accuracy:
                        best_accuracy = validate_accuracy
                        best_params = params


'''

# use GridSearchCV to find best parameter

dtree = tree.DecisionTreeClassifier(criterion="gini")
grid_search = GridSearchCV(dtree, param_grid, cv=3, scoring='accuracy')
grid_search.fit(train_feat, train_label)
best_params = grid_search.best_params_

print ('best_params =',best_params )


#instatiate the model with best params
best_model = tree.DecisionTreeClassifier(criterion="gini", **best_params )
best_model.fit(train_feat, train_label)

#Accuracy on test data
test_pred_label = best_model.predict(test_feat)
test_accuracy= accuracy_score(test_pred_label,test_label)
print ('test accuracy =',test_accuracy )

# Print first three rules from the decision tree
rules = tree.export_text(best_model,feature_names=train_feat.columns,class_names=['No Diabetes','Diabetes'])
rule_arr = rules.split('\n')
counter = 0
for rule in rule_arr :
    print(rule)
    if rule.find('class') > -1 :
        counter += 1
    if(counter >= 3) :
        break