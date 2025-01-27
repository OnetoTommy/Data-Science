import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import export_graphviz
import graphviz

'''
Train two RF models and two Adaboost models on all data in dataset 
    with ‘outcome’ as label and all other attributes as features,
    using 3 and 50 as the number of estimators.
Using a cross-validation method, 
    calculate scores of all four models for 5 folds.
For each pair of RF and Adaboost models with the same number of estimators compare mean of scores.
'''

#Input Dateset
org_df = pd.read_csv("diabetes.csv")

#Define features and label for KNN
label_df =  org_df.loc[:,org_df.columns == 'Outcome']
feat_df =  org_df.loc[:,org_df.columns != 'Outcome']

#Random Forest
rf3 = RandomForestClassifier(n_estimators=3)
rf50 = RandomForestClassifier(n_estimators=50)

ad3 = AdaBoostClassifier(n_estimators=3,algorithm="SAMME")
ad50 = AdaBoostClassifier(n_estimators=50,algorithm="SAMME")

k_folds = KFold(n_splits = 5)
scores ={
"rf3" : cross_val_score(rf3, feat_df, label_df.values.ravel(), cv = k_folds),
"rf50" : cross_val_score(rf50, feat_df, label_df.values.ravel(), cv = k_folds),
"ad3" : cross_val_score(ad3, feat_df, label_df.values.ravel(), cv = k_folds),
"ad50" : cross_val_score(ad50, feat_df, label_df.values.ravel(), cv = k_folds)
}
print("scores:")
print(scores)

score_means = {
"rf3" : scores["rf3"].mean(),
"rf50" :scores["rf50"].mean(),
"ad3" : scores["ad3"].mean(),
"ad50" : scores["ad50"].mean()
}

# Preparing results for comparison
print("score_means:")
print(score_means)

compare = {
    "rf3_scores_mean vs ad3_scores" : ">" if score_means["rf3"] > score_means["ad3"] else "<",
    "rf50_scores_mean vs ad50_scores": ">" if score_means["rf50"] > score_means["ad50"] else "<",
}
print(compare)
