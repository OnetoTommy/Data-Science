import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import *

##########################################
##D-Tree Hyper-parameters
##########################################
min_impurity_thr = 0.01               #Default min_impurity threshold for Dtree (Default is 0.01)
ccp_thr = 0.001                       #Default ccp_thr threshold for Dtree (Default is 0.001)
max_depth_thr = 10                    #Default max_depth threshold for Dtree (Default is 10)
min_samples_leaf_thr = 5              #Default min_samples_leaf threshold for Dtree (Default is 5)


#Input Dateset
org_df = pd.read_csv("heart_attack.csv")

#Define features to predict Resistance label
label_df = org_df.loc[:,org_df.columns == 'heart_attack']
feat_df = org_df.loc[:,org_df.columns != 'heart_attack']

#Seperate test and train data
train_feat,test_feat,train_label,test_label = train_test_split(feat_df,label_df,test_size=0.25)

#Create a model using Hyper-parameters
treemodel= tree.DecisionTreeClassifier(criterion="gini",
                                       min_impurity_decrease=min_impurity_thr,
                                       max_depth=max_depth_thr,
                                       min_samples_leaf=min_samples_leaf_thr,
                                       ccp_alpha=ccp_thr)
#Train the model
treemodel.fit(train_feat, train_label)

#Visualize the model
plt.figure(figsize=(9,9))
tree.plot_tree(treemodel,feature_names=train_feat.columns,class_names=['No Heart Attack','Heart Attack'],filled=True)
plt.show()

#Accuracy on training data
train_pred_label = treemodel.predict(train_feat)
training_accuracy= accuracy_score(train_pred_label,train_label)
print ('training accuracy =',training_accuracy )

#Accuracy on test data
test_pred_label = treemodel.predict(test_feat)
testing_accuracy= accuracy_score(test_pred_label,test_label)
print ('testing accuracy =',testing_accuracy )


#Confusion matrix
confusion_matrix = confusion_matrix(test_label, test_pred_label)
ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['No Heart Attack','Heart Attack']).plot()
plt.show()

#Accuracy = (True Positive + True Negative) / ALL
Accuracy = accuracy_score(test_label, test_pred_label)

#Sensitivity (Recall) = True Positive / (True Positive + False Negative)
Sensitivity = recall_score(test_label, test_pred_label)

#Specificity = True Negative / (True Negative + False Positive)
Specificity = recall_score(test_label, test_pred_label, pos_label=0)

print("Accuracy:",Accuracy,"Sensitivity:",Sensitivity,"Specificity:",Specificity)

