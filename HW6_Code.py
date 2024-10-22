import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from mlxtend.evaluate import accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import  RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt



#import database
org_df =  pd.read_csv('diabetes.csv')

#Define label and features
df_label = org_df['Outcome']
df_feat = org_df.loc[:, org_df.columns != 'Outcome']

#Seperate train1 and train2
train_x, test_x, train_y, test_y = train_test_split(df_feat, df_label, test_size=0.25, random_state=0)

#Train RF and Ada models
rf_3 = RandomForestClassifier(n_estimators=3, random_state=42)
rf_50 = RandomForestClassifier(n_estimators=50, random_state=42)
ad_3 = AdaBoostClassifier(n_estimators=3, algorithm='SAMME')
ad_50 = AdaBoostClassifier(n_estimators=50,  algorithm='SAMME')


#Cross-validation method
k_folds = KFold(n_splits=5, shuffle=True, random_state=42)
score_rf3 = cross_val_score(rf_3, train_x, train_y, cv=k_folds)
score_rf50 = cross_val_score(rf_50, train_x, train_y, cv=k_folds)
score_ad3 = cross_val_score(ad_3, train_x, train_y, cv=k_folds)
score_ad50 = cross_val_score(ad_50, train_x, train_y, cv=k_folds)

#Score_means
score_rf3_mean = score_rf3.mean()
score_rf50_mean = score_ad50.mean()
score_ad3_mean = score_ad3.mean()
score_ad50_mean = score_ad50.mean()

if score_rf3_mean > score_ad3_mean:
    print('score_rf3_mean =',score_rf3_mean,
          'score_ad3_mean =', score_ad3_mean,
          'RF 3 score is better than AD 3.')
else:
    print('score_rf3_mean =',score_rf3_mean,
          'score_ad3_mean =', score_ad3_mean,
          'RF 3 score is less than AD 50.')

if score_rf50_mean > score_ad50_mean:
    print('score_rf50_mean =',score_rf50_mean,
          'score_ad50_mean =', score_ad50_mean,
          'RF 50 score is better than AD 3.')
elif score_rf50_mean == score_ad50_mean:
    print('score_rf50_mean =', score_rf50_mean,
          'score_ad50_mean =', score_ad50_mean,
          'RF 50 score is equal to AD 50.')
else:
    print('score_rf50_mean =',score_rf50_mean,
        'score_ad50_mean =', score_ad50_mean,
          'RF 50 score is less than AD 50.')

#################################################
# Stacking model
#################################################

# Define base models (estimators)
estimators_3 = [
    ('rf_3', RandomForestClassifier(n_estimators=3, random_state=42)),
    ('ad_3', AdaBoostClassifier(n_estimators=50, algorithm='SAMME')),
]

estimators_50 = [
    ('rf_50', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('ad_50', AdaBoostClassifier(n_estimators=50, algorithm='SAMME')),
]

#Define the stacking model with a Logistic Regression meta_model
stack_model_3 = StackingClassifier(estimators=estimators_3, final_estimator=LogisticRegression())
stack_model_50 = StackingClassifier(estimators=estimators_50, final_estimator=LogisticRegression())

#Stacking scores
stack_score_3 = cross_val_score(stack_model_3, train_x, train_y, cv=k_folds)
stack_score_50 = cross_val_score(stack_model_50, train_x, train_y, cv=k_folds)
print('stack_score_3', stack_score_3.mean())
print('stack_score_50', stack_score_50.mean())

#Fit the stacking model
stack_model_3.fit(train_x, train_y)
stack_model_50.fit(train_x, train_y)

#Predict on the test data
stack_pred_3 = stack_model_3.predict(test_x)
stack_pred_50 = stack_model_50.predict(test_x)

# Calculate accuracy
stack3_accuracy = accuracy_score(test_y, stack_pred_3)
stack50_accuracy = accuracy_score(test_y, stack_pred_50)
print('stack_accuracy_3 =', stack3_accuracy)
print('stack_accuracy_50 =', stack3_accuracy)

#Predict on test data
rf_3.fit(train_x, train_y)
rf_50.fit(train_x, train_y)
ad_3.fit(train_x, train_y)
ad_50.fit(train_x, train_y)
rf3_pred_y = rf_3.predict(test_x)
ad3_pred_y = ad_3.predict(test_x)
rf50_pred_y = rf_50.predict(test_x)
ad50_pred_y = rf_50.predict(test_x)

#Accuracy
rf3_accuracy = accuracy_score(test_y, rf3_pred_y)
ad3_accuracy = accuracy_score(test_y, ad3_pred_y)
ad50_accuracy = accuracy_score(test_y, ad50_pred_y)
rf50_accuracy = accuracy_score(test_y, rf50_pred_y)
print('rf3_accuracy=',rf3_accuracy,
      'rf50_accuracy =',rf50_accuracy,
      'ad3_accuracy =', ad3_accuracy,
      'ad50_accuracy =', ad50_accuracy)

#Line plot for accuracy
accuracy = [rf3_accuracy, ad3_accuracy, stack3_accuracy,rf50_accuracy, ad50_accuracy, stack50_accuracy]
name = ['rf3_accuracy', 'ad3_accuracy', 'stack3_accuracy', 'rf50_accuracy', 'ad50_accuracy','stack50_accuracy']

plt.figure(figsize=(12,5))  # Adjusts figure size
plt.plot(name, accuracy, color='green')
plt.title('Different Model Accuracies')
# plt.ylabel('Accuracy')
# plt.xlabel('Model')
plt.show()

#Line plot for scores
score = [score_rf3_mean, score_ad3_mean, stack_score_3.mean(),score_rf50_mean,score_ad50_mean, stack_score_50.mean()]
model = ['score_rf3', 'score_ad3', 'score_stack3', 'score_rf50', 'score_ad50', 'score_stack50']
plt.figure(figsize=(12,5))  # Adjusts figure size
plt.plot(model, score, color='red')
plt.title('Different Model Scores')
# plt.ylabel('Accuracy')
# plt.xlabel('Model')
plt.show()