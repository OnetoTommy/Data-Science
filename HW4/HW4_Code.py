import pandas as pd
import numpy as np
from scipy.stats import describe
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

pd.options.mode.copy_on_write = True
'''
Train a multiple linear regression on Train Dataset  with ‘BloodPressure’ as outcome (dependent variable) and all other attributes as features (Independent variables).
Using the regression model, predict and set values of ‘BloodPressure’ in Test Dataset.
Train 19 KNN models with k from 1 to 19 on Train Dataset  dataset with ‘Outcome’ as label and all other attributes as features.
Calculate accuracy of the model for test data (containing predicted values of ‘BloodPressure’).
Report the best K for this KNN model.
'''
####common function
def remove_outliers(dataframe, columns,upper=75, lower=25):
    for column in columns :
        q3, q1 = np.nanpercentile(dataframe[column], [upper, lower])
        fence = 1.5 * (q3 - q1)
        upper_band = q3 + fence
        lower_band = q1 - fence
        print('column:',column,'q1=', q1, ' q3=', q3, ' IQR=', q3 - q1, ' upper=', upper_band,
              'lower=', lower_band)
        dataframe.loc[(dataframe[column] < lower_band) |
                            (dataframe[column] > upper_band), column] = np.nan
    return dataframe

def showCorrelation(dataframe,colx,coly):
    x = dataframe[colx]
    y = dataframe[coly]
    corr = x.corr(y)
    plt.title('Correlation:' +  str(corr))
    plt.scatter(x, y)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))
    (np.unique(x)), color='red')
    plt.xlabel(colx + ' axis')
    plt.ylabel(coly + 'y axis')
    plt.show()

def impute(dataframe):
    minimum_before = list(dataframe.iloc[:, :].min(axis=0))
    maximum_after = list(dataframe.iloc[:, :].max(axis=0))
    im = IterativeImputer(max_iter=10, random_state=0,min_value=minimum_before, max_value=maximum_after)
    im_df = im.fit_transform(dataframe)
    im_df = pd.DataFrame(im_df, columns=dataframe.columns)
    return im_df

##############################################Multiple Reg
org_df = pd.read_csv("hw4_train.csv")

#Define features and outcome for Regression
outcome_df =  org_df.loc[:,org_df.columns == 'BloodPressure']
feat_df =  org_df.loc[:,org_df.columns != 'BloodPressure']

#transform seems not necessary
#feat_df['Glucose'] = np.log(feat_df['Glucose'])

#Seperate test and train data
train_x,test_x,train_y,test_y = train_test_split(feat_df,outcome_df,test_size=0.25, random_state=0)

#Create a multiple Reg model
model = LinearRegression()
model.fit(train_x,  train_y)
print("intercept = ", model.intercept_)

print("slop = ")
print(model.feature_names_in_)
print(model.coef_)

#get the r_sq value
r_sq = model.score(test_x, test_y)
#test_pred_y = model.predict(test_x)

print ('R2 =',r_sq )

#Question 2########################################################
#Using the regression model, predict and set values of ‘BloodPressure’ in Test Dataset.

#Input Dateset
orig_df_2 = pd.read_csv("hw4_test.csv")

#get feature data frame
feat_df_2 =  orig_df_2.loc[:, orig_df_2.columns != 'BloodPressure']

#predict values
test_pred_y_2 = model.predict(feat_df_2)

#save to dataframe,and csv_file
df_2 = orig_df_2.loc[:]
df_2["BloodPressure"] = test_pred_y_2.astype(int)
df_2.to_csv("hw4_linear_predicted.csv",index=False)

####question 3:
# Train 19 KNN models with k from 1 to 19 on Train dataset with ‘Outcome’ as label and all other attributes as features.

#Define features and label for KNN of training data
train_y =  org_df.loc[:,org_df.columns == 'Outcome']
train_x =  org_df.loc[:,org_df.columns != 'Outcome']

#Define features and label for KNN of test data
test_x = df_2.loc[:,df_2.columns != 'Outcome']
test_y = df_2.loc[:,df_2.columns == 'Outcome']

#KNN Model
accuracy = []
for k in range(1,19):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_x, np.ravel(train_y))
    test_pred_y = knn.predict(test_x)
    accuracy.append( accuracy_score(test_y, test_pred_y))

# accuracy
print( "Accuracy=", accuracy)

plt.plot(range(1,19), accuracy, marker='o')

plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Accuracy')
plt.show()

print("K=9")
