import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.evaluate.bootstrap_point632 import accuracy
from networkx.algorithms.bipartite.cluster import clustering
from helper.utils import remove_outliers, impute, categorize,print_nan
import scipy.stats as stats
from sklearn.model_selection import GridSearchCV


pd.set_option('future.no_silent_downcasting', True)

orig_df = pd.read_csv("./data/Airline_Satisfaction.csv")
df = pd.get_dummies(orig_df,
                            columns=['Class',],
                            prefix=['Class'])

### Step 1: Data pre-processing phase

# after checking the data, we found that there are no outliers in the columns;
# there are some missing values in Arrival Delay in Minutes
# but there are Categorical columns that need to be converted to numerical values
df = categorize(df)
df = impute(df)


cols_to_norm = ['Age','Flight Distance','Departure Delay in Minutes', 'Arrival Delay in Minutes']
df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.mean()) /x.std())

# plot the correlation between Age and Glucose
'''
plt.scatter(df_imputed["Age"],df_imputed["Glucose"])
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.title("Correlation between Age and Glucose")
plt.show()
'''

### Step 3: Feature Extraction
# Split data into test and training sets (consider 20% for test).
from sklearn.model_selection import train_test_split

y = df["satisfied"]
X = df.drop(columns=["satisfied"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Use PCA on the training data to create 3 new components from existing features (all columns except outcome).
# Transfer training and test data to the new dimensions (PCs).
from helper.pca import PCA_extract
X_train_pca_df, X_test_pca_df, y_train_pca, y_test_pca = (
    PCA_extract(X_train, X_test, y_train, y_test, n_components=3))

# Train Naive Bayes model
from helper.nb import train_nb
model_nb = train_nb(X_train_pca_df, y_train_pca, X_test_pca_df, y_test_pca)

# Train KNN model
from helper.knn import train_knn
model_knn = train_knn(X_train_pca_df, y_train_pca, X_test_pca_df, y_test_pca)

# Neural Network Model
from helper.nn import train_nn
best_hp = {'hidden_layers':3,'neurons':30, 'activation':'relu','optimizer': 'adam', 'batch_size':16, 'epochs':100}
model_nn = train_nn(X_train_pca_df, y_train_pca, X_test_pca_df, y_test_pca,best_hp)

#all models, Ensemble
models = { 'nb': model_nb, 'knn': model_knn, 'nn': model_nn }
from helper.ensemble import ensemble
ensemble(X_train_pca_df, y_train_pca, X_test_pca_df, y_test_pca, models)

exit()
