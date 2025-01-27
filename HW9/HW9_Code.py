from cProfile import label

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load data
org_df = pd.read_csv("world_ds.csv", index_col=0)

# Define features and label
label_df = org_df.loc[:, org_df.columns == 'development_status']
feat_df = org_df.loc[:, org_df.columns != 'development_status']

label_arr = label_df.values.ravel()

# Mutual Info
# Correlation Matrix
corr_matrix = org_df.corr()
# print(corr_matrix)

# Wrapper methods

# KNN Model as classifier
knn = KNeighborsClassifier(n_neighbors=9)

# Forward
sfs = SFS(knn, k_features='best', forward=True, scoring='accuracy', cv=5)
sfs.fit(feat_df, label_arr)
sfs_metric_df = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
# print(sfs_metric_df)
selected_features_df_sfs = feat_df[list(sfs_metric_df['feature_names'][3])]
# selected_features_df_sfs.head()
print("*** 2: Employ forward wrapper method to select best three features from the dataset.")
print(list(selected_features_df_sfs.columns))

# PCA

# Normalize all features
norm_feat_df = (feat_df - feat_df.mean()) / feat_df.std()

# Create and train PCA model
pca_model = PCA(n_components=3)
pca_model.fit(norm_feat_df)

# Show covariance matrix
covariance_matrix = pca_model.get_covariance()
# print(model.get_covariance())

# Show eigenvalues and eigenvectors
pc_feature_relationship = pd.DataFrame(pca_model.components_, columns=feat_df.columns, index=['PC1', 'PC2', 'PC3'])
print("*** 3: Use a PCA model to create 3  new components from existing features:")
print(pc_feature_relationship)

# Explain:
print("""*** 4.Explain each PC (new features) based on the correlations with old features: 
 PC1 Strongly correlates to :
    (Positively) :Life expectancy,Physicians Per K, Minimum wage",
    (Negatively): Birth Rate, Fertility Rate, Infant Mortality ; Maternal Mortality ratio
 suggesting it represents the healthy care
 
 PC2 Strongly correlates to 
    (Positively) Co2 Emission, GDP, Population
 suggesting it represents the macro economy activity

 PC3: Strongly correlates to 
    (Positively): Gasoline Price, Minimum Wage
    (Negatively): unemployment rate
  suggesting it represents the labor market and energy price.
 """)

# LDA
# Create and train LDA model
lda = LDA(n_components=2)
lda_model = lda.fit(feat_df, label_arr)
transformed_features_df_lda = lda.transform(feat_df)

lda_df = pd.DataFrame(transformed_features_df_lda, columns=['LD1', 'LD2'])

print("*** 5:Use a LDA model to create 2 new components from existing features.")
lda_df.head()

# accuracy comparison
print("*** 6:Compare the accuracy of a KNN classifier on new or selected features resulting by forward, PCA and LDA.")
# accuracy by SFS
knn.fit(selected_features_df_sfs, label_arr)
y_pred_sfs = knn.predict(selected_features_df_sfs)
accuracy_sfs = accuracy_score(label_arr, y_pred_sfs)
print("Accuracy of SFS : ", accuracy_sfs)

# accuracy by PCA
transformed_features_df_pca = pca_model.transform(norm_feat_df)
knn.fit(transformed_features_df_pca, label_arr)
y_pred_pca = knn.predict(transformed_features_df_pca)
accuracy_pca = accuracy_score(label_arr, y_pred_pca)
print("Accuracy of PCA : ", accuracy_pca)

# accuracy by LDA
knn.fit(transformed_features_df_lda, label_arr)
y_pred_lda = knn.predict(transformed_features_df_lda)
accuracy_lda = accuracy_score(label_arr, y_pred_lda)
print("Accuracy of LDA : ", accuracy_lda)

print("Over all, LDA demonstrates the highest classification accuracy on this data, while PCA shows the lowest accuracy.")
