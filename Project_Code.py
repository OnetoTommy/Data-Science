import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
from mlxtend.evaluate import accuracy_score
from networkx import eccentricity
from scikeras.wrappers import KerasClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



# Normalize the data
def get_Normalize(data):
    norm_df = (data - data.mean()) / data.std()
    return norm_df

# Encoding categorical variables

def encoding_method(data):
    # Remove outliers from all data
    numeric_columns = data.select_dtypes(include=['number']).columns
    for column in numeric_columns:
        q3, q1 = np.percentile(data[column], [75,25])
        fence = (q3 - q1) * 1.5
        upper = q3 + fence
        lower = q1 - fence
        data.loc[(data[column] > upper) | (data[column] < lower), column] = np.nan

    # Encoding data
    non_numeric_columns = data.select_dtypes(exclude=['number']).columns
    for column in non_numeric_columns:
        label_encoder = LabelEncoder()
        data[column] = label_encoder.fit_transform(data[column])

    # Impute missing values
    df_encoded = pd.get_dummies(data, drop_first=True)
    imputer = IterativeImputer(max_iter=10, random_state=0)
    imputed_dataset = imputer.fit_transform(df_encoded)
    imputed_dataframe = pd.DataFrame(imputed_dataset, columns=df_encoded.columns)
    return imputed_dataframe

#####################################
# Step 1: Data pre-processing
#################################
def pre_process(data):

    # Remove outliers from all data
    numeric_columns = data.select_dtypes(include=['number']).columns
    for column in numeric_columns:
        q3, q1 = np.percentile(data[column], [75,25])
        fence = (q3 - q1) * 1.5
        upper = q3 + fence
        lower = q1 - fence
        data.loc[(data[column] > upper) | (data[column] < lower), column] = np.nan

    # Impute missing values
    imputer = IterativeImputer(max_iter=10,random_state=0)
    imputed_dataset = imputer.fit_transform(data)
    imputed_dataframe = pd.DataFrame(imputed_dataset, columns=data.columns)
    return imputed_dataframe

# ######################################
# # Step 2: Unsupervised Learning for generating labels
# ######################################
def generate_Label(data):
    # Use K-means clustering on three features of Glucose, BMI and Age to cluster data into two clusters
    three_feat_df = data[['Glucose', 'BMI', 'Age']]
    # print(three_feat_df)
    kmeans_model = KMeans(n_clusters=2, random_state=0)
    data["Cluster"] = kmeans_model.fit_predict(three_feat_df)

    # Assign ‘Diabetes’ name to the cluster with higher average Glucose and ‘No Diabetes’ to the other cluster
    data['Outcome_Glucose'] = data['Glucose'].apply(lambda x: 'Diabetes' if x > data['Glucose'].mean() else 'No Diabetes')

    # Assign ‘Diabetes’ name to the cluster with higher average Glucose and ‘No Diabetes’ to the other cluster
    data['Outcome'] = data['Glucose'].apply(lambda x: 1 if x > data['Glucose'].mean() else 0)

    return data

# # ######################################
# # # Step 3: Feature Extraction
# # ######################################
def extract_features(data):
    # Remove the columns of "Cluster" and "Outcome_Glucose"
    data = data.drop(columns=['Cluster', 'Outcome_Glucose'], errors='ignore')

    #Separate data into features and label
    feat_df = data.drop(columns=['Outcome'])
    label_df = data['Outcome']

    #split data into test and training sets
    train_x, test_x, train_y, test_y = train_test_split(feat_df, label_df, test_size=0.2)
    # return train_x, test_x, train_y, test_y

    # # RFE on the training data
    # model = LogisticRegression(max_iter=200)
    # rfe = RFE(estimator=model, n_features_to_select=6)
    # rfe.fit(train_x, train_y)
    #
    # # transform the data
    # train_x = rfe.transform(train_x)
    # test_x = rfe.transform(test_x)

    #PCA on the training data
    pca = PCA(n_components=3)
    # pca.fit(train_x)
    transformed_train_x = pca.fit_transform(train_x)
    transformed_test_x = pca.transform(test_x)

    pca_train_df = pd.DataFrame(transformed_train_x, columns=['PC1', 'PC2', 'PC3'])
    pca_test_df = pd.DataFrame(transformed_test_x, columns=['PC1', 'PC2', 'PC3'])

    # pca_relationship_train = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=train_x.columns)
    # return pca_relationship_train

    return pca_train_df, pca_test_df, train_y, test_y

# # ######################################
# # # Step 3: Feature Extraction
# # ######################################
def extract_features_as(data):
    # Remove the columns of "Cluster" and "Outcome_Glucose"
    # data = data.drop(columns=['Cluster', 'Outcome_Glucose'])

    # #Separate data into features and label
    # feat_df = data.drop(columns=['satisfied'])
    # feat_df  = (feat_df - feat_df.mean()) / feat_df.std()
    # label_df = data[['satisfied']]

    feat_df = data.iloc[:, :-1]
    label_df = data.iloc[:, -1]

    #split data into test and training sets
    train_x, test_x, train_y, test_y = train_test_split(feat_df, label_df, test_size=0.2)
    # return train_x, test_x, train_y, test_y

    #PCA on the training data
    pca = PCA(n_components=3)
    pca.fit(train_x)
    transformed_train_x = pca.transform(train_x)
    transformed_test_x = pca.transform(test_x)

    pca_train_df = pd.DataFrame(transformed_train_x, columns=['PC1', 'PC2', 'PC3'])
    pca_test_df = pd.DataFrame(transformed_test_x, columns=['PC1', 'PC2', 'PC3'])

    # pca_relationship_train = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=train_x.columns)
    # return pca_relationship_train

    return pca_train_df, pca_test_df, train_y, test_y

# ######################################
# # Step 4: Classification using a super learner
# ######################################

def get_classification_accuracy(train_x, test_x, train_y, test_y, max_iter, param_grid, cv):
    nb_model = GaussianNB()
    knn_model = KNeighborsClassifier()
    nn_model = MLPClassifier(random_state=42, max_iter=max_iter, verbose=0)

    base_classifiers = [
        ('nb', nb_model),
        ('knn', knn_model),
        ('nn', nn_model)
    ]
    meta_learner = DecisionTreeClassifier()
    # Super learner using Stacking
    super_learner = StackingClassifier(estimators = base_classifiers, final_estimator = meta_learner, cv = cv)
    # Find hyperparameters for all these models which provide the best accuracy rate
    grid_search = GridSearchCV(super_learner, param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(train_x, train_y)
    # best_model = grid_search.best_estimator_
    # Get the best hyperparameters
    best_params = grid_search.best_params_
    # print("Best hyperparameters:", best_params)
    # Get the best cross-validated accuracy score
    best_accuracy_train = grid_search.best_score_
    # print("Best cross-validation accuracy:", best_accuracy)
    # Get the testing accuracy
    best_accuracy_model = grid_search.best_estimator_
    y_pred_test = best_accuracy_model.predict(test_x)
    accuracy_test = accuracy_score(test_y, y_pred_test)

    return best_accuracy_train, best_params, accuracy_test


    # Second methods
    # Hyperparameters for all models
    # hidden_units = [5, 10, 15]
    # knn__n_neighbors = [3,5 ,8]
    # min_impurity_thrs = [0.001, 0.0001]
    # min_samples_split_thrs = [5, 10]
    # max_depths = [3, 5]
    # min_samples_leaf_thrs = [3, 5]
    # ccp_thrs = [0.001, 0.0001]
    # best_accuracy = 0
    # best_params = {}
    #
    # # loop function to select the hyperparameters for the best accuracy
    # for k in knn__n_neighbors:
    #     for hidden_unit in hidden_units:
    #         for min_impurity_thr in min_impurity_thrs:
    #             for min_samples_split_thr in min_samples_split_thrs:
    #                 for max_depth in max_depths:
    #                     for min_samples_leaf_thr in min_samples_leaf_thrs:
    #                        for ccp_thr in ccp_thrs:
    #
    #                             #nb/nn/knn model
    #                             nb = GaussianNB()
    #                             knn = KNeighborsClassifier(n_neighbors=k)
    #                             nn = KerasClassifier(build_fn=create_nn_model(train_x, hidden_unit))
    #                             # nn = KerasClassifier(model=create_nn_model, hidden_units = 5,epochs=10)
    #
    #                             # Base classifiers
    #                             base_classifiers = [
    #                                 ('nb', nb),
    #                                 ('knn', knn),
    #                                 ('nn', nn),
    #                             ]
    #
    #                             # Meta Learner
    #                             meta_learner = DecisionTreeClassifier(
    #                                 max_depth=max_depth,
    #                                 min_samples_split=min_samples_split_thr,
    #                                 min_impurity_decrease=min_impurity_thr,
    #                                 min_samples_leaf=min_samples_leaf_thr,
    #                                 ccp_alpha=ccp_thr,
    #                             )
    #
    #                             #Super learner using Stacking
    #                             super_learner = StackingClassifier(estimators = base_classifiers, final_estimator = meta_learner, cv = 5)
    #                             super_learner.fit(train_x, train_y)
    #                             y_pred = super_learner.predict(train_x)
    #                             accuracy = accuracy_score(train_y, y_pred)
    #                             if accuracy > best_accuracy:
    #                                 best_accuracy = accuracy
    #                                 best_params = {
    #                                     'hidden_units': hidden_unit,
    #                                     'n_neighbors': k,
    #                                     'max_depth': max_depth,
    #                                     'min_samples_split':min_samples_split_thr,
    #                                     'min_impurity_decrease': min_impurity_thr,
    #                                     'min_samples_leaf':min_samples_leaf_thr,
    #                                     'ccp_alpha': ccp_thr,
    #                                 }
    #
    # return best_accuracy,best_params

    #
    # # # Hyperparameter grid
    # # param_grid = {
    # #     'knn__n_neighbors': list(range(3, 8)),
    # #     'final_estimator__max_depth': list(range(3, 10)),
    # #     'final_estimator__min_impurity_decrease': [0.001,0.0001],  # Decision tree min impurity threshold
    # #     'final_estimator__min_samples_split': [5, 10],  # Minimum samples required to split
    # #     'final_estimator__min_samples_leaf': [3,5],  # Minimum samples required in a leaf
    # #     'final_estimator__ccp_alpha': [0.001,0.0001],
    # # }
    #
    # # Grid search to find the best hyperparameters
    # grid_search = GridSearchCV(super_learner, param_grid, cv=5, scoring='accuracy')
    # grid_search.fit(train_x, train_y)
    #
    # # Best model and hyperparameters
    # # print("Best parameters found:", grid_search.best_params_)
    # # print("Best cross-validation accuracy:", grid_search.best_score_)
    #
    # # Evaluate on test data
    # best_model = grid_search.best_estimator_
    # y_pred = best_model.predict(test_x)
    # test_accuracy = accuracy_score(test_y, y_pred)
    # # print("Test accuracy:", test_accuracy)
    # return (print("Best parameters found:", grid_search.best_params_),
    #         print("Best cross-validation accuracy:", grid_search.best_score_),
    #         print("Test accuracy:", test_accuracy)
    #         )


# Input dataset
org_df = pd.read_csv("diabetes_project.csv")

# Step 1: Data pre-processing phase
imputer_df = pre_process(org_df)
# print(imputer_df)

norm_df = get_Normalize(imputer_df)
# print("Step 1: Data pre-processing phase result: ", norm_df)

# Step 2: Unsupervised Learning for generating labels
cluster_df = generate_Label(norm_df)
# print("Step 2:Unsupervised Learning for generating labels: ",cluster_df)

# Step 3: Feature Extraction
pca_train_df, pca_test_df, train_y, test_y = extract_features(cluster_df)
# print("Step 3: Feature extraction on training data: ", pca_train_df)
# print("Step 3: Feature extraction on test data: ", pca_test_df)

# pca_relationship_train = extract_features(cluster_df)
# print(pca_relationship_train)

# Step 4: Classification using a super learner
param_grid = {
            'knn__n_neighbors': [3, 5],
            # 'knn__weights': ['uniform', 'distance'],
            'nn__hidden_layer_sizes': [(2,), (3,)],
            'final_estimator__max_depth': [3, 5],
            'final_estimator__min_impurity_decrease': [0.001,0.0001],  # Decision tree min impurity threshold
            'final_estimator__min_samples_split': [5, 10],  # Minimum samples required to split
            'final_estimator__min_samples_leaf': [3,5],  # Minimum samples required in a leaf
            'final_estimator__ccp_alpha': [0.001,0.0001],
            }
supper_learn_accuracy, best_params, accuracy_test = get_classification_accuracy(pca_train_df, pca_test_df, train_y, test_y, max_iter=500, param_grid=param_grid, cv=5)
print("supper_learn_accuracy", supper_learn_accuracy)
print("best_params", best_params)
print("accuracy_test", accuracy_test)

# # Step 5: Employing the model
#
# # Input dataset
# org_df_as = pd.read_csv("Airline_Satisfaction.csv")
#
# # Step 1: Data pre-processing phase
# imputer_df = encoding_method(org_df_as)
# # print(imputer_df)
#
# # Step 3: Feature Extraction
# pca_train_df, pca_test_df, train_y, test_y = extract_features_as(imputer_df)
# # print("Step 3: Feature extraction on training data:\n", pca_train_df)
# # print("Step 3: Feature extraction on test data:\n", pca_test_df)
#
# # pca_relationship_train = extract_features(cluster_df)
# # print(pca_relationship_train)
#
# # Step 4: Classification using a super learner
# param_grid = {
#             'knn__n_neighbors': [3, 5],
#             'nn__hidden_layer_sizes': [ (50,), (100,)],
#             'final_estimator__max_depth': [3, 5],
#             'final_estimator__min_impurity_decrease': [0.001,0.0001],  # Decision tree min impurity threshold
#             'final_estimator__min_samples_split': [5, 10],  # Minimum samples required to split
#             'final_estimator__min_samples_leaf': [3,5],  # Minimum samples required in a leaf
#             'final_estimator__ccp_alpha': [0.001,0.0001],
#             }
# supper_learn_accuracy, best_params, accuracy_test = get_classification_accuracy(pca_train_df, pca_test_df, train_y, test_y, max_iter=200, param_grid=param_grid, cv=5)
# print("supper_learn_accuracy_as", supper_learn_accuracy)
# print("best_params_as", best_params)
# print("accuracy_test_as", accuracy_test)
