import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import PrintTime
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.neighbors import NearestNeighbors
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from  sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


#input data
data = pd.read_csv('market_ds.csv')
data_scaled =  (data - data.mean()) / data.std()
print(data_scaled)

# #Detect Outliers in Income for Glucose
# q3, q1 = np.percentile(data['Income'], [75,25])
# fence = 1.5 * (q3 - q1)
# upper = q3 + fence
# lower = q1 - fence
# data.loc[(data['Income'] < lower) | (data['Income'] > upper), 'Income'] = None
#
# ##Detect Outliers in Spending for Glucose
# q3, q1 = np.percentile(data['Spending'], [75,25])
# fence = 1.5 * (q3 - q1)
# upper = q3 + fence
# lower = q1 - fence
# data.loc[(data['Spending'] < lower) | (data['Spending'] > upper), 'Spending'] = None
# print(data)
#
# #MICE
# imputer = IterativeImputer(max_iter=10,random_state=0)
# imputer.dataset = imputer.fit_transform(data)
# data = pd.DataFrame(imputer.dataset,columns=data.columns)

#DBSCAN Step 1: Preprocess the data (optional scaling)

#DBSCAN for Income and Spending


# #linkage
# linkage_data = linkage(data_scaled, method='single', metric='euclidean')
# plt.figure(figsize=(10, 5))
# dendrogram(linkage_data, truncate_mode='level', p=7)
# plt.title('Dendrogram (Single Linkage)')
# plt.xlabel('Sample Index')
# plt.ylabel('Distance')
# plt.show()



# # 绘制 k-distance 图以选择 eps
# neighbors = NearestNeighbors(n_neighbors=5)
# neighbors_fit = neighbors.fit(data_scaled)
# distances, indices = neighbors_fit.kneighbors(data_scaled)
# distances = np.sort(distances[:, 4])
# plt.plot(distances)
# plt.title('K-distance Graph')
# plt.xlabel('Data Points')
# plt.ylabel('Distance to 5th Nearest Neighbor')
# plt.grid(True)
# plt.show()

# Step 2: Apply DBSCAN
#DBSCSN to detect outliers
#detect outliers for DBSCAN
dbscan = DBSCAN(eps=0.6, min_samples=5)
labels = dbscan.fit_predict(data_scaled)
data['Cluster'] = labels
outliers = data[data['Cluster'] == -1]
cleaned_data = data[data['Cluster'] != -1]
cleaned_data = cleaned_data.drop('Cluster',axis=1)

# print('old/n',cleaned_data['Spending'])

# x = np.array(cleaned_data['Income']).reshape(-1, 1)
# y = np.array(cleaned_data['Spending'])
#
# # Step 2: Split the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#
# # Step 3: Transform the data for polynomial features
# degree = 2  # You can change the degree as needed
# poly = PolynomialFeatures(degree=degree)
# x_poly = poly.fit_transform(x_train)
#
# # Step 4: Fit the polynomial regression model
# model = LinearRegression()
# model.fit(x_poly, y_train)
#
# # Step 5: Make predictions
# x_test_poly = poly.transform(x)
# y_pred = model.predict(x_test_poly)
# cleaned_data['Spending'] = y_pred



# #DBSCAN for Income and Age
# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(cleaned_data[['Income', 'Age']])
#
# # # 绘制 k-distance 图以选择 eps
# # neighbors = NearestNeighbors(n_neighbors=5)
# # neighbors_fit = neighbors.fit(data_scaled)
# # distances, indices = neighbors_fit.kneighbors(data_scaled)
# # distances = np.sort(distances[:, 4])
# # plt.plot(distances)
# # plt.title('K-distance Graph')
# # plt.xlabel('Data Points')
# # plt.ylabel('Distance to 5th Nearest Neighbor')
# # plt.grid(True)
# # plt.show()
#
# # Step 2: Apply DBSCAN
# dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust eps and min_samples based on your data
# labels = dbscan.fit_predict(data_scaled)
#
# # Step 3: Identify core points, border points, and outliers
# cleaned_data['Cluster'] = labels
#
# # Points with label -1 are considered noise (outliers)
# outliers = cleaned_data[cleaned_data['Cluster'] == -1]
# cleaned_data = cleaned_data[cleaned_data['Cluster'] != -1]
# cleaned_data = cleaned_data.drop('Cluster',axis=1)
# cleaned_data.to_csv('cleaned_data.csv')
#standardscaler for cleaned_data
scaled_data = (cleaned_data - cleaned_data.mean()) / cleaned_data.std()

#Find an optimal number of clusters for k-means
wcss =[]
k_values = range(1, 11)
for k in k_values:
    k_means = KMeans(n_clusters=k )
    k_means.fit(scaled_data)
    wcss.append(k_means.inertia_)
    k += 1

# Optimal number of clusters for k-means
silhouette_scores = []
best_score = 0
optimal_cluster = 0
for k in range(2, 11):  # Checking k=2 to k=11
    k_means = KMeans(n_clusters=k, random_state=42)
    k_means.fit(scaled_data)
    score = silhouette_score(scaled_data, k_means.labels_)
    silhouette_scores.append(score)
    if score > best_score:
        best_score = score
        optimal_cluster = k
    k += 1

print('optimal_cluster=',optimal_cluster)
print('silhouette_score=',best_score)

#Plotting the Elbow chart for k-means
plt.figure(figsize=(10,5))
plt.plot(k_values, wcss)
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('wcss')
plt.grid(True)
plt.show()

#Optimal number of clusters for k-means
model = KMeans(optimal_cluster) #optimal_cluster
model.fit(scaled_data)

#scatter plot of Income and Spending
clusters = {}
i = 0
for i in range(optimal_cluster):  #optimal_cluster
    clusters[i] = scaled_data.loc[model.labels_ == i,:]
    plt.scatter(clusters[i].loc[:, 'Income'], clusters[i].loc[:, 'Spending'])
    i += 1

plt.show()

#scatter plot of Income and Age
for i in range(optimal_cluster): #optimal_cluster
    clusters[i] = scaled_data.loc[model.labels_ == i,:]
    plt.scatter(clusters[i].loc[:, 'Income'], clusters[i].loc[:, 'Age'])
    i += 1

plt.show()

#find names for different clusters
data_names = pd.DataFrame()
model.fit(cleaned_data)
for i in range(optimal_cluster):
    clusters[i] = cleaned_data.loc[model.labels_ == i, :]
    mean_income = clusters[i].loc[:, 'Income'].mean()
    mean_spending = clusters[i].loc[:, 'Spending'].mean()
    mean_age = clusters[i].loc[:, 'Age'].mean()
    data_names.loc[i, 'cluster'] = f'cluster_{i}'
    data_names.loc[i, 'mean_Age'] = mean_age
    data_names.loc[i, 'mean_Income'] = mean_income
    data_names.loc[i, 'mean_Spending'] = mean_spending
    i += 1
print(data_names)

##mean value for different clusters
#      cluster   mean_Age  mean_Income  mean_Spending
# 0  cluster_0  54.083333    47.083333      41.683333
# 1  cluster_1  32.700000    96.500000      87.800000
# 2  cluster_2  24.300000    26.200000      77.450000
# 3  cluster_3  46.529412    82.588235      19.294118
# 4  cluster_4  27.000000    56.657895      49.131579
# 5  cluster_5  32.760000    76.960000      80.800000
# Summary for different clusters
# Cluster 0: Older age, lower income, lower spending.
# Cluster 1: Younger age, higher income, higher spending.
# Cluster 2: Younger age, lower income, lower spending.
# Cluster 3: Middle-aged, moderate income, lower spending.
# Cluster 4: Younger age, moderate income, moderate spending.
# Cluster 5: Younger age, moderate income, higher spending.



