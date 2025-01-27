import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn_extra.cluster import KMedoids
from kmodes.kmodes import KModes
from pyclustering.cluster.kmedians import kmedians
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
'''
* Find an optimal number of clusters for k-means over the marketing dataset.
* Use the optimal cluster number in K-means to cluster the marketing data.
* Visualize the relations between income and spending for all resulting clusters using a scatter plot.
* Visualize the relations between income and age for all resulting clusters using another scatter plot.
* Try to find names for different clusters based on these visualizations.
'''

#Input Dateset
marketing_df = pd.read_csv("market_ds.csv")

#normalize data (hard to explain the relationship according to the plot if we normalize the data)
#marketing_df = (marketing_df - marketing_df.mean()) / marketing_df.std()

#kmeans
inertias = []
for k in range(1,11):
    model = KMeans(n_clusters=k, init='k-means++' ,random_state=42)
    model.fit(marketing_df)
    inertias.append(model.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')
#plt.show()

#According to the plot, we choose 4 as the cluster number.
optimal_clusters = 6

model = KMeans(n_clusters=optimal_clusters, random_state=42)
marketing_df['Cluster'] = model.fit_predict(marketing_df)

plt.figure(figsize=(8, 5))
for cluster in range(optimal_clusters):
    plt.scatter(marketing_df[marketing_df['Cluster'] == cluster]['Income'], marketing_df[marketing_df['Cluster'] == cluster]['Spending'],label=f'Cluster {cluster}')
plt.title('Income vs Spend by Cluster')
plt.xlabel('Income')
plt.ylabel('Spend')
plt.legend()
plt.grid(True)
#plt.show()

plt.figure(figsize=(8, 5))
for cluster in range(optimal_clusters):
    plt.scatter(marketing_df[marketing_df['Cluster'] == cluster]['Age'], marketing_df[marketing_df['Cluster'] == cluster]['Income'],label=f'Cluster {cluster}')
plt.title('Income vs Age by Cluster')
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()
plt.grid(True)
plt.show()
