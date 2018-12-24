# K-Mean Clustering
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Importing the dataset
k_m_dataset = pd.read_csv("Mall_Customers.csv")
X = k_m_dataset.iloc[:,3:5].values

# Splitting the dataset into the Training set and Test set ||  no need to split the dataset as testing is not needed

# Finding the cluster number using Elbow method
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    k_cluster_analiser = KMeans(n_clusters = i, init = 'k-means++', n_init = 10 ,max_iter = 300, random_state = 42)
    k_cluster_analiser.fit(X)
    wcss.append(k_cluster_analiser.inertia_)

# Visualising the Elbow
plt.figure("Elbow")
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.xticks(np.arange(10))
plt.grid()
plt.show()    

# Training the correct model to cluster
better_n_cluster = 5
k_clustering = KMeans(n_clusters = better_n_cluster, init = 'k-means++', n_init = 10 ,max_iter = 300, random_state = 42 )
y_means = k_clustering.fit_predict(X)

# Visualising the clusters
plt.figure("KMean Clustering")
for i in range(0,better_n_cluster):
    color = list(np.random.choice(range(256), size=3))
    rgbl=[255,0,0]
    plt.scatter(X[y_means == i, 0], X[y_means == i, 1], s = 100, c = np.random.shuffle(rgbl), label = "Cluster and {}".format(i))
plt.scatter(k_clustering.cluster_centers_[:, 0], k_clustering.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid()
plt.legend()
plt.show()