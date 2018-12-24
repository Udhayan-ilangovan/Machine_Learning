# Hierarchical clustering

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Importing the dataset
k_m_dataset = pd.read_csv("Mall_Customers.csv")
X = k_m_dataset.iloc[:,3:5].values

# Finding the cluster number using dendrogram
from scipy.cluster import hierarchy 
plt.figure("Dendrogram")
dendro = hierarchy.dendrogram(hierarchy.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.grid()
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')

# Fitting the Hierarchical clustering
better_n_cluster = 5
from sklearn.cluster import AgglomerativeClustering
agg_h_clustring = AgglomerativeClustering(n_clusters=better_n_cluster,affinity='euclidean',linkage='ward')

# Training the correct model to cluster
y_hc = agg_h_clustring.fit_predict(X)

# Visualising the clusters
plt.figure("Hierarchical clustering using Agglomerative Clustering ")
for i in range(0,better_n_cluster):
    color = list(np.random.choice(range(256), size=3))
    rgbl=[255,0,0]
    plt.scatter(X[y_hc == i, 0], X[y_hc == i, 1], s = 100, c = np.random.shuffle(rgbl), label = "Cluster and {}".format(i))
    
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid()
plt.legend()
plt.show()