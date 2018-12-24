# Hierarchical clustering
* Hierarchical clustering is an algorithm that groups similar objects into clusters. 
* Each cluster is distinct from each other 
* The data in each cluster are broadly similar to each other.
* working of Hierarchical clustering 
    * It starts by treating every data as a separate cluster.
    * Then, it identifies the two clusters that are closest together and merges the two most similar clusters 
    * The distance between two is the Euclidean distance.
    * This continues until all the similar clusters are merged together.

* For example
    * Analysis of antimicrobial activity
    * Grouping of shopping items
    * Search result grouping
* The Dendrogram 
    * It shows the hierarchical relationship between objects. 
    * It is created as an output from hierarchical clustering. 
    * The main purpose of a dendrogram is to identify the best number of cluster. 


## Hierarchical clustering
import numpy as np # used for mathematical operations

import matplotlib.pyplot as plt # used for plot graphically

import pandas as pd # used for importing dataset and manage dataset

### Importing the dataset
k_m_dataset = pd.read_csv("Mall_Customers.csv")

X = k_m_dataset.iloc[:,3:5].values

### Finding the cluster number using dendrogram
from scipy.cluster import hierarchy 

plt.figure("Dendrogram")

dendro = hierarchy.dendrogram(hierarchy.linkage(X, method='ward'))

plt.title('Dendrogram')

plt.grid()

plt.xlabel('Customers')

plt.ylabel('Euclidean Distance')

### Fitting the Hierarchical clustering
better_n_cluster = 5

from sklearn.cluster import AgglomerativeClustering

agg_h_clustring = AgglomerativeClustering(n_clusters=better_n_cluster,affinity='euclidean',linkage='ward')

### Training the correct model to cluster
y_hc = agg_h_clustring.fit_predict(X)

### Visualising the clusters
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

<img width="1316" alt="hierarchical_clustering_dendrogram" src="https://user-images.githubusercontent.com/32480274/50400406-c1cd9a80-0786-11e9-9472-95c5ee6224b6.png">

￼
### Clustered data

    *  5 centroids and cluster 

<img width="1311" alt="hierarchical_clustering" src="https://user-images.githubusercontent.com/32480274/50400411-c5f9b800-0786-11e9-80ab-0a08184217db.png">
￼