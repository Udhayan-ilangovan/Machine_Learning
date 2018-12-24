# K-Mean Clustering
* K-means clustering is an unsupervised learning algorithm.
* The K-means clustering algorithm is used to find groups which have not been explicitly labelled in the data
* It is used when you have unlabelled data ( data without defined classifications or groups). 
* The purpose of this algorithm is to find groups in the data, with the number of groups represented by the variable K. 
* The algorithm works iteratively to assign each data point to one of the K groups based on the features that are provided. Data points are clustered based on feature similarity. 
* The outcome of the K-means clustering algorithm is:
    * The centroids of the K clusters, which can be used to label new data
    * Labels for the training data (each data point is assigned to a single cluster)

* For example
    * Recommender systems
    * Anomaly detection
    * Human genetic clustering

* Example 
    * In this example, we are clustering the customer based on their salary and expenditures in a mall.
    * clustering Variable  => salary and expenditures 

## K-Mean Clustering
import numpy as np # used for mathematical operations

import matplotlib.pyplot as plt # used for plot graphically

import pandas as pd # used for importing dataset and manage dataset

### Importing the dataset
k_m_dataset = pd.read_csv("Mall_Customers.csv")

X = k_m_dataset.iloc[:,3:5].values

### Splitting the dataset into the Training set and Test set ||  no need to split the dataset as testing is not needed

### Finding the cluster number using Elbow method
from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):

    k_cluster_analiser = KMeans(n_clusters = i, init = 'k-means++', n_init = 10 ,max_iter = 300, random_state = 42)
    k_cluster_analiser.fit(X)
    wcss.append(k_cluster_analiser.inertia_)

### Visualising the Elbow
plt.figure("Elbow")

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.xticks(np.arange(10))

plt.grid()

plt.show()    

###Training the correct model to cluster
better_n_cluster = 5

k_clustering = KMeans(n_clusters = better_n_cluster, init = 'k-means++', n_init = 10 ,max_iter = 300, random_state = 42 )

y_means = k_clustering.fit_predict(X)

### Visualising the clusters

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

### Elbow Graph

    * It is used to determine the best number of cluster

<img width="1348" alt="k-means_clustering_elbow" src="https://user-images.githubusercontent.com/32480274/50400273-baf25800-0785-11e9-8673-aaee12550ab6.png">
￼
# Clustered data


    *  5 centroids and cluster 

<img width="1313" alt="k-means_clustering" src="https://user-images.githubusercontent.com/32480274/50400301-fbea6c80-0785-11e9-93dd-ff45aa2878d2.png">
￼
