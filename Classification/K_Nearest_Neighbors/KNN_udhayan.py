# K Nearest Algorithm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing and organizing the dataset
k_dataset = pd.read_csv("Social_Network_Ads.csv")
X = k_dataset.iloc[:,2:4].values
Y = k_dataset.iloc[:,-1].values

# Splitting the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size = 3/4,random_state = 0)

# Feature Scaling the dataset
from sklearn.preprocessing import StandardScaler
x_standerdscaler = StandardScaler()
x_train = x_standerdscaler.fit_transform(x_train)
x_test = x_standerdscaler.transform(x_test)

# Training the model 
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 5,metric ='minkowski',p = 2 )
knn_classifier.fit(x_train,y_train)

# Predicting using the trained model
y_predict_train = knn_classifier.predict(x_train)
y_predict_test = knn_classifier.predict(x_test)

# Evaluvating the predicted result
from sklearn.metrics import confusion_matrix
cm_tr = confusion_matrix(y_train,y_predict_train)
cm_te = confusion_matrix(y_test,y_predict_test)

print(cm_tr,"",cm_te)

# visualizing
from matplotlib.colors import ListedColormap
plt.figure("Training Set")
x_set ,y_set = x_train,y_train
x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min() - 1, stop = x_set[:,0].max() + 1, step = 0.01 ),
                    np.arange(start = x_set[:,0].min() - 1, stop = x_set[:,1].max() + 1, step = 0.01))
plt.contourf(x1,x2,knn_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.8, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('blue', 'yellow'))(i), label = j)
plt.title(' KNN Classifier (Training set)')
plt.xlabel('red (not bought)')
plt.ylabel('green (bought)')
plt.legend()
plt.show()




# visualisation || The data || Training Data
from matplotlib.colors import ListedColormap
plt.figure("Test Set")
x_set,y_set = x_test,y_test
x1,x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1,stop = x_set[:,0].max() + 1 , step = 0.01),
                    np.arange(start = x_set[:, 1].min() - 1,stop = x_set[:,1].max() + 1 , step = 0.01))
plt.contourf(x1,x2,knn_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('blue', 'yellow'))(i), label = j)
plt.title('KNN Classifier (Test set)')
plt.xlabel('red (not bought)')
plt.ylabel('green (bought)')
plt.legend()
plt.show()

