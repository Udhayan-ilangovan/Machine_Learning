# Principal Component Analysis
import numpy as np # used for mathematical operations
import matplotlib.pyplot as plt # used for plot graphically
import pandas as pd # used for importing dataset and manage dataset

# Importing the dataset
pca_dataset = pd.read_csv("Wine.csv")

# Separating the dataset
X = pca_dataset.iloc[:,0:13].values
y = pca_dataset.iloc[:,-1].values

# Splitting the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
x_standedScaler = StandardScaler()
x_train = x_standedScaler.fit_transform(x_train)
x_test = x_standedScaler.transform(x_test)

# Applying PCA
from sklearn.decomposition import PCA
pca_analysier = PCA(n_components= 2)
x_train = pca_analysier.fit_transform(x_train)
x_test = pca_analysier.transform(x_test)
explaind_variance = pca_analysier.explained_variance_ratio_

# Training the logistic model using linear regressor
from sklearn.linear_model import LogisticRegression
pca_log_classifier = LogisticRegression(random_state=0)
pca_log_classifier.fit(x_train,y_train)


# predicting the trainied model || Verification
y_pred_train = pca_log_classifier.predict(x_train)

# predicting the trainied model || Validation
y_pred_test = pca_log_classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_test)

# visualisation || The data || Training Data
from matplotlib.colors import ListedColormap
plt.figure("Logistic: Training Set  with PCA")
x_set,y_set = x_train,y_train
x1,x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1,stop = x_set[:,0].max() + 1 , step = 0.01),
                    np.arange(start = x_set[:, 1].min() - 1,stop = x_set[:,1].max() + 1 , step = 0.01))
plt.contourf(x1,x2,pca_log_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.8, cmap = ListedColormap(('red', 'green','purple')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('blue', 'yellow','cyan'))(i), label = j)
plt.title('Logistic Regression with PCA (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# visualisation || The data || Test Data
from matplotlib.colors import ListedColormap
plt.figure("Logistic: Test Set with PCA")
x_set,y_set = x_test,y_test
x1,x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1,stop = x_set[:,0].max() + 1 , step = 0.01),
                    np.arange(start = x_set[:, 1].min() - 1,stop = x_set[:,1].max() + 1 , step = 0.01))
plt.contourf(x1,x2,pca_log_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.8, cmap = ListedColormap(('red', 'green','purple')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('blue', 'yellow','cyan'))(i), label = j)
plt.title('Logistic Regression with PCA (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

