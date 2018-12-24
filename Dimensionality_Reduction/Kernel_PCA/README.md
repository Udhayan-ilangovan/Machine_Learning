# Kernel Principal Component Analysis
* Kernel -  PCA is used for Non-linear problems
* K-PCA uses Kernel trick to obtain principal components in different space.
* KPCA finds new directions based on kernel matrix
* PCA - Principal Component Analysis
    * PCA is used to emphasize variation and bring out the influential patterns in the dataset.
    * It is a dimensionality reduction algorithm.
    * It is an unsupervised algorithm.
* Uses
    * Novelty detection
    * Image de-noising
* Dependent variable
    * It is the target variable (outcome).
    * The variable expected to change based on the manipulation of the independent variable.
* Independent variable
    * It is the input variable (predictor).		
    * The independent variable can be manipulated to get the different outcomes independent variable. 
* Example 
    * In this example, we are predicting the customer who will buy a car based on their salary and age.
    * Dependent variable  => Purchased the car (Yes || No).
    * Independent variable => Salary and age.
## Kernel PCA

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

### Importing the dataset
lr_dataset = pd.read_csv("Social_Network_Ads.csv")

### Separating the dataset
X = lr_dataset.iloc[:,[2,3]].values

Y = lr_dataset.iloc[:,-1].values

### Splitting the dataset
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=3/4,random_state=0)

### Feature Scaling
x_test1 = x_test

from sklearn.preprocessing import StandardScaler

x_standedScaler = StandardScaler()

x_train = x_standedScaler.fit_transform(x_train)

x_test = x_standedScaler.transform(x_test)

### Applying kernel PCA
from sklearn.decomposition import KernelPCA

pca_analysier = KernelPCA(n_components= 2, kernel='rbf')

x_train = pca_analysier.fit_transform(x_train)

x_test = pca_analysier.transform(x_test)

### Training the logistic model using linear regressor
from sklearn.linear_model import LogisticRegression

k_pca_classifier = LogisticRegression(random_state=0)

k_pca_classifier.fit(x_train,y_train)


### predicting the trainied model || Verification
y_pred_train = k_pca_classifier.predict(x_train)

# predicting the trainied model || Validation
y_pred_test = k_pca_classifier.predict(x_test)


### Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)


### visualisation || The data || Training Data
from matplotlib.colors import ListedColormap

plt.figure("Logistic: Training Set with Kernel PCA")

x_set,y_set = x_train,y_train

x1,x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1,stop = x_set[:,0].max() + 1 , step = 0.01),
                    np.arange(start = x_set[:, 1].min() - 1,stop = x_set[:,1].max() + 1 , step = 0.01))

plt.contourf(x1,x2,k_pca_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.8, cmap = ListedColormap(('red', 'green')))

plt.xlim(x1.min(), x1.max())

plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('blue', 'yellow'))(i), label = j)

plt.title('Logistic Regression with Kernel PCA (Training set)')

plt.xlabel('KPCA_1')

plt.ylabel('KPCA_2')

plt.legend()

plt.show()

### visualisation || The data || Test Data
from matplotlib.colors import ListedColormap

plt.figure("Logistic: Test Set with Kernel PCA")

x_set,y_set = x_test,y_test

x1,x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1,stop = x_set[:,0].max() + 1 , step = 0.01),
                    np.arange(start = x_set[:, 1].min() - 1,stop = x_set[:,1].max() + 1 , step = 0.01))

plt.contourf(x1,x2,k_pca_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.8, cmap = ListedColormap(('red', 'green')))

plt.xlim(x1.min(), x1.max())

plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('blue', 'yellow'))(i), label = j)

plt.title('Logistic Regression with Kernel PCA (Test set)')

plt.xlabel('KPCA_1')

plt.ylabel('KPCA_2')

plt.legend()

plt.show()

### Sample Dataset

<img width="607" alt="kernel_pca_dataset" src="https://user-images.githubusercontent.com/32480274/50404845-03713c00-07ad-11e9-836e-7bf60ae305c3.png">
￼
### Prediction results.

    * Predicted using confusion matrix.
    * 0 - 0 => Correct prediction of the Customer who hadn't purchased the car.
    * 1 - 1  => Correct prediction of the Customer who purchased the car.
    * 0 - 1 => Incorrect prediction of the Customer who hadn't purchased the car.
    * 1 - 0 => Incorrect prediction of the Customer who purchased the car.

<img width="214" alt="kernel_pca_cm" src="https://user-images.githubusercontent.com/32480274/50404849-0835f000-07ad-11e9-9ba3-5c9665d22254.png">

### Training Set

### Visualization 

    * Blue and yellow Dots => Data used for training (Training set)
    * Red part  =>  Customer who hadn't purchased the car
    * Green part => Customer who purchased the car
<img width="1312" alt="kernal_pca_test_set" src="https://user-images.githubusercontent.com/32480274/50404851-0c620d80-07ad-11e9-8a74-d44c3dea2dad.png">

### Test Set

### Visualization 

    * Blue and yellow Dots => Data used for testing (Test set)
    * Red part  =>  Customer who hadn't purchased the car
    * Green part => Customer who purchased the car
<img width="1288" alt="kernal_pca_training_set" src="https://user-images.githubusercontent.com/32480274/50404853-108e2b00-07ad-11e9-9cf7-04755bd50714.png">

￼

