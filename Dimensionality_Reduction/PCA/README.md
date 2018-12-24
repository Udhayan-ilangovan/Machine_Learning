# Principal Component Analysis
* PCA is used to emphasize variation and bring out the influential patterns in the dataset.
* It is a dimensionality reduction algorithm.
* It is an unsupervised algorithm.
* It is used to identify the patterns in data and detect the correlation between variables
* Using the strong correlation it reduces the dimensionality.
* The main purpose of PCA is to explore and visualize the data in an easy approach.
* Uses
    * Noise Filtering
    * visualisation,
    * Feature Extraction
    * Stock market Prediction
* Dependent variable
    * It is the target variable (outcome).
    * The variable expected to change based on the manipulation of the independent variable.
* Independent variable
    * It is the input variable (predictor).		
    * The independent variable can be manipulated to get the different outcomes independent variable. 
* Example
    * We are recommending the new the wine to the customers by classifying the wine based on the chemical characteristics of it.
    * We have 178 records of wine's chemical characteristics.
    * Dependent variable  => Customers ( Customer1, Customers 2, Customers 3 ).
    * Independent variable => Chemical characteristics of wine.

## Principal Component Analysis
import numpy as np # used for mathematical operations

import matplotlib.pyplot as plt # used for plot graphically

import pandas as pd # used for importing dataset and manage dataset

### Importing the dataset
pca_dataset = pd.read_csv("Wine.csv")

# Separating the dataset
X = pca_dataset.iloc[:,0:13].values

y = pca_dataset.iloc[:,-1].values

### Splitting the dataset
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=0)

### Feature Scaling
from sklearn.preprocessing import StandardScaler

x_standedScaler = StandardScaler()

x_train = x_standedScaler.fit_transform(x_train)

x_test = x_standedScaler.transform(x_test)

### Applying PCA
from sklearn.decomposition import PCA

pca_analysier = PCA(n_components= 2)

x_train = pca_analysier.fit_transform(x_train)

x_test = pca_analysier.transform(x_test)

explaind_variance = pca_analysier.explained_variance_ratio_

### Training the logistic model using linear regressor
from sklearn.linear_model import LogisticRegression

pca_log_classifier = LogisticRegression(random_state=0)

pca_log_classifier.fit(x_train,y_train)


### predicting the trainied model || Verification
y_pred_train = pca_log_classifier.predict(x_train)

### predicting the trainied model || Validation
y_pred_test = pca_log_classifier.predict(x_test)

### Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)

### visualisation || The data || Training Data
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

### visualisation || The data || Test Data
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

plt.title('Logistic Regression with PCA (Training set)')

plt.xlabel('PC1')

plt.ylabel('PC2')

plt.legend()

plt.show()

### Sample Dataset
<img width="1399" alt="pca_dataset" src="https://user-images.githubusercontent.com/32480274/50404946-3cf67700-07ae-11e9-999d-da975e7fd11a.png">


￼
### Explained variance 
* In this, there are extracted 13 independent variables.
* To plot in 2D, we are taking the first two principal components which are 37+19 =56 % of the variance.

<img width="125" alt="pca_analysis" src="https://user-images.githubusercontent.com/32480274/50404953-42ec5800-07ae-11e9-85aa-bcf1ddedcaee.png">
￼
### Prediction results.

    * Predicted using confusion matrix.
    * 0 - 0 => Correct prediction of the customer (1) who like the wine.
    * 1 - 1  => Correct prediction of the customer (2) who like the wine.
    * 2 - 2 => Correct prediction of the customer (3) who like the wine.
    * 0 - 1, 0-2 => Incorrect prediction of the customer (1) who didn't like the wine.
    * 1 - 0, 1-2 => Incorrect prediction of the customer (2) who didn't like the wine.
    * 2 - 0, 2-1 => Incorrect prediction of the customer (3) who didn't like the wine.
<img width="317" alt="pca_cm" src="https://user-images.githubusercontent.com/32480274/50404954-467fdf00-07ae-11e9-9104-93335b3dd78a.png">

### Visualization 

    * Blue and yellow Dots => Data used for Training (Training set)
    * Red part  and blue dot=>   customer (1) who like the wine
    * Green part and yellow dot => customer (2) who like the wine
    * Purple part and cyan dot => customer (3) who like the wine

<img width="1285" alt="pca_test_set" src="https://user-images.githubusercontent.com/32480274/50404958-4a136600-07ae-11e9-9d40-302e835b9eb6.png">

### Visualization 

    * Blue and yellow Dots => Data used for Test (Test set)
    * Red part  and blue dot=>   customer (1) who like the wine
    * Green part and yellow dot => customer (2) who like the wine
    * Purple part and cyan dot => customer (3) who like the wine
<img width="1265" alt="pca_test_set" src="https://user-images.githubusercontent.com/32480274/50404965-55ff2800-07ae-11e9-8673-2e485fa5aaf2.png">
￼



