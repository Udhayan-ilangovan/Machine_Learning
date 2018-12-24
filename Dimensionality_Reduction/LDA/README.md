# Linear Discriminant Analysis
* LDA computes the directions (linear discriminants) that will represent the axes that that maximize the separation between multiple classes.
* It is a dimensionality reduction algorithm.
* It is a supervised algorithm.
* Maximizing the component axes for class separation.
* Uses
    * Earth science
    * Biomedical studies
    * Face recognition
    * Marketing
    * Bankruptcy prediction
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
## Linear Discriminant Analysis

import numpy as np # used for mathematical operations

import matplotlib.pyplot as plt # used for plot graphically

import pandas as pd # used for importing dataset and manage dataset

### Importing the dataset
lda_dataset = pd.read_csv("Wine.csv")


### Separating the dataset
X = lda_dataset.iloc[:,0:13].values

y = lda_dataset.iloc[:,-1].values

### Splitting the dataset
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=0)

### Feature Scaling
from sklearn.preprocessing import StandardScaler

x_standedScaler = StandardScaler()

x_train = x_standedScaler.fit_transform(x_train)

x_test = x_standedScaler.transform(x_test)

### Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda_analysier = LinearDiscriminantAnalysis(n_components= 2)

x_train = lda_analysier.fit_transform(x_train,y_train)

x_test = lda_analysier.transform(x_test)

explaind_variance = lda_analysier.explained_variance_ratio_

### Training the logistic model using linear regressor
from sklearn.linear_model import LogisticRegression

lda_log_classifier = LogisticRegression(random_state=0)

lda_log_classifier.fit(x_train,y_train)


### predicting the trainied model || Verification
y_pred_train = lda_log_classifier.predict(x_train)

### predicting the trainied model || Validation
y_pred_test = lda_log_classifier.predict(x_test)


### Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)


### visualisation || The data || Training Data
from matplotlib.colors import ListedColormap

plt.figure("Logistic: Training Set  with LDA")

x_set,y_set = x_train,y_train

x1,x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1,stop = x_set[:,0].max() + 1 , step = 0.01),
                    np.arange(start = x_set[:, 1].min() - 1,stop = x_set[:,1].max() + 1 , step = 0.01))

plt.contourf(x1,x2,lda_log_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.8, cmap = ListedColormap(('red', 'green','magenta')))

plt.xlim(x1.min(), x1.max())

plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('blue', 'yellow','cyan'))(i), label = j)

plt.title('Logistic Regression  with LDA (Training set)')

plt.xlabel('LD1')

plt.ylabel('LD2')

plt.legend()

plt.show()

### visualisation || The data || Test Data
from matplotlib.colors import ListedColormap

plt.figure("Logistic: Test Set  with LDA")

x_set,y_set = x_test,y_test

x1,x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1,stop = x_set[:,0].max() + 1 , step = 0.01),
                    np.arange(start = x_set[:, 1].min() - 1,stop = x_set[:,1].max() + 1 , step = 0.01))

plt.contourf(x1,x2,lda_log_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.8, cmap = ListedColormap(('red', 'green','magenta')))

plt.xlim(x1.min(), x1.max())

plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('blue', 'yellow','cyan'))(i), label = j)

plt.title('Logistic Regression with LDA (Training set)')

plt.xlabel('LD1')

plt.ylabel('LD2')

plt.legend()

plt.show()

### Sample Dataset

<img width="1399" alt="pca_dataset" src="https://user-images.githubusercontent.com/32480274/50405020-079e5900-07af-11e9-93cb-22ea98f9fe0f.png">

￼
###Explained variance 

* Extracted 2 independent variables.

<img width="123" alt="lda_ev" src="https://user-images.githubusercontent.com/32480274/50405024-0a994980-07af-11e9-921a-54ae90f57626.png">
￼
### Prediction results.

    * Predicted using confusion matrix.
    * 0 - 0 => Correct prediction of the customer (1) who like the wine.
    * 1 - 1  => Correct prediction of the customer (2) who like the wine.
    * 2 - 2 => Correct prediction of the customer (3) who like the wine.
    * 0 - 1, 0-2 => Incorrect prediction of the customer (1) who didn't like the wine.
    * 1 - 0, 1-2 => Incorrect prediction of the customer (2) who didn't like the wine.
    * 2 - 0, 2-1 => Incorrect prediction of the customer (3) who didn't like the wine.
<img width="320" alt="lda_cm" src="https://user-images.githubusercontent.com/32480274/50405027-1258ee00-07af-11e9-99dd-26ed6a791666.png">
￼
### Training set

### Visualization 

    * Blue and yellow Dots => Data used for Training (Training set)
    * Red part  and blue dot=>   customer (1) who like the wine
    * Green part and yellow dot => customer (2) who like the wine
    * Magenta part and cyan dot => customer (3) who like the wine
<img width="1282" alt="lda_test_set" src="https://user-images.githubusercontent.com/32480274/50405030-171da200-07af-11e9-8c34-24d22a43c631.png">
￼
### Test Set

### Visualization 
    * Blue and yellow Dots => Data used for Test (Test set)
    * Red part  and blue dot=>   customer (1) who like the wine
    * Green part and yellow dot => customer (2) who like the wine
    * Magenta part and cyan dot => customer (3) who like the wine
<img width="1223" alt="lda_training_set" src="https://user-images.githubusercontent.com/32480274/50405036-213fa080-07af-11e9-9fbc-b4f2d394afd7.png">

￼


