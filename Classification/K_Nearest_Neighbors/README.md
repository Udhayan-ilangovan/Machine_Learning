# k-nearest neighbour classifier (KNN)
* The K-nearest neighbour classifier algorithms are to predict the target by finding the nearest neighbour data (class). 
* The closest class will be recognised using the distance measures like Euclidean distance.
* For Example
    * Predicting the credit score.
        * The process of calculating the credit is expensive.
        * To reduce the cost of predicting credit score, 
        * The customers with similar background details get a similar credit score.
        * Therefore, using previously calculated credit score data we can predict the credit scores of new customers.
* Dependent variable
    * It is the target variable (outcome).
    * The variable expected to change based on the manipulation of the independent variable.
* Independent variable
    * It is the input variable (predictor).		
    * The independent variable can be manipulated to get the different outcomes independent variable. 
* Example 
    * In this example, we are predicting the customer who will buy a car based on their salary and age.
    * Dependent variable  => Purchased the car (Yes || No)
    * Independent variable => Salary and age 
## k-nearest neighbour classifier (KNN)

import numpy as np # used for mathematical operations

import matplotlib.pyplot as plt # used for plot graphically

import pandas as pd # used for importing dataset and manage dataset

### Splitting the dataset
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size = 3/4,random_state = 0)

### Feature Scaling the dataset
from sklearn.preprocessing import StandardScaler

x_standerdscaler = StandardScaler()

x_train = x_standerdscaler.fit_transform(x_train)

x_test = x_standerdscaler.transform(x_test)

### Training the model 
from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier(n_neighbors = 5,metric ='minkowski',p = 2 )

knn_classifier.fit(x_train,y_train)

### Predicting using the trained model
y_predict_train = knn_classifier.predict(x_train)

y_predict_test = knn_classifier.predict(x_test)

### Evaluvating the predicted result
from sklearn.metrics import confusion_matrix

cm_tr = confusion_matrix(y_train,y_predict_train)

cm_te = confusion_matrix(y_test,y_predict_test)

print(cm_tr,"",cm_te)

### visualizing
### visualisation || The data || Training Data
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

# visualisation || The data || Test Data
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


### Training Set

<img width="634" alt="image" src="https://user-images.githubusercontent.com/32480274/50398406-a3ac6e00-0777-11e9-8757-67b40312a130.png">
￼
### Visualization 
    * Blue and yellow Dots => Data used for training (Training set)
    * Red part  =>  Customer who hadn't purchased the car
    * Green part => Customer who purchased the car

### Test Set
<img width="634" alt="image" src="https://user-images.githubusercontent.com/32480274/50398418-bde64c00-0777-11e9-9499-c906dd2158c3.png">
￼
### Visualization 
    * Blue and yellow Dots => Data used for Testing (Test set)
    * Red part  =>  Customer who won't purchase the car
    * Green part => Customer who would purchase the car
