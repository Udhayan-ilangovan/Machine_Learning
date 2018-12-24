# Bayes Theorem
* It works on conditional probability.
* Using the conditional probability, It can determine the probability of an event using its prior knowledge (Data).
* It predicts the membership probabilities for each class such as for the probability that given data point refers to a particular class.  
* The class with the highest probability is considered the most likely class. 
* Types of Naive Bayes Algorithm
    * Gaussian Naive Bayes
    * MultiNomial Naive Bayes
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

## Naive Bayes

import numpy as np # used for mathematical operations

import matplotlib.pyplot as plt # used for plot graphically

import pandas as pd # used for importing dataset and manage dataset

### Importing the dataset
svm_dataset = pd.read_csv("Social_Network_Ads.csv")

X = svm_dataset.iloc[:,2:4].values

y = svm_dataset.iloc[:,-1].values

### Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y,train_size = 3/4,random_state = 0)

### Feature Scaling
from sklearn.preprocessing import StandardScaler

x_standardscaler = StandardScaler()

x_train = x_standardscaler.fit_transform(x_train)

x_test = x_standardscaler.transform(x_test)

### Training the model 
from sklearn.naive_bayes import GaussianNB

nb_classifier = GaussianNB()

nb_classifier.fit(x_train,y_train)

### Predicting the Test set results
y_pred_test = nb_classifier.predict(x_test)

### Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred_test)

### visualizing
### visualisation || The data || Training Data
from matplotlib.colors import ListedColormap

plt.figure("Training Set")

x_set ,y_set = x_train,y_train

x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min() - 1, stop = x_set[:,0].max() + 1, step = 0.01 ),
                    np.arange(start = x_set[:,0].min() - 1, stop = x_set[:,1].max() + 1, step = 0.01))

plt.contourf(x1,x2,nb_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.8, cmap = ListedColormap(('red', 'green')))

plt.xlim(x1.min(), x1.max())

plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('blue', 'yellow'))(i), label = j)

plt.title(' Naive Baise Classifier (Training set)')

plt.xlabel('red (not bought)')

plt.ylabel('green (bought)')

plt.legend()

plt.show()

### visualisation || The data || Test Data
from matplotlib.colors import ListedColormap

plt.figure("Test Set")

x_set,y_set = x_test,y_test

x1,x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1,stop = x_set[:,0].max() + 1 , step = 0.01),
                    np.arange(start = x_set[:, 1].min() - 1,stop = x_set[:,1].max() + 1 , step = 0.01))

plt.contourf(x1,x2,nb_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(x1.min(), x1.max())

plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],

                c = ListedColormap(('blue', 'yellow'))(i), label = j)

plt.title('Naive Baise Classifier (Test set)')

plt.xlabel('red (not bought)')

plt.ylabel('green (bought)')

plt.legend()

plt.show()

### Training Set

<img width="634" alt="screenshot 2018-12-24 at 13 36 23" src="https://user-images.githubusercontent.com/32480274/50399605-ea529600-0780-11e9-9866-b1edda2f5070.png">
￼


### Visualization 

    * Blue and yellow Dots => Data used for training (Training set)
    * Red part  =>  Customer who hadn't purchased the car
    * Green part => Customer who purchased the car

### Test Set

￼<img width="634" alt="screenshot 2018-12-24 at 13 36 05" src="https://user-images.githubusercontent.com/32480274/50399601-deff6a80-0780-11e9-9392-db34777450af.png">

### Visualization 

    * Blue and yellow Dots => Data used for Testing (Test set)
    * Red part  =>  Customer who won't purchase the car
    * Green part => Customer who would purchase the car
