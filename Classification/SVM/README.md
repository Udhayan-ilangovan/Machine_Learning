# Support Vector Machine Kernel (Non-Linear)
* In this KVM(Non-linear) data are not linearly separable in a p-dimensional(finite) space. 
* Therefore, by mapping the p-dimensional space into a much higher dimensional space. We can draw customized /non-linear hyperplanes using different Kernels.
* Kernels
    * Polynomial (homogeneous) Kernel
    * Polynomial(non-homogeneous) Kernel
    * Radial Basis Function Kernel
* For Example
    * Text Categorization
    * Handwritten digit recognition
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

## Support Vector Machine Kernel (Non-Linear)
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
from sklearn.svm import SVC

svm_classifier = SVC(kernel = 'rbf',gamma=1,random_state = 0) # rbf is good kernel for this model || increasing the gamma will vary in prediction by reducing the boundries based on training set    

"#"svm_classifier = SVC(kernel = 'poly',degree=3,random_state = 0) # (Based on the data set used) Here For this kernel poly degree 3 is better if degree is increased it i\ties to over fit the data 

svm_classifier.fit(x_train,y_train)

### Predicting the Test set results

y_pred_test = svm_classifier.predict(x_test)

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

plt.contourf(x1,x2,svm_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.8, cmap = ListedColormap(('red', 'green')))

plt.xlim(x1.min(), x1.max())

plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):
    
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('blue', 'yellow'))(i), label = j)

plt.title(' SVM Classifier (Training set)')

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

plt.contourf(x1,x2,svm_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(x1.min(), x1.max())

plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):
    
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('blue', 'yellow'))(i), label = j)

plt.title('SVM Classifier (Test set)')

plt.xlabel('red (not bought)')

plt.ylabel('green (bought)')

plt.legend()

plt.show()



### Training Set

<img width="634" alt="image" src="https://user-images.githubusercontent.com/32480274/50398733-33531c00-077a-11e9-9597-cc3fa6ca052b.png">
￼
### Visualization 

    * Blue and yellow Dots => Data used for training (Training set)
    * Red part  =>  Customer who hadn't purchased the car
    * Green part => Customer who purchased the car

### Test Set

<img width="634" alt="image" src="https://user-images.githubusercontent.com/32480274/50398741-4239ce80-077a-11e9-993d-372377578d8c.png">

### Visualization 

    * Blue and yellow Dots => Data used for Testing (Test set)
    * Red part  =>  Customer who won't purchase the car
    * Green part => Customer who would purchase the car
