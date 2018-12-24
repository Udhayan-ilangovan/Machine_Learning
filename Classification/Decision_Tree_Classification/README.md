# Decision Tree Classification
* Decision Tree is to create a training model which can use to predict the value of target variables by learning decision rules learned from prior data(training dataset).
* It creates and organizes a series of questions and conditions in a tree structure.
* Every time it catches a response, a follow-up question is asked until a conclusion about the class label of the record is accomplished.
* In decision trees, for predicting a value, It starts from the root of the tree. 
* It compares the values of the root attribute with record’s attribute. 
* On the basis of comparison, It follows the branch corresponding to that value and jumps to the next node.
* Plan of assigning the attributes as root or internal node of the tree is done by applying the statistical method.
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

## Decision Tree Classification
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

### Feature Scaling || Feature Scaling is not needed but using feature scaling we can increase the performance (speed) of the process
from sklearn.preprocessing import StandardScaler

x_standardscaler = StandardScaler()

x_train = x_standardscaler.fit_transform(x_train)

x_test = x_standardscaler.transform(x_test)

### Training the model 
from sklearn.tree import DecisionTreeClassifier

dtc_classifier = DecisionTreeClassifier(criterion = "entropy" ,random_state=0)

dtc_classifier.fit(x_train,y_train)

### Predicting the Test set results
y_pred_test = dtc_classifier.predict(x_test)

### Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred_test)

### visualizing
### visualisation || The data || Training Data
from matplotlib.colors import ListedColormap

plt.figure("Decision Tree Classifier (Training set)")

x_set ,y_set = x_train,y_train

x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min() - 1, stop = x_set[:,0].max() + 1, step = 0.01 ),
                    np.arange(start = x_set[:,0].min() - 1, stop = x_set[:,1].max() + 1, step = 0.01))

plt.contourf(x1,x2,dtc_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.8, cmap = ListedColormap(('red', 'green')))

plt.xlim(x1.min(), x1.max())

plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('blue', 'yellow'))(i), label = j)

plt.title(' Decision Tree Classifier (Training set)')

plt.xlabel('Age')

plt.ylabel('Estimated Salary')

plt.legend()

plt.show()

### visualisation || The data || Test Data
from matplotlib.colors import ListedColormap

plt.figure("Decision Tree Classifier (Test set)")

x_set,y_set = x_test,y_test

x1,x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1,stop = x_set[:,0].max() + 1 , step = 0.01),
                    np.arange(start = x_set[:, 1].min() - 1,stop = x_set[:,1].max() + 1 , step = 0.01))

plt.contourf(x1,x2,dtc_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(x1.min(), x1.max())

plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('blue', 'yellow'))(i), label = j)

plt.title('Decision Tree Classifier (Test set)')

plt.xlabel('Age')

plt.ylabel('Estimated Salary')

plt.legend()

plt.show()

### Training Set

<img width="1318" alt="decision_tree_classifier_training_c" src="https://user-images.githubusercontent.com/32480274/50399779-381bce00-0782-11e9-8b61-808fad06dcce.png">

￼
### Visualization 
    * Blue and yellow Dots => Data used for training (Training set)
    * Red part  =>  Customer who hadn't purchased the car
    * Green part => Customer who purchased the car

### Test Set

<img width="1340" alt="decision_tree_classifier_test_c" src="https://user-images.githubusercontent.com/32480274/50399782-3eaa4580-0782-11e9-8524-fb4b7dc86dd2.png">

￼
### Visualization 
    * Blue and yellow Dots => Data used for Testing (Test set)
    * Red part  =>  Customer who won't purchase the car
    * Green part => Customer who would purchase the car

### Visualization Tree

import pydotplus
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz

dot_data = StringIO()
data_feature_names = [ 'Age', 'Salary']
export_graphviz(dtc_classifier, out_file=dot_data,  
                feature_names=data_feature_names,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())

graph.write_png('tree.png')

![tree](https://user-images.githubusercontent.com/32480274/50399819-8335e100-0782-11e9-922f-b66c85dd0c42.png)

