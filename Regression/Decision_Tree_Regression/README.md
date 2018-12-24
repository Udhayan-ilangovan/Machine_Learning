# Decision Tree regression 
* Decision Tree is to create a training model which can use to predict the value of target variables by learning decision rules learned from prior data(training dataset).
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
    * In this example, we are predicting the salary of the person in a company based on their position(posting).
    * Dependent variable  => Salary
    * Independent variable => Position
## Decision Tree
import numpy as np # used for mathematical operations

import matplotlib.pyplot as plt # used for plot graphically

import pandas as pd # used for importing dataset and manage dataset

D_dataset = pd.read_csv("Position_Salaries.csv")

### Creation of Dependent and independent variable
X = D_dataset.iloc[:,1:2].values

y = D_dataset.iloc[:,2:3].values

### creating the regressor / predictor / training the model
from  sklearn.tree import DecisionTreeRegressor

dec_regressor = DecisionTreeRegressor(random_state= 0)

dec_regressor.fit(X,y)

### For visualizing purpose
y_pred_train = dec_regressor.predict(X)

### Predicting given input / Manualy
y_pred_man = dec_regressor.predict(6.5)


### visualisation
plt.figure("Salary predictor based on position by Decision Tree High resolution")

x_grid = np.arange(min(X),max(X),0.001) #to visualise the data in higher resolution

x_grid = x_grid.reshape((len(x_grid),1))

plt.scatter(X,y,marker='x',color ='red') # plotting the train set data

plt.scatter(6.5,y_pred_man,marker='X',color='green') # plotting the test set data after predicting

plt.plot(x_grid,dec_regressor.predict(x_grid), color = 'blue')  # Drawing the prediction line 

plt.xticks(np.arange(1,11,0.5))

plt.yticks(np.arange(0,max(y),50000))

### Labeling
plt.ylabel("Position in the company",verticalalignment='baseline')

plt.xlabel("Salary",fontsize = 'larger')
# Enabling the grid
plt.grid()
# Displaying the model
plt.show()
<img width="1310" alt="decision_tree_reg" src="https://user-images.githubusercontent.com/32480274/50397349-17e31380-0770-11e9-9a95-ce2a9b6638fc.png">
￼

### Visualization 
    * Red Dots => Data used for training (Training set)
    * Blue Dots =>  Data used for testing (Test set)
    * Blue Line => Predicted value using trained set 
