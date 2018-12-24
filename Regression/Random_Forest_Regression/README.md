# Random forest algorithm
* Random forest algorithm creates the forest with a number of decision tree algorithms.
* The higher the number of trees in the forest gives the high accuracy results.
    * Random forest takes the test dataset and uses the rules of each randomly created decision tree to predict the outcome and stores the predicted outcome.
    * Calculates the votes for each predicted outcome.
    * Consider the high voted predicted target as the final prediction from the random forest algorithm.

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

## Random Forest
import numpy as np # used for mathematical operations

import matplotlib.pyplot as plt # used for plot graphically

import pandas as pd # used for importing dataset and manage dataset

R_dataset = pd.read_csv("Position_Salaries.csv");

### Creation of Dependent and independent variable

X = R_dataset.iloc[:,1:2].values

y = R_dataset.iloc[:,2:3].values

### creating the regressor || predictor / training the model

from  sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor(n_estimators= 500, random_state= 0)

rf_regressor.fit(X,y)

### For visualizing purpose
y_pred_train = rf_regressor.predict(X)

### Predicting given input / Manualy
y_pred_man = rf_regressor.predict(6.5)

print(y_pred_man)

### visualisation

plt.figure("Salary predictor based on position by Decision Tree High resolution")

x_grid = np.arange(min(X),max(X),0.001)

x_grid = x_grid.reshape((len(x_grid),1))

plt.scatter(X,y,marker='x',color ='red') # plotting the train set data

plt.scatter(6.5,y_pred_man,marker='X',color='green') # plotting the test set data after predicting 

plt.plot(x_grid,rf_regressor.predict(x_grid), color = 'blue')  # Drawing the Random Forest line 

plt.xticks(np.arange(1,11,0.5))

plt.yticks(np.arange(0,max(y),50000))
#### Labeling
plt.title("Salary predictor based on position by Decision Tree High resolution")

plt.ylabel("Position in the company",verticalalignment='baseline')

plt.xlabel("Salary",fontsize = 'larger')

#### Enabling the grid
plt.grid()
#### Displaying the model
plt.show()

<img width="1353" alt="random_forest" src="https://user-images.githubusercontent.com/32480274/50397415-a5befe80-0770-11e9-9599-21d76c644082.png">

￼

### Visualization 
    * Red Dots => Data used for training (Training set)
    * Blue Dots =>  Data used for testing (Test set)
    * Blue Line => Predicted value using trained set 

