# Random Forest
import numpy as np # used for mathemetical operations
import matplotlib.pyplot as plt # used for plot graphically
import pandas as pd # used for importing dataset and manage dataset
R_dataset = pd.read_csv("Position_Salaries.csv");

# Creation of Dependent and independent variable
X = R_dataset.iloc[:,1:2].values
y = R_dataset.iloc[:,2:3].values

#creating the regressor || predictor / training the model
from  sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators= 500, random_state= 0)
rf_regressor.fit(X,y)

# For visualizing purpose
y_pred_train = rf_regressor.predict(X)

# Predicting given input / Manualy
y_pred_man = rf_regressor.predict(6.5)
print(y_pred_man)

# visualisation
plt.figure("Salary predictor based on position by Decision Tree High resolution")
x_grid = np.arange(min(X),max(X),0.001)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(X,y,marker='x',color ='red') # ploting the train set data
plt.scatter(6.5,y_pred_man,marker='X',color='green') # ploting the test set data after predicitng
plt.plot(x_grid,rf_regressor.predict(x_grid), color = 'blue')  # Drawing the Random Forest line 
plt.xticks(np.arange(1,11,0.5))
plt.yticks(np.arange(0,max(y),50000))
# Labeling
#plt.title("Salary predictor based on position by Decision Tree High resolution")
plt.ylabel("Position in the company",verticalalignment='baseline')
plt.xlabel("Salary",fontsize = 'larger')
# Enabling the grid
plt.grid()
# Displaying the model
plt.show()