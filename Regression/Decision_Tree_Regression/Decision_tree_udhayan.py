# Decision Tree
import numpy as np # used for mathemetical operations
import matplotlib.pyplot as plt # used for plot graphically
import pandas as pd # used for importing dataset and manage dataset

D_dataset = pd.read_csv("Position_Salaries.csv")

# Creation of Dependent and independent variable
X = D_dataset.iloc[:,1:2].values
y = D_dataset.iloc[:,2:3].values


#creating the regressor / predictor / training the model
from  sklearn.tree import DecisionTreeRegressor
dec_regressor = DecisionTreeRegressor(random_state= 0)
dec_regressor.fit(X,y)

# For visualizing purpose
y_pred_train = dec_regressor.predict(X)

# Predicting given input / Manualy
y_pred_man = dec_regressor.predict(6.5)


# visualisation
plt.figure("Salary predictor based on position by Decision Tree High resolution")
x_grid = np.arange(min(X),max(X),0.001) #to visualise the data in higher resolution
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(X,y,marker='x',color ='red') # ploting the train set data
plt.scatter(6.5,y_pred_man,marker='X',color='green') # ploting the test set data after predicitng
plt.plot(x_grid,dec_regressor.predict(x_grid), color = 'blue')  # Drawing the prediction line 
plt.xticks(np.arange(1,11,0.5))
plt.yticks(np.arange(0,max(y),50000))
# Labeling
plt.ylabel("Position in the company",verticalalignment='baseline')
plt.xlabel("Salary",fontsize = 'larger')
# Enabling the grid
plt.grid()
# Displaying the model
plt.show()