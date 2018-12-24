# SVR
* SVR uses a various optimization objective compared to others ( logistic, linear regression or neural networks) 
* The kernel determines how similar various features are with respect to each other, and thus grants weights to their corresponding cost functions. 
* The cost function involves using a kernel
    * such as linear, Gaussian, polynomial. 
* There are various properties (Kernels) associated with this cost function which gives a good solution by reducing the computations cost. 

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


## SVR
import numpy as np # importing for mathematical operation
import matplotlib.pyplot as plt # used for plot graphically
import pandas as pd # used for the importing dataset and manage dataset

### Importing the DataSet
S_dataset = pd.read_csv("Position_Salaries.csv")

### Creation of Dependent and independent variable
X = S_dataset.iloc[:,1:2].values
Y = S_dataset.iloc[:,2:3].values

### Seperatning the dataset Test and Train without shuffling
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.1,random_state=0,shuffle=False)

### Data prepocessing / Feature Scaling
from sklearn.preprocessing import StandardScaler

x_standedscaler = StandardScaler()

y_standedscaler = StandardScaler()

x_train = x_standedscaler.fit_transform(x_train)

y_train = y_standedscaler.fit_transform(y_train)

x_test = x_standedscaler.transform(x_test)

### SVR fitting dataset
from sklearn.svm import SVR
svr_Regressor = SVR(kernel = 'rbf')
svr_Regressor.fit(x_train,y_train)

### predicting training set to draw line
y_pred_train = svr_Regressor.predict(x_train)

### predicting test 
y_pred_test = svr_Regressor.predict(x_test)

print(y_pred_test)

y_pred_test_inverse = y_standedscaler.inverse_transform(y_pred_test)

print(y_pred_test_inverse)

### manual input: To predict 
x_in = 6.5

x_in = np.array([[x_in]])

dir_in = x_standedscaler.transform(x_in)

y_pred_test_dir_inp = svr_Regressor.predict(dir_in)

# Converting the value back into real formate 

y_pred_test_dir_inp_inverse = y_standedscaler.inverse_transform(y_pred_test_dir_inp)

print(y_pred_test_dir_inp_inverse)

### visualisation
plt.figure(1)
plt.scatter(x_train,y_train,marker='x',color ='red') # plotting the train set data
plt.scatter(x_test,y_pred_test,marker='X',color='green') # plotting the test set data after predicting
plt.scatter(dir_in,y_pred_test_dir_inp,marker='X',color='m')  # plotting the test set (manual input) data after predicting
plt.plot(x_train,y_pred_train, color = 'blue')  # Drawing the SVR line 
#### Labeling
plt.title("Salary predictor based on position by SVR")
plt.ylabel("Position in the company",verticalalignment='baseline')
plt.xlabel("Salary",fontsize = 'larger')
#### Enabling the grid
plt.grid()
#### Displaying the model
plt.show()

<img width="642" alt="svr" src="https://user-images.githubusercontent.com/32480274/50397089-44962b80-076e-11e9-89f5-9eb3615af049.png">

￼
### Visualization 
    * Red Dots => Data used for training (Training set)
    * Green Dots =>  Data used for testing (Test set)
    * Magenta Dots => Data used for testing (Test data - user)
    * Blue Line => Predicted value using trained set 
