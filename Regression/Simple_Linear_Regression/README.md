 # Simple Linear Regression 

* Simple linear regression is a statistical method. 
* It is the study of only one predictor variable.
* For summarize and study relationships between two variables.
* Dependent variable
    * It is the target variable (outcome).
    * The variable expected to change based on the manipulation of the independent variable.
* Independent variable
    * It is the input variable (predictor).		
    * The independent variable can be manipulated to get the different outcomes independent variable. 
* Example 
    * In this example, we are predicting salary based on the year of experience.
    * Dependent variable  => Years Of Experience
    * Independent variable => Salary


## Simple_linear_regression
import matplotlib.pyplot as plt # used for plot graphically

import pandas as pd # used for importing dataset and manage dataset

from sklearn.model_selection import train_test_split # used for splitting the dataset

from sklearn.linear_model import LinearRegression # used for creating a regressor

# Importing the dataset
S_dataset = pd.read_csv("Salary_Data.csv")

X = S_dataset.iloc[:,:-1].values

Y = S_dataset.iloc[:,-1].values

### Splitting_the_data_Test & Train
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size = 2/3,random_state = 0)

### creating the Simple Linear Regression
sLinearReg = LinearRegression()

sfit = sLinearReg.fit(x_train,y_train)

### predicting the data (Test)
y_pred_t = sfit.predict(x_train)

y_pred_s = sfit.predict(x_test)


### visualizing
plt.figure("Salary Predictor")

plt.scatter(x_train,y_train,color = 'red')

plt.scatter(x_test,y_test,color = 'green')

plt.plot(x_train,y_pred_t)

plt.title("Salary Predictor including actual Test Set")

plt.grid()

plt.xlabel("Years Of Experience")

plt.ylabel("Salary")

plt.show()

![salary_predictor_test_set](https://user-images.githubusercontent.com/32480274/50396931-414e7000-076d-11e9-938d-d90b461483bb.png)
￼
### Visualization 
    * Red Dots => Data used for training (Training set)
    * Green Dots =>  Data used for testing (Test set)
    * Blue Line => Predicted value using trained set 
