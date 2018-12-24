# polynomial Regression

* polynomial regression is to transform linear model to better fit our non linear data.
* when you add degrees to your features in your model, you essentially turn a linear model into a spline surface or curve which then is said to be polynomial in nature.
* Polynomial regression fits a nonlinear relationship between the value of x and the corresponding conditional mean of y, denoted E(y |x),  and has been used to describe nonlinear phenomena
    * Example the growth rate of tissues
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

## Polynomial regression
import matplotlib.pyplot as plt # used for plot graphically
import pandas as pd # used for importing dataset and manage dataset

### Importing the dataset
p_dataset = pd.read_csv("Position_Salaries.csv")
X = p_dataset.iloc[:,1:2].values
Y= p_dataset.iloc[:,-1].values


### Splitting_the_data_Test & Train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size = 0.9,random_state = 5,shuffle=False) 

### polynomial variable Creation
from sklearn.preprocessing import PolynomialFeatures
x_polynomial = PolynomialFeatures(degree = 5)
x_polyValue = x_polynomial.fit_transform(x_train)
x_polynomial.fit(x_polyValue,y_train)

### Polinomial regression
from sklearn.linear_model import LinearRegression
p_linearRegressor = LinearRegression()
p_linearRegressor.fit(x_polyValue,y_train)


### Polinomial regression predict
y_pred_p = p_linearRegressor.predict(x_polynomial.fit_transform(6.5))

### visualizing
plt.figure("Salary Predictor Training Set Polynomial Predict")
plt.scatter(x_train,y_train,color = 'red')
plt.scatter(6.5,y_pred_p)
plt.plot(x_train,p_linearRegressor.predict(x_polynomial.fit_transform(x_train)))
plt.title("Salary Predictor Training Set Polynomial Predict")
plt.grid()
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.show()


<img width="640" alt="poly" src="https://user-images.githubusercontent.com/32480274/50396804-83c37d00-076c-11e9-97ef-2416dec840c0.png">
￼

###Visualization 

    * Red Dots => Data used for training (Training set)
    * Blue Dots =>  Data used for testing (Test set)
    * Blue Line => Predicted value using trained set 


