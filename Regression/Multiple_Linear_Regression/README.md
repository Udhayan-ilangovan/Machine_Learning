# Multi Linear Regression

* Multiple linear regression is a statistical technique that uses several explanatory variables to predict the outcome of a response variable
* The difference between simple linear regression and multiple linear regression is that multiple linear regression has more than one independent variables, whereas simple linear regression has only 1 independent variable
* Dependent variable
    * It is the target variable (outcome).
    * The variable expected to change based on the manipulation of the independent variable.
* Independent variable
    * It is the input variable (predictor).		
    * The independent variable can be manipulated to get the different outcomes independent variable. 
* Example 
    * In this example, we are predicting the profit of a company based on their investment in each department (Administration, Geography, Marketing, R&D)
    * Dependent variable  => Profit
    * Independent variable => (Administration, Geography, Marketing, R&D)


## Multi_linear_regression
import numpy as np # for mathematical operations

import pandas as pd # for importing the dataset and to manage the dataset

### importing the dataset
m_dataset = pd.read_csv("50_Startups.csv")

### Creating X (matrix) and Y (vector) variable
X = m_dataset.iloc[:,:-1].values

Y = m_dataset.iloc[:,-1].values

### Changing the categorical variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder_A = LabelEncoder()

X[:,3]=labelEncoder_A.fit_transform(X[:,3])

oneHotEncoder_A = OneHotEncoder(categorical_features=[3])

X = oneHotEncoder_A.fit_transform(X).toarray() # ML (python) dosent konw its a categorical Variable

# Avoiding the dummy variable
X = X[:,1:]

### Analysing the independent variables to get the best model 
import statsmodels.formula.api as sm

X = np.append(arr = np.ones((50,1)).astype(int),values = X,axis=1)

def best_opt_model(X_col):

    x_opt = X[:,X_col]
    
    mLinearRegressor_ols = sm.OLS(endog=Y,exog=x_opt).fit()
    
    mSummary = mLinearRegressor_ols.summary()
    
    print(mSummary)
    
"""removing independent variable one by one based on the Pvalue (pvalue should be less than 0.05)"""

# X_col = [0,1,2,3,4,5] 
# X_col = [0,1,3,4,5] 
# X_col = [0,3,4,5]

X_col = [0,3,5] # we are keeping this variable even the p value is greater than 5% because  the Adj R-squared is high for this model

# X_col = [0,3] # removing independent variable due to the fall of value in Adj R-squared

best_opt_model(X_col)

X_T_opt = X[:,[0,3,5]]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X_T_opt,Y,train_size = 0.8,random_state = 0)


### multi Liner Regression trainning

from sklearn.linear_model import LinearRegression

mLinearRegressor = LinearRegression()

mfit = mLinearRegressor.fit(x_train,y_train)

### predicting the data (Test)

y_pred_m_opt = mfit.predict(x_train)

### In this model [0, 3 ,5],  5 has a P value of 0.06 it is very close to 0.05 so we can remove the independent variable 5 and we can check the Adj R-squared

<img width="587" alt="x_opt_0_3_5" src="https://user-images.githubusercontent.com/32480274/50396965-78bd1c80-076d-11e9-9676-a5fa6e5801c5.png">
￼

### Even [0,3] Has a P Value less then 0.05, we are removing the independent variable due to the fall of value in Adj R-squared

<img width="693" alt="x_opt_0_3" src="https://user-images.githubusercontent.com/32480274/50396970-7e1a6700-076d-11e9-90cc-f497a0cff88d.png">
￼
