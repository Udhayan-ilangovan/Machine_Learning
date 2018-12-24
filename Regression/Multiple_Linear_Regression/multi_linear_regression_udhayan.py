# Multi_linear_regression
import numpy as np # for mathematical operations
import pandas as pd # for importing the dataset and to manage the dataset

# importing the dataset
m_dataset = pd.read_csv("50_Startups.csv")

# Creating X (matrix) and Y (vector) variable
X = m_dataset.iloc[:,:-1].values
Y = m_dataset.iloc[:,-1].values

# Changing the categorical variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_A = LabelEncoder()
X[:,3]=labelEncoder_A.fit_transform(X[:,3])
oneHotEncoder_A = OneHotEncoder(categorical_features=[3])
X = oneHotEncoder_A.fit_transform(X).toarray() # ML (python) dosent konws its categorical Variable
# Avoiding the dummy variable
X = X[:,1:]

# Analysing the independent variables to get the best model 
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


# multi Liner Regression trainning
from sklearn.linear_model import LinearRegression
mLinearRegressor = LinearRegression()
mfit = mLinearRegressor.fit(x_train,y_train)

# predicting the data (Test)
y_pred_m_opt = mfit.predict(x_train)

