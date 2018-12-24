# Simple_linear_regression
import matplotlib.pyplot as plt # used for plot graphically
import pandas as pd # used for importing dataset and manage dataset
from sklearn.model_selection import train_test_split # used for splitting the dataset
from sklearn.linear_model import LinearRegression # used for creating a regressor

# Importing the dataset
S_dataset = pd.read_csv("Salary_Data.csv")
X = S_dataset.iloc[:,:-1].values
Y = S_dataset.iloc[:,-1].values

# Splitting_the_data_Test & Train
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size = 2/3,random_state = 0)

# creating the Simple Linear Regression
sLinearReg = LinearRegression()
sfit = sLinearReg.fit(x_train,y_train)

# predicting the data (Test)
y_pred_t = sfit.predict(x_train)
y_pred_s = sfit.predict(x_test)


# visualizing
#Salary Predictor Test Set
plt.figure("Salary Predictor")
plt.scatter(x_train,y_train,color = 'red')
plt.scatter(x_test,y_test,color = 'green')
plt.plot(x_train,y_pred_t)
plt.title("Salary Predictor including actual Test Set")
plt.grid()
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.show()
