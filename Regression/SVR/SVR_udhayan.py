# SVR
import numpy as np # importing for mathematical operation
import matplotlib.pyplot as plt # used for plot graphically
import pandas as pd # used for importing dataset and manage dataset

# Importing the DataSet
S_dataset = pd.read_csv("Position_Salaries.csv")

# Creation of Dependent and independent variable
X = S_dataset.iloc[:,1:2].values
Y = S_dataset.iloc[:,2:3].values

# Seperatning the dataset Test and Train without shuffling
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.1,random_state=0,shuffle=False)

# Data prepocessing / Feature Scaling
from sklearn.preprocessing import StandardScaler
x_standedscaler = StandardScaler()
y_standedscaler = StandardScaler()
x_train = x_standedscaler.fit_transform(x_train)
y_train = y_standedscaler.fit_transform(y_train)
x_test = x_standedscaler.transform(x_test)

# SVR fitting dataset
from sklearn.svm import SVR
svr_Regressor = SVR(kernel = 'rbf')
svr_Regressor.fit(x_train,y_train)

# predicting training set to draw line
y_pred_train = svr_Regressor.predict(x_train)

# predicting test 
y_pred_test = svr_Regressor.predict(x_test)
print(y_pred_test)
y_pred_test_inverse = y_standedscaler.inverse_transform(y_pred_test)
print(y_pred_test_inverse)

# manuel input: To predict 
x_in = 6.5
x_in = np.array([[x_in]])
dir_in = x_standedscaler.transform(x_in)
y_pred_test_dir_inp = svr_Regressor.predict(dir_in)

# Converting the value back into real formate 
y_pred_test_dir_inp_inverse = y_standedscaler.inverse_transform(y_pred_test_dir_inp)
print(y_pred_test_dir_inp_inverse)

# visualisation
plt.figure(1)
plt.scatter(x_train,y_train,marker='x',color ='red') # ploting the train set data
plt.scatter(x_test,y_pred_test,marker='X',color='green') # ploting the test set data after predicitng
plt.scatter(dir_in,y_pred_test_dir_inp,marker='X',color='m')  # ploting the test set (manuel input) data after predicitng
plt.plot(x_train,y_pred_train, color = 'blue')  # Drawing the SVR line 
# Labeling
plt.title("Salary predictor based on position by SVR")
plt.ylabel("Position in the company",verticalalignment='baseline')
plt.xlabel("Salary",fontsize = 'larger')
# Enabling the grid
plt.grid()
# Displaying the model
plt.show()

