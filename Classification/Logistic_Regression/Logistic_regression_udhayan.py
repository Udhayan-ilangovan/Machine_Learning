# Logistic Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
lr_dataset = pd.read_csv("Social_Network_Ads.csv")

# Separating the dataset
X = lr_dataset.iloc[:,[2,3]].values
Y = lr_dataset.iloc[:,-1].values

# Splitting the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=3/4,random_state=0)

# Feature Scaling
x_test1 = x_test
from sklearn.preprocessing import StandardScaler
x_standedScaler = StandardScaler()
x_train = x_standedScaler.fit_transform(x_train)
x_test = x_standedScaler.transform(x_test)

# Training the logistic model using linear regressor
from sklearn.linear_model import LogisticRegression
log_classifier = LogisticRegression(random_state=0)
log_classifier.fit(x_train,y_train)

# predicting the trainied model || Verification
y_pred_train = log_classifier.predict(x_train)

# predicting the trainied model || Validation
y_pred_test = log_classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_test)

# visualisation || The data || Training Data
from matplotlib.colors import ListedColormap
plt.figure("Logistic: Training Set")
x_set,y_set = x_train,y_train
x1,x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1,stop = x_set[:,0].max() + 1 , step = 0.01),
                    np.arange(start = x_set[:, 1].min() - 1,stop = x_set[:,1].max() + 1 , step = 0.01))
plt.contourf(x1,x2,log_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.8, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('blue', 'yellow'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('red (not bought)')
plt.ylabel('green (bought)')
plt.legend()
plt.show()

# visualisation || The data || Test Data
from matplotlib.colors import ListedColormap
plt.figure("Logistic: Test Set")
x_set,y_set = x_test,y_test
x1,x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1,stop = x_set[:,0].max() + 1 , step = 0.01),
                    np.arange(start = x_set[:, 1].min() - 1,stop = x_set[:,1].max() + 1 , step = 0.01))
plt.contourf(x1,x2,log_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.8, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('blue', 'yellow'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('red (not bought)')
plt.ylabel('green (bought)')
plt.legend()
plt.show()