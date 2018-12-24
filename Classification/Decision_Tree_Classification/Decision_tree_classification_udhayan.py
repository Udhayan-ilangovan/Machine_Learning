# Decision Tree Classification

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Importing the dataset
dt_dataset = pd.read_csv("Social_Network_Ads.csv")
X = dt_dataset.iloc[:,2:4].values
y = dt_dataset.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,train_size = 3/4,random_state = 0)

# Feature Scaling || Feature Scaling is not needed but using feature scaling we can increase the performance (speed) of the process
from sklearn.preprocessing import StandardScaler
x_standardscaler = StandardScaler()
x_train = x_standardscaler.fit_transform(x_train)
x_test = x_standardscaler.transform(x_test)

# Training the model 
from sklearn.tree import DecisionTreeClassifier
dtc_classifier = DecisionTreeClassifier(criterion = "entropy" ,random_state=0)
dtc_classifier.fit(x_train,y_train)

# Predicting the Test set results
y_pred_test = dtc_classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred_test)

# visualizing
from matplotlib.colors import ListedColormap
plt.figure("Decision Tree Classifier (Training set)")
x_set ,y_set = x_train,y_train
x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min() - 1, stop = x_set[:,0].max() + 1, step = 0.01 ),
                    np.arange(start = x_set[:,0].min() - 1, stop = x_set[:,1].max() + 1, step = 0.01))
plt.contourf(x1,x2,dtc_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.8, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('blue', 'yellow'))(i), label = j)
plt.title(' Decision Tree Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# visualisation || The data || Training Data
from matplotlib.colors import ListedColormap
plt.figure("Decision Tree Classifier (Test set)")
x_set,y_set = x_test,y_test
x1,x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1,stop = x_set[:,0].max() + 1 , step = 0.01),
                    np.arange(start = x_set[:, 1].min() - 1,stop = x_set[:,1].max() + 1 , step = 0.01))
plt.contourf(x1,x2,dtc_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('blue', 'yellow'))(i), label = j)
plt.title('Decision Tree Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()




import pydotplus
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz

dot_data = StringIO()
data_feature_names = [ 'Age', 'Salary']
export_graphviz(dtc_classifier, out_file=dot_data,  
                feature_names=data_feature_names,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())

graph.write_png('tree.png')



