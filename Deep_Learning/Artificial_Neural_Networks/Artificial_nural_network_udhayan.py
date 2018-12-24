# Artificial neural network
# Importing the libraries
import pandas as pd # used for importing dataset and manage dataset
# Importing the dataset
ann_dataset = pd.read_csv('Churn_Modelling.csv')
X = ann_dataset.iloc[:, 3:13].values
y = ann_dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Bulding Artificial Nueral Networ using Keras
#import keras
from keras.models import Sequential
from keras.layers import Dense

#intialising ANN
ann_seq_classifier = Sequential()

# Intializing the input layer and  first hidden layer
ann_seq_classifier.add(Dense(6,kernel_initializer='uniform',activation='relu',input_dim =11)) # 6 is the output dim 

# Intializing the second layer 
ann_seq_classifier.add(Dense(6,kernel_initializer='uniform',activation='relu'))

# Intializing the third layer 
ann_seq_classifier.add(Dense(6,kernel_initializer='uniform',activation='relu'))

# Intializing the fourth layer 
ann_seq_classifier.add(Dense(6,kernel_initializer='uniform',activation='relu'))

# Intializing the output layer 

ann_seq_classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid')) # soft max if there is more than one two categories

# Compiling the Artificial Neural Network
ann_seq_classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics= ["accuracy"])
# Trainning the  ANN

ann_seq_classifier.fit(X_train, y_train, batch_size=10 , epochs=100) # nb_epoch gives higher result

# Predicting the Test set results
y_pred_test = ann_seq_classifier.predict(X_test)
y_pred_test_inv_tra = (y_pred_test > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_test_inv_tra)
accurcy = ((cm[0,0]+cm[1,1])/len(X_test))*100


# New prediction from dataset
ann_dataset_in = pd.read_csv('Churn_Modelling_in.csv')
X_in = ann_dataset_in.iloc[:, 3:13].values
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X_in[:, 1] = labelencoder_X_1.fit_transform(X_in[:, 1])
labelencoder_X_2 = LabelEncoder()
X_in[:, 2] = labelencoder_X_2.fit_transform(X_in[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X_in = onehotencoder.fit_transform(X_in).toarray()
X_in = X_in[:, 1:]
# Feature Scaling
X_in = sc.transform(X_in)
y_pred_in = ann_seq_classifier.predict(X_in)
y_pred_test_inv_tra_in = (y_pred_in > 0.5)
