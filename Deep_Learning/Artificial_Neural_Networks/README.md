# Artificial neural networks
* Artificial neural networks (ANNs)  is to simulate the biological neural networks which are the densely interconnected neurons in the animal brain. 
* This ANN learns to perform tasks, recognize patterns and make decisions by recognising examples (samples || dataset). 
* The amazing thing is it can learn explicitly without being programmed with any specific rules.
* It can learn or solve problems through failures and trial. 
* It is a branch in machine learning.
* There are three layers in the ANN.
    * Input Layer
    * Hidden Layer
    * Output Layer
* Input Layer
    * To input the data initially into the neural network for further processing by succeeding layers of artificial neurons.
* Hidden Layer 
    * It holds the majority of the artificial brain.
    * It is a layer between input layers and output layers,
    * It generates an activation function using a set of weighted inputs.
    * Connections between the units are denoted by a number called a weight.
* output layer:
    *  The final layer of the neural network
    *  It delivers the outputs for the given program.
* Dependent variable
    * It is the target variable (outcome).
    * The variable expected to change based on the manipulation of the independent variable.
* Independent variable
    * It is the input variable (predictor).		
    * The independent variable can be manipulated to get the different outcomes independent variable. 
* Example
    * We have 10000 sample data of bank customers.
    * We need to predict the people who are going to leave the bank in the future.
    * Dependent variable  => People who left the bank  (Yes || No).
    * Independent variable => credit score, Location, gender, age, tenure, balance, number of products, credit card, frequent transactions, estimated salary.

## Artificial neural network
### Importing the libraries
import pandas as pd # used for importing dataset and manage dataset
### Importing the dataset
ann_dataset = pd.read_csv('Churn_Modelling.csv')

X = ann_dataset.iloc[:, 3:13].values

y = ann_dataset.iloc[:, 13].values

### Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()

X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()

X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])

X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]

### Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

### Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

### Building Artificial Neural Network using Keras
‘#’import keras
from keras.models import Sequential

from keras.layers import Dense

###intialising ANN
ann_seq_classifier = Sequential()

### Intializing the input layer and  first hidden layer
ann_seq_classifier.add(Dense(6,kernel_initializer='uniform',activation='relu',input_dim =11))

### Initializing the second layer 

ann_seq_classifier.add(Dense(6,kernel_initializer='uniform',activation='relu'))

### Initializing the output layer 

ann_seq_classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))

### Compiling the Artificial Neural Network

ann_seq_classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics= ["accuracy"])

### Trainning the  ANN

ann_seq_classifier.fit(X_train, y_train, batch_size=10 , epochs=100) # nb_epoch gives higher result

### Predicting the Test set results
y_pred_test = ann_seq_classifier.predict(X_test)

y_pred_test_inv_tra = (y_pred_test > 0.5)

### Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test_inv_tra)

accurcy = ((cm[0,0]+cm[1,1])/len(X_test))*100

* Sample Dataset
<img width="1238" alt="ann_dataset" src="https://user-images.githubusercontent.com/32480274/50404710-43372400-07ab-11e9-82b3-68c1d89fc27e.png">
￼
* The accuracy percentage of correct prediction.

<img width="428" alt="ann_acc" src="https://user-images.githubusercontent.com/32480274/50404712-4b8f5f00-07ab-11e9-9ae2-7b42c8c020ab.png">
￼
* Prediction result.
    * Predicted using confusion matrix.
    * 0 - 0 => Correct prediction of the People who didn't leave the bank.
    * 1 - 1 => Correct prediction of the People who left the bank.
    * 0 - 1 => Incorrect prediction of the People who didn't leave the bank.
    * 1 - 0 => Incorrect prediction of the People who left the bank.
<img width="220" alt="ann_cm" src="https://user-images.githubusercontent.com/32480274/50404714-51854000-07ab-11e9-9537-367419aea2f9.png">
￼


