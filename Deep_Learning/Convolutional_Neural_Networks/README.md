# Convolutional Neural Network
* CNN has extensive applications in image recognition and classification, recommender systems and NLP.
* CNN is a category of Neural Networks.
* CNN is made up of neurons with weights and biases. 
* Every neuron receives various inputs and takes a weighted sum over them then passes it by activation function to present an output. 
* The whole network has similar methods like neural networks.

* The primary purpose of Convolution is to extract features from the input image.
* The Convolution layer is the core layer of a convolutional network that does largest and complex part of the computational.

* CNN has two components
    * The Feature extraction part.
    * The Classification part.
* The Feature extraction part
    * A series of convolutions and pooling operations are performed to detect the features.
    * For example 
    * In this part, it will identify the dog's features like four legs, shape of the ears, hair structure, etc.
* The Classification part
    * It is the fully connected layer which is a classifier.
    * Extracted features are the input for this layer. 
    * They will give a probability to predict or classify the input.

* Steps in CNN
    * Convolution
    * Input (image)
    * Feature Detector (Filtering)
    * Feature Map
    *  Apply the ReLu 
        * Anyone of this functions can be applied based on their needs
        * Rectified function 
        *  Sigmoid function
* Pooling
    * Anyone of this pooling can be applied based on their needs
        * Max pooling
        * Mean polling
        * Sum pooling
* Flattening
* Full connection
    * Input layer
    * Fully connect Layer(In ANN it is called Hidden layer)
    * Output Layer

* Dependent variable
    * It is the target variable (outcome).
    * The variable expected to change based on the manipulation of the independent variable.
* Independent variable
    * It is the input variable (predictor).		
    * The independent variable can be manipulated to get the different outcomes independent variable. 
* Example
    * We have 10000 sample data of dog and cat.
    * We need to classify the cat and dog.
    * Dependent variable  => Name of the animal (cat || dog).
    * Independent variable => input images.

## Building Convolutional Neural networks
from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

### Initialising Convolutional neural network

cnn_classifier = Sequential()

### First step convolutional

cnn_classifier.add(Convolution2D(32,3,3,input_shape = (64,64,3,), activation = 'relu'))

### Second step polling

cnn_classifier.add(MaxPooling2D(pool_size=(2,2)))

### Third step flattening

cnn_classifier.add(Flatten())

### Initialising Artificial neural network

cnn_classifier.add(Dense(output_dim = 128 , activation= "relu"))

cnn_classifier.add(Dense(output_dim = 1 , activation= "sigmoid"))

### Compiling the Convolutional neural network
cnn_classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])


### Fitting the CNN to images
from keras.preprocessing.image import ImageDataGenerator
 
train_datagen = ImageDataGenerator(

        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
        )

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(

        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(

        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

cnn_classifier.fit_generator(

        training_set,
        steps_per_epoch=8000,
        epochs=25,
        max_queue_size=20,
        workers=40,
        use_multiprocessing=True,
        validation_data=test_set,
        validation_steps=2000)

### testing the image apart from dataset

path, dirs, files = next(os.walk("dataset/realtime_predict/"))

file_count = len(files)

import numpy as np
for i in range(1,(file_count+1)):

    from PIL import Image as saveimage
    from keras.preprocessing import image
    test_image_p = saveimage.open('dataset/realtime_predict/img_{}.jpg'.format(i))
    test_image = image.load_img('dataset/realtime_predict/img_{}.jpg'.format(i), target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn_classifier.predict(test_image)
    training_set.class_indices
    test_image_p.save("output.jpg")
    if result[0][0] == 1:

        prediction = 'dog'
        test_image_p.save('dataset/RT_output/img_dog_{}.jpg'.format(i))
    else:

        prediction = 'cat'
        test_image_p.save('dataset/RT_output/img_cat_{}.jpg'.format(i))
    print(prediction)   

* Epoch
    * 25 Epochs are used to update the weights 
* Images
    * 8000 images found in the training set
    * 2000 images found in the test set
* Accuracy
    * Final accuracy rate is 75 %
<img width="856" alt="cnn_epoch" src="https://user-images.githubusercontent.com/32480274/50404583-f3a42880-07a9-11e9-9d6e-df305226bfc5.png">
￼

* Dataset folder
    * Test set
        * Cats
        * Dogs
    * Training Set
        * Cats
        * Dogs
    * Realtime prediction
        * Input => realtime_predict
        * Output => RT_output
<img width="1167" alt="cnn_dataset" src="https://user-images.githubusercontent.com/32480274/50404586-fb63cd00-07a9-11e9-92d5-36e0f95d0cef.png">
￼




