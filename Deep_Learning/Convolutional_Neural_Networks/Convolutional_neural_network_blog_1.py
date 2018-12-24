# Convolutional neural network

# Bulding Convolutional neural network
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising Convolutional neural network

cnn_classifier = Sequential()

# First step convolutional

cnn_classifier.add(Convolution2D(32,3,3,input_shape = (64,64,3,), activation = 'relu'))

# Second step polling

cnn_classifier.add(MaxPooling2D(pool_size=(2,2)))

# Third step flattening

cnn_classifier.add(Flatten())

# Initialising Artificial neural network

cnn_classifier.add(Dense(output_dim = 128 , activation= "relu"))
cnn_classifier.add(Dense(output_dim = 1 , activation= "sigmoid"))

# Compiling the Convolutional neural network
cnn_classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])


# Fitting the CNN to images
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

# testing the image apart from dataset
import numpy as np
from PIL import Image as saveimage
from keras.preprocessing import image
test_image_p = saveimage.open('dataset/realtime_predict/img_{}.jpg'.format(1))
test_image = image.load_img('dataset/realtime_predict/img_{}.jpg'.format(1), target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn_classifier.predict(test_image)
training_set.class_indices
test_image_p.save("output.jpg")
if result[0][0] == 1:
    prediction = 'dog'
    test_image_p.save('dataset/realtime_predict/img_dog_{}.jpg'.format(1))
else:
    prediction = 'cat'
    test_image_p.save('dataset/realtime_predict/img_cat_{}.jpg'.format(1))
print(prediction)   



# Saving model to JSON
from keras.models import model_from_json
model_json = cnn_classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
cnn_classifier.save_weights("model.h5")
print("Saved model to disk")
 

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")