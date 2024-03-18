# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
The aim  of this project is to develop a convolutional deep neural network (CNN) capable of accurately classifying handwritten digits extracted from scanned images.
The CNN model should moreover classify handwritten digits apart from those given in dataset.
### MNIST DATASET
It stands for Modified National Institute of Standards and Technology database. The dataset consists of a large collection of 28x28 pixel grayscale images of handwritten digits (0 through 9). It is widely used for training and testing various machine learning algorithms, particularly in the context of image classification tasks.
## Neural Network Model

<img width="591" alt="image" src="https://github.com/Shavedha/mnist-classification/assets/93427376/848f1f79-ccc1-45c4-9c42-f67c789ec799">

## DESIGN STEPS
1. Import necessary libraries
2. Load the MNIST dataset and preprocess the same using onehot encoder
3. Build a tensorflow keras model with input layer, convolutional layer, MaxPool layer and dense layers.
4. Compile and fit the model with the training dataset.
5. Evaluate the model by plotting training and validation metrics, confusion matrix and classification report.
6. Test the trained model by giving handwritten digit as an input and check its accuracy.

## PROGRAM

### Name: Y SHAVEDHA 
### Register Number: 212221230095
```PYTHON
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[0]
single_image.shape
plt.imshow(single_image)
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0                  #for changing colour image to gray scale image
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
y_train_onehot.shape
y_test_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)               #reshaping the ip values with h,w,depth
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)                

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))  # shape of the image
model.add(layers.Conv2D(filters=45,kernel_size=(7,7),activation="relu"))
model.add(layers.MaxPool2D(pool_size=(2,2)))     #reduces the size of the image. -> Here 2,2 will reduce the image size into half
model.add(layers.Flatten())
model.add(layers.Dense(12,activation="relu"))
model.add(layers.Dense(10,activation="relu"))
model.add(layers.Dense(10,activation="softmax"))    
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))
img = image.load_img('img4.jpg')
type(img)
img = image.load_img('/content/img4.jpg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)
plt.imshow(img_28_gray_scaled)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
<img width="405" alt="image" src="https://github.com/Shavedha/mnist-classification/assets/93427376/e23d63c9-ead4-484b-9f64-8b6e4029d3f6">

<img width="409" alt="image" src="https://github.com/Shavedha/mnist-classification/assets/93427376/a5c64f62-85dd-415e-bb37-8b86fce642a7">


### Classification Report

<img width="329" alt="image" src="https://github.com/Shavedha/mnist-classification/assets/93427376/8c8d9ee2-0aa8-449d-acf7-8f37a42c864a">


### Confusion Matrix

<img width="301" alt="image" src="https://github.com/Shavedha/mnist-classification/assets/93427376/a74c4ccb-2b43-4da1-a7ac-713412929eff">



### New Sample Data Prediction

<img width="302" alt="image" src="https://github.com/Shavedha/mnist-classification/assets/93427376/02ec3005-83e0-4393-a5cf-e11b382d66f6">

<img width="299" alt="image" src="https://github.com/Shavedha/mnist-classification/assets/93427376/7af8122d-e07e-4eaa-828d-76890ac3e154">

## RESULT
Thus a convolutional deep neural network for digit classification is developed successfully.
