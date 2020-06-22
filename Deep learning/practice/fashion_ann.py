# -*- coding: utf-8 -*-

import numpy as np
import keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# normilize the dataset
train_images = train_images / 255
test_images = test_images / 255

# reshape the dataset into 2d vector to give input for NN 28*28 = 784
train_images = train_images.reshape((-1,784))
test_images = test_images.reshape((-1,784))

#neural network axcept the 10 dim vector for labels then
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(62,init='uniform',activation='relu',input_dim=(784)))
classifier.add(Dense(62,init='uniform',activation='relu'))
classifier.add(Dense(10,init='uniform',activation='sigmoid'))


classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
  

classifier.fit(train_images,
               train_labels,
               batch_size=32,
               epochs=10)

# Evaluate the model
classifier.evaluate(test_images,test_labels)


prediction = classifier.predict(test_images[5:10])
print(np.argmax(prediction,axis=1))
print(test_labels[5:10])

# ploting the first 5 test images
import matplotlib.pyplot as plt

for i in range(5,10):
    first_image = test_images[i]
    first_image = np.array(first_image,dtype='float')
    pixels = first_image.reshape((28,28))
    plt.imshow(pixels)
    plt.show()
    
    
    
    