# -*- coding: utf-8 -*-

import numpy as np
from keras.datasets import mnist

# example of loading the mnist dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# normilize the images 
train_images = train_images/255
test_images = test_images/255

# Flatten the images .Flatten each 28*28 image into 784 (28*28) dim vector
# to pass into the neural network
train_images = train_images.reshape((-1,784))
test_images = test_images.reshape((-1,784))

#neural network axcept the 10 dim vector for labels then
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(32,init='uniform',activation='relu',input_dim=(784)))
classifier.add(Dense(32,init='uniform',activation='relu'))
classifier.add(Dense(10,activation='softmax'))

classifier.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics = ['accuracy']
        )

classifier.fit(
        train_images,
        train_labels,
        batch_size=32,
        epochs = 10
        )

# Evaluate the model
classifier.evaluate(test_images,test_labels)

# predict the test data
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


# save the model
classifier.save_weights('model.h5')







