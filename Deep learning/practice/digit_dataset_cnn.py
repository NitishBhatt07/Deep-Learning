# -*- coding: utf-8 -*-

import numpy as np
from keras.datasets import mnist

# example of loading the mnist dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape(train_images.shape[0],28,28,1)
test_images = test_images.reshape(10000,28,28,1)

train_images = train_images.astype('float')
test_images = test_images.astype('float')

train_images = train_images/255
test_images = test_images/255

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout

model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(32,activation='relu',init='uniform'))
model.add(Dense(10,activation='sigmoid'))

model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])



from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


model.fit(train_images,train_labels,batch_size=32,epochs=10)

model.evaluate(test_images,test_labels)












