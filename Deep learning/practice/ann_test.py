# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

model = Sequential()

model.add(Conv2D(32,(3,3),input_shape = (28,28,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=1,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics = ['accuracy'])


model.fit_generator(x_train, 
              validation_data=x_test, 
              steps_per_epoch=50000, 
              epochs=5, 
              validation_steps=10000)

plt.matshow(x_train[0])