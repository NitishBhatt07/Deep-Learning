# -*- coding: utf-8 -*-
 #predicting whether the house price is above or below median value.
import pandas as pd
import numpy as np

dataset = pd.read_csv("housepricedata.csv")

dataset_X_train = dataset.iloc[:,:10].values
dataset_y_train = dataset.iloc[:,10].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
scaled_X_train = sc.fit_transform(dataset_X_train)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(
        scaled_X_train,dataset_y_train,test_size=0.2,random_state=0)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(output_dim=32,init='uniform',activation='relu',input_dim=10))
model.add(Dense(output_dim=32,init='uniform',activation='relu'))
model.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=10,nb_epoch=100)

y_pred =model.predict(X_test)
y_pred = (y_pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

model.evaluate(X_test, y_test)
