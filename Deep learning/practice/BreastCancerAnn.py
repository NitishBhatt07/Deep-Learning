# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

dataset = pd.read_csv('BreastCancer.csv')
dataset = dataset.drop(['id','Unnamed: 32'],axis=1)

data = {'M':1,'B':0}
dataset.diagnosis = [data[i] for i in dataset.diagnosis] 

X = dataset.drop('diagnosis',axis=1).values
y = dataset['diagnosis'].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split   
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(32,activation='relu',init='uniform',input_dim=30))
model.add(Dense(32,activation='relu',init='uniform'))
model.add(Dense(1,activation='sigmoid',init='uniform'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=10,epochs=10)
y_pred = model.predict(X_test)

y_pred = (y_pred>0.5)

model.evaluate(X_test,y_test)