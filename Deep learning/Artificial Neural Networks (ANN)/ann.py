# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 18:09:00 2020

@author: Nbbhatt
"""

'''# Part 1 - Data preprocessing.............................'''

# Importing Libraries
import numpy as np
import pandas as pd
# Importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values
# Encoding catagorical Data
#catagorical meanns the string values column or data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
# dummie var for column country becoze it has 3 attribute
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
# removing 1st dummi var to avoid dummi var trap
X = X[:,1:]
# Spliting the data set into training and testing data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''# Part 2 - Make ANN..........................................'''
# Importing the Keras libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
# Initializing the ANN
classifier = Sequential()
# Adding the input layer the first hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
# Adding Second hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
# Adding Output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
# Compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# optimizer => simply the algo we wana use to find optimal set of weights in NN.
# optimizer => stocastic G D algo and one of them is Adam.
# loss => is a loss function used by adam(SGD) or adam is based on.
# matrics => when weight is changed the algo used this criteria to improve model.
# Fitting the ANN to traning set
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=50)
''' Part 3 - Making predication and evaluatinf model.....................'''
# Predicting the Test set result
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)


''' #HomeWork...............................
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
''' 
X_train_new = sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]]))
new_predication = classifier.predict(X_train_new)
new_predication = (new_predication>0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# save the model
classifier.save_weights('model_ann.h5')










































