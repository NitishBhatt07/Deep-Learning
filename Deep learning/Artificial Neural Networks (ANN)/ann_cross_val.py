# -*- coding: utf-8 -*-

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


'''# Part 4 - evaluating ,improving and tuning th ANN...........'''

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
# Dropout class
from keras.layers import Dropout

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    # adding dropout to hidden layer one
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    # adding dropout to hidden layer two
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier,batch_size=10,nb_epoch=50)

accuracies = cross_val_score(estimator=classifier,X = X_train,y= y_train,cv=10)
mean = accuracies.mean()
variance = accuracies.std()


# Imporving the ANN

# Dropout Regularization to reduce overfiting if needed






