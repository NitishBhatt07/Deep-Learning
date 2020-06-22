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


# Imporving the ANN

# Dropout Regularization to reduce overfiting if needed

# Tuning the ANN
# perameter tuning is simply finding the best value of hiper parameter.
# hiper parameter are some fixed value like batchSize,epoch,optimizer.
# we use gridsearch who try several combination for hiper perameters.

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense


def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)

parameters = {'batch_size':[25,32],
              'nb_epoch':[50,80],
              'optimizer':['adam','rmsprop']}

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv = 10)

grid_search =  grid_search.fit(X_train,y_train)

best_parameters = grid_search.best_params_ 
best_accuracy = grid_search.best_score_






















