# -*- coding: utf-8 -*-

# Part -1 Data Preprocessing........

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
traning_set = dataset_train.iloc[:,1:2].values

'''
# Feature Scaling..............
1: Standardisation
Xstand = (X - mean(X))/standard deviation (X)

2: Normalisation
Xnorm = (X - min(X))/max(X) - min(X)

'''

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
traning_set_scaled = sc.fit_transform(traning_set)

# Creating a data structure with 60 timesteps and 1 output
#60 timesteps means we are looking 60 days back (t) and then 
#predict one day data (t+1)
X_train = []
y_train = []
for i in range(60,1258):
    X_train.append(traning_set_scaled[i-60:i,0])# 0 for cloumn
    y_train.append(traning_set_scaled[i,0])
# make numpy array
X_train,y_train = np.array(X_train),np.array(y_train)

# Reshapig
# X_train.shape[0] is equal to the rows of dataset
# X_train.shape[1] is equal to the clomns or timesteps of dataset
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))


# Part 2 - Building the RNN
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the 1st LSTM layer and some Dropout
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

# Adding the 2st LSTM layer and some Dropout
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

# Adding the 3st LSTM layer and some Dropout
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

# Adding the 4st LSTM layer and some Dropout
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# Adding the output layer
# the unit in output layer is simply a dim of that layer
regressor.add(Dense(units=1))

# Compile the RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train,y_train,epochs=100,batch_size=32)


# Part 3 - Making the predictions and validating the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs= sc.transform(inputs)

X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])# 0 for cloumn
X_test = np.array(X_test)
X_test= np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
# predicting the price
predicated_stock_price = regressor.predict(X_test)
# get scaled back for use
predicated_stock_price = sc.inverse_transform(predicated_stock_price)


# Visualising the results
plt.plot(real_stock_price,color='red',label='Real Stock Price')
plt.plot(predicated_stock_price,color='blue',label='Predicted Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('T')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()



