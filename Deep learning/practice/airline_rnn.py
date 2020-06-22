# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

dataset = pd.read_csv('airline-passengers.csv')
total_dataset = dataset.iloc[:,1:2]

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
total_dataset = sc.fit_transform(total_dataset)

# spliting into train test
train_size = int(len(total_dataset)*0.80)
test_size = int(len(total_dataset) - train_size)

X_train= total_dataset[0:train_size,:]
X_test = total_dataset[train_size:]
