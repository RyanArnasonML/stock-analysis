# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 09:51:24 2021

@author: ryanar
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from pprint import pprint
from pylab import plt, mpl

plt.style.use('seaborn')

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Dense 

def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)   
set_seeds()

# Create sample data
a = np.arange(100)
# Change a array to a two column vector
a = a.reshape((len(a), -1))

lags = 3

# Create batches of lagged sequential data.
g = TimeseriesGenerator(a, a, length=lags, batch_size=5)

# The sequential mode is designed to work with exactly one input tensor and one output tensor.
model = Sequential()

model.add(SimpleRNN(100, activation='relu', input_shape = (lags, 1)))

model.add(Dense(1, activation='linear'))

model.compile(optimizer='adagrad', loss='mse', metrics=['mae'])

model.summary()

h = model.fit(g, epochs=1000, steps_per_epoch=5, verbose=False)

res = pd.DataFrame(h.history)

res.tail(3)

# Preformance metrics during RNN training
res.iloc[10:].plot(figsize=(10, 6), style=['--','--']);

# In sample prediction
x = np.array([21, 22, 23]).reshape((1, lags, 1))
y = model.predict(x, verbose=False)
print(int(round(y[0,0])))

# Out of sample prediction
x = np.array([1187, 1188, 1189]).reshape((1, lags, 1))
y = model.predict(x, verbose=False)
print(int(round(y[0,0])))