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

def transform(x):
    y = 0.05 * x ** 2 + 0.2 * x + np.sin(x) + 5
    y += np.random.standard_normal(len(x)) * 0.2
    return y

# Create sample data
x = np.linspace(-2 * np.pi, 2 * np.pi, 500)
a = transform(x)

plt.figure(figsize=(10,6))
plt.plot(x,a)

# Change a array to a two column vector
a = a.reshape((len(a), -1))

lags = 5

# Create batches of lagged sequential data.
g = TimeseriesGenerator(a, a, length=lags, batch_size=5)

# The sequential mode is designed to work with exactly one input tensor and one output tensor.
model = Sequential()

model.add(SimpleRNN(500, activation='relu', input_shape = (lags, 1)))

model.add(Dense(1, activation='linear'))

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

model.summary()

model.fit(g, epochs=500, steps_per_epoch=10, verbose=False)

x = np.linspace(-6 * np.pi, 6 * np.pi, 1000)
d = transform(x)

g_ = TimeseriesGenerator(d, d, length=lags, batch_size=len(d))

f = list(g_)[0][0].reshape((len(d) - lags, lags, 1))
y = model.predict(f, verbose=False)

plt.figure(figsize=(10,6))
plt.plot(x[lags:], d[lags:], label='data', alpha=0.75)
plt.plot(x[lags:], y, 'r.', label='pred', ms=3)
plt.axvline(-2*np.pi, c='g' , ls='--')
plt.axvline(2*np.pi, c='g' ,ls = '--')
plt.text(-15, 22, 'out-of-sample') 
plt.text(-2, 22, 'in-sample') 
plt.text(10, 22, 'out-of-sample') 
plt.legend()