# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 15:44:00 2021

conda install -c conda-forge kneed

@author: ryanar
"""

import yfinance
from sklearn.cluster import KMeans
import numpy as np
from kneed import DataGenerator, KneeLocator

df = yfinance.download('AAPL', '2013-1-1','2020-1-1')

X = np.array(df['Close'])

sum_of_squared_distances = []

# Search for elbow point with the biggest improvement.  
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X.reshape(-1,1))
    sum_of_squared_distances.append(km.inertia_)
    
kn = KneeLocator(K, sum_of_squared_distances, S=1.0, curve="convex", direction="decreasing")
# kn.plot_knee()

kmeans = KMeans(n_clusters = kn.knee+1).fit(X.reshape(-1,1))
c = kmeans.predict(X.reshape(-1,1))
minmax = []

for i in range(kn.knee+1):
    minmax.append([-np.inf,np.inf])

# Search for the minimum and maximum values inside of each cluster.      
for i in range(len(X)):
    
    cluster = c[i]
    
    if X[i] > minmax[cluster][0]:
        minmax[cluster][0] = X[i]
        
    if X[i] < minmax[cluster][1]:
        minmax[cluster][1] = X[i]
        
from matplotlib import pyplot as plt

for i in range(len(X)):
    colors = ['b','g','r','c','m','y','k','w']
    c = kmeans.predict(X[i].reshape(-1,1))[0]
    color = colors[c]
    plt.scatter(i,X[i], c = color, s = 1)
    
for i in range(len(minmax)):
    plt.hlines(minmax[i][0], xmin = 0,  xmax = len(X), colors = 'g')
    plt.hlines(minmax[i][1], xmin = 0,  xmax = len(X), colors = 'r')        