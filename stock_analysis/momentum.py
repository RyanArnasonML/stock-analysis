# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 10:54:08 2021

@author: ryanar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

def TSMStrategy(returns, period=1, shorts=False):
    
    if shorts:
        position = returns.rolling(period).mean().map(lambda x:-1 if x <= 0 else 1)
    else:
        position = returns.rolling(period).mean().map(lambda x: 0 if x <= 0 else 1)
    
    performance = position.shift(1) * returns
    
    return performance

#ticker = 'GME'
ticker = 'TSLA'

yfObj = yf.Ticker(ticker)
data = yfObj.history(start='2000-01-01',end='2020-12-31')

returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()

performance = TSMStrategy(returns, period=1, shorts=False).dropna()
years = (performance.index.max() - performance.index.min()).days / 365
perf_cum = np.exp(performance.cumsum())
tot = perf_cum[-1] - 1
ann = perf_cum[-1] ** (1 / years) -1
vol = performance.std() * np.sqrt(252)
rfr = 0.02
sharpe = (ann - rfr) / vol

print("1-day TSM Strategy yields:" +
      f"\n\t{tot*100:.2f}% total returns" +
      f"\n\t{ann*100:.2f}% annual returns" +
      f"\n\t{sharpe:.2f} Sharpe Ratio")    

gme_ret = np.exp(returns.cumsum())   
b_tot = gme_ret[-1] - 1
b_ann = gme_ret[-1] ** (1 / years) -1
b_vol = returns.std() * np.sqrt(252)
b_sharpe = (ann - rfr) / vol

print("Baseline Buy-and-Hold Strategy yields:" +
      f"\n\t{b_tot*100:.2f}% total returns" +
      f"\n\t{b_ann*100:.2f}% annual returns" +
      f"\n\t{b_sharpe:.2f} Sharpe Ratio")
     
