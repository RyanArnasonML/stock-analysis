# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 11:20:05 2021

@author: ryanar
"""

import numpy as np
import yfinance
import talib as ta 
import pandas as pd

ticker = yfinance.Ticker("SPY")
df = ticker.history(period = '5mo')

# for i in range(2,df.shape[0]):
#   current = df.iloc[i,:]
#   prev = df.iloc[i-1,:]
#   prev_2 = df.iloc[i-2,:]
#   realbody = abs(current['Open'] - current['Close'])
#   candle_range = current['High'] - current['Low']
  
#   idx = df.index[i]

#   df.loc[idx,'Bullish swing'] = current['Low'] > prev['Low'] and prev['Low'] < prev_2['Low']
  
df['Two Crows'] = ta.CDL2CROWS(df['Open'], df['High'], df['Low'], df['Close'])
df['Three Black Crows'] = ta.CDL3BLACKCROWS(df['Open'], df['High'], df['Low'], df['Close'])
df['Three Inside Up/Down'] = ta.CDL3INSIDE(df['Open'], df['High'], df['Low'], df['Close'])
df['Three-Line Strike'] = ta.CDL3LINESTRIKE(df['Open'], df['High'], df['Low'], df['Close'])  
df['Three Outside Up/Down'] = ta.CDL3OUTSIDE(df['Open'], df['High'], df['Low'], df['Close'])
df['Three Stars In The South'] = ta.CDL3STARSINSOUTH(df['Open'], df['High'], df['Low'], df['Close'])
df['Three Advancing White Soldiers'] = ta.CDL3WHITESOLDIERS(df['Open'], df['High'], df['Low'], df['Close'])
df['Abandoned Baby'] = ta.CDLABANDONEDBABY(df['Open'], df['High'], df['Low'], df['Close'])

df['Advance Block'] = ta.CDLADVANCEBLOCK(df['Open'], df['High'], df['Low'], df['Close'])
df['Belt-hold'] = ta.CDLBELTHOLD(df['Open'], df['High'], df['Low'], df['Close'])
df['Breakaway'] = ta.CDLBREAKAWAY(df['Open'], df['High'], df['Low'], df['Close'])
df['Closing Marubozu'] = ta.CDLCLOSINGMARUBOZU(df['Open'], df['High'], df['Low'], df['Close'])  
df['Concealing Baby Swallow'] = ta.CDLCONCEALBABYSWALL(df['Open'], df['High'], df['Low'], df['Close'])
df['Counterattack'] = ta.CDLCOUNTERATTACK(df['Open'], df['High'], df['Low'], df['Close'])
df['Dark Cloud Cover'] = ta.CDLDARKCLOUDCOVER(df['Open'], df['High'], df['Low'], df['Close'])

df['Doji'] = ta.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
df['Doji Star'] = ta.CDLDOJISTAR(df['Open'], df['High'], df['Low'], df['Close'])
df['Dragonfly Doji'] = ta.CDLDRAGONFLYDOJI(df['Open'], df['High'], df['Low'], df['Close'])
df['Engulfing Pattern'] = ta.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])  
df['Evening Doji Star'] = ta.CDLEVENINGDOJISTAR(df['Open'], df['High'], df['Low'], df['Close'])
df['Evening Star'] = ta.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
df['Up/Down-gap side-by-side white lines'] = ta.CDLGAPSIDESIDEWHITE(df['Open'], df['High'], df['Low'], df['Close'])
df['Gravestone Doji'] = ta.CDLGRAVESTONEDOJI(df['Open'], df['High'], df['Low'], df['Close'])

df['Hammer'] = ta.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
df['Hanging Man'] = ta.CDLHANGINGMAN(df['Open'], df['High'], df['Low'], df['Close'])
df['Harami Pattern'] = ta.CDLHARAMI(df['Open'], df['High'], df['Low'], df['Close'])
df['Harami Cross Pattern'] = ta.CDLHARAMICROSS(df['Open'], df['High'], df['Low'], df['Close'])  
df['High-Wave Candle'] = ta.CDLHIGHWAVE(df['Open'], df['High'], df['Low'], df['Close'])
df['Hikkake Pattern'] = ta.CDLHIKKAKE(df['Open'], df['High'], df['Low'], df['Close'])
df['Modified Hikkake Pattern'] = ta.CDLHIKKAKEMOD(df['Open'], df['High'], df['Low'], df['Close'])
df['Homing Pigeon'] = ta.CDLHOMINGPIGEON(df['Open'], df['High'], df['Low'], df['Close'])

df['Identical Three Crows'] = ta.CDLIDENTICAL3CROWS(df['Open'], df['High'], df['Low'], df['Close'])
df['In-Neck Pattern'] = ta.CDLINNECK(df['Open'], df['High'], df['Low'], df['Close'])
df['Inverted Hammer'] = ta.CDLINVERTEDHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
df['Kicking'] = ta.CDLKICKING(df['Open'], df['High'], df['Low'], df['Close'])  
df['Kicking By Length'] = ta.CDLKICKINGBYLENGTH(df['Open'], df['High'], df['Low'], df['Close'])
df['Ladder Bottom'] = ta.CDLLADDERBOTTOM(df['Open'], df['High'], df['Low'], df['Close'])
df['Long Legged Doji'] = ta.CDLLONGLEGGEDDOJI(df['Open'], df['High'], df['Low'], df['Close'])
df['Long Line Candle'] = ta.CDLLONGLINE(df['Open'], df['High'], df['Low'], df['Close'])

df['Rising/Falling Three Methods'] = ta.CDLRISEFALL3METHODS(df['Open'], df['High'], df['Low'], df['Close'])
df['Separating Lines'] = ta.CDLSEPARATINGLINES(df['Open'], df['High'], df['Low'], df['Close'])
df['Shooting Star'] = ta.CDLSHOOTINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
df['Short Line Candle'] = ta.CDLSHORTLINE(df['Open'], df['High'], df['Low'], df['Close'])  
df['Spinning Top'] = ta.CDLSPINNINGTOP(df['Open'], df['High'], df['Low'], df['Close'])
df['Stalled Pattern'] = ta.CDLSTALLEDPATTERN(df['Open'], df['High'], df['Low'], df['Close'])
df['Stick Sandwich'] = ta.CDLSTICKSANDWICH(df['Open'], df['High'], df['Low'], df['Close'])
df['Takuri'] = ta.CDLTAKURI(df['Open'], df['High'], df['Low'], df['Close'])

# Tasuki Gap is a three-bar candlestick formation that is commonly used to signal the continuation of the current trend.
df['Tasuki Gap'] = ta.CDLTASUKIGAP(df['Open'], df['High'], df['Low'], df['Close'])

# The pattern is thought to act as a continuation pattern, but in reality, it acts as a reversal pattern about half the time.
df['Thrusting Pattern'] = ta.CDLTHRUSTING(df['Open'], df['High'], df['Low'], df['Close'])

# A tri-star is a three line candlestick pattern that can signal a possible reversal in the current trend, be it bullish or bearish.
df['Tristar Pattern'] = ta.CDLTRISTAR(df['Open'], df['High'], df['Low'], df['Close'])
df['Unique 3 River'] = ta.CDLUNIQUE3RIVER(df['Open'], df['High'], df['Low'], df['Close'])  
df['Upside Gap Two Crows'] = ta.CDLUPSIDEGAP2CROWS(df['Open'], df['High'], df['Low'], df['Close'])
df['Upside/Downside Gap Three Methods'] = ta.CDLXSIDEGAP3METHODS(df['Open'], df['High'], df['Low'], df['Close'])
