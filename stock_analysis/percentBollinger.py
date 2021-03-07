# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 21:46:06 2021

@author: ryanar
"""
import numpy as np
import yfinance
import talib as ta 
import pandas as pd

my_data = yfinance.download('AAPL', '2013-1-1','2021-2-28')


def ma(Data, lookback, what='Close'):   
        
    return Data[what].rolling(lookback).mean()

def volatility(Data, lookback, what='Close'):   
        
    return Data[what].rolling(lookback).std()

def BollingerBands(Data, lookback, standard_distance, what='Close'):
    
    bollBand = pd.DataFrame()
    
    # Calculating the mean
    bollBand['mean'] = Data[what].rolling(lookback).mean()
    
    # Calculating the volatility
    bollBand['std'] = Data[what].rolling(lookback).std()
    
    # Upper Bollinger Band
    bollBand['upper'] = bollBand['mean'] + standard_distance * bollBand['std'] 
    
    # Lower Bollinger Band
    bollBand['lower'] = bollBand['mean'] - standard_distance * bollBand['std'] 
    
    return bollBand

def percent_bollinger_indicator(Data, lookback, standard_distance, what='Close'):
    
        
    bollBand = BollingerBands(Data, lookback, standard_distance, what)  

    bollBand['%'] = ((Data[what ] - bollBand['lower']) / (bollBand['upper']- bollBand['lower']))       
        
    return bollBand

def divergence(Data, indicator, lower_barrier, upper_barrier, width, buy, sell):
    
    for i in range(len(Data)):
        
        try:
            
            if[i, indicator] < lower_barrier:
                
                for a in range(i + 1, i + width):
                    
                    if Data[a, indicator] > lower_barrier:
                        
                        for r in range(a+1, a + width):
                            
                            if Data[r, indicator] < lower_barrier and Data[r, indicator] > Data[i, indicator] and Data[r, 3] < Data[i, 3]:
                                
                                for s in range(r+1, r+width):
                                    if Data[s, indicator] > lower_barrier:
                                        Data[s,buy] = 1
                                        break
                                    else:
                                        break
                            else:
                                break
                    else:
                        break
            else:
                break
        except IndexError:
            pass
        
        
    for i in range(len(Data)):
        
        try:
            
            if[i, indicator] > upper_barrier:
                
                for a in range(i + 1, i + width):
                    
                    if Data[a, indicator] < upper_barrier:
                        
                        for r in range(a + 1, a + width):
                            
                            if Data[r, indicator] < upper_barrier and Data[r, indicator] < Data[i, indicator] and Data[r, 3] > Data[i, 3]:
                                
                                for s in range(r+1, r+width):
                                    if Data[s, indicator] < upper_barrier:
                                        Data[s, sell] = -1
                                        break
                                    else:
                                        break
                            else:
                                break
                    else:
                        break
            else:
                break
        except IndexError:
            pass
        
    return Data

result = percent_bollinger_indicator(my_data, 14, 2)


# my_data = divergence(my_data, bollinger_percentage_column, 0, 1, 20, buy_column, sell_column)