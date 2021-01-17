# -*- coding: utf-8 -*-
"""
Visualize financial instruments

Created on Sat Oct 31 14:15:43 2020

@author: ryanar
"""

import math
import scipy


import talib as ta  #https://github.com/mrjbq7/ta-lib

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import probplot
from pykalman import KalmanFilter

from .utils import validate_df

class Technical:
    """ 
    Base visualizer class not intended for direct use.
    """
    
    @validate_df(columns={'open','high','low','close','volume'})
    def __init__(self, df):
        """
        Visualizer Initializer

        Parameters
        ----------
        df : Dataframe
            The Dataframe contains the stock market data.

        Returns
        -------
        None.

        """
        self.data = df
        
    def OnBalanceVolume(self):
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        return ta.OBV(self.data.close,self.data.volume)
    
    def RelativeStrengthIndex(self, timeperiod):
        return ta.RSI(self.data.close,timeperiod)
    
    def MovingAverageConvergenceDivergence(self, fastperiod=12, slowperiod=26,signalperiod=9):
        """
        MACD - Moving Average Convergence/Divergence

        Parameters
        ----------
        fastperiod : TYPE, optional
            DESCRIPTION. The default is 12.
        slowperiod : TYPE, optional
            DESCRIPTION. The default is 26.
        signalperiod : TYPE, optional
            DESCRIPTION. The default is 9.

        Returns
        -------
        df : TYPE
            DESCRIPTION.

        """
        
        df = pd.DataFrame()
        
        df['macd'], df['signal'], df['history'] = ta.MACDFIX(self.data.close, 9)
                       
        return df[-30:]
    
    def SimpleMovingAverage(self, timeperiod = 14):
        """
        The Simple Moving Average (SMA) is calculated by adding the price of an instrument over a number of time periods  and then dividing the sum by the number of time periods. The SMA is basically the average price of the given time period, with equal  weighting given to the price of each period.

        Simple Moving Average
        SMA = ( Sum ( Price, n ) ) / n    

        Where: n = Time Period
        
        Parameters
        ----------
        timeperiod : Unsigned Integer, optional
            DESCRIPTION. The default is 14 days.

        Returns
        -------
        sma : DataFrame
            Returns the mean value over the provided time period.

        """                
        return ta.SMA(self.data.close,timeperiod)
    
    def BollingerBands(self, timeperiod = 5, nbdevup = 2, nbdevdn=2, matype=0):
        df = pd.DataFrame()
        
        df['close'] = self.data.close
        
        df['upper'], df['middle'], df['lower'] = ta.BBANDS(self.data.close, timeperiod, nbdevup, nbdevdn, matype)
        return df[-180:]
        
    def AverageTrueRange(self, timeperiod = 14):
        atr = []
        
        history = [] # to track a history of prices
        
        self.data['price_range'] = self.data.high - self.data.low
        
        for price_range in self.data.price_range:
            
            history.append(price_range)
            
            if len(history) > timeperiod: 
                del (history[0])       
        
            atr.append(np.mean(history))
        
        return atr
    
    def KalmanAverage(self):
        
        kalman = pd.DataFrame()
        
        kf = KalmanFilter(transition_matrices = [1],
                  observation_matrices = [1],
                  initial_state_mean = 0,
                  initial_state_covariance = 1,
                  observation_covariance=1,
                  transition_covariance=.01)
        
        state_means, _ = kf.filter(self.data.close)       
        
        return state_means        
    
    def ChoppinessIndex(self, timeperiod = 14):
        """
        # 100 * LOG10( SUM(ATR(1), n) / ( MaxHi(n) - MinLo(n) ) ) / LOG10(n)
        # n = User defined period length.
        # LOG10(n) = base-10 LOG of n
        # ATR(1) = Average True Range (Period of 1)
        # SUM(ATR(1), n) = Sum of the Average True Range over past n bars MaxHi(n) = The highest high over past n bars
        """
        return ta.C
    
    def IchimokuCloud(self):
        
        cloud = pd.DataFrame()
        
        conversionLine = (ta.SMA(self.data.high, 9) + ta.SMA(self.data.low, 9))/2
        baseLine = (ta.SMA(self.data.high, 26) + ta.SMA(self.data.low, 26))/2
        leadingSpanA = (conversionLine + baseLine)/2
        leadingSpanB = (ta.SMA(self.data.high, 52) + ta.SMA(self.data.low, 52))/2
        
        cloud['conversionLine'] = conversionLine
        cloud['baseLine'] = baseLine
        cloud['leadingSpanA'] = leadingSpanA
        cloud['leadingSpanB'] = leadingSpanB
        
        cloud.dropna(inplace=True)
        
        print(cloud)
        
        return cloud
    
    def ParabolicSAR(self, acceleration = 0, maximum = 0):
        
        real = ta.SAR(self.data.high, self.data.low, acceleration, maximum)
        
        return real
    
    def Aroon(self, timeperiod = 14):
        
        aroon = pd.DataFrame()
        
        aroon['down'], aroon['up'] = ta.AROON(self.data.high, self.data.low, timeperiod=14)
        
        return aroon
    
    def AroonOscillator(self, timeperiod = 14):
        return ta.AROONOSC(self.data.high,self.data.low, timeperiod=14)