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

# Another Way to Extract Trading Signals From Moving Average Crosses.
# https://codeburst.io/a-new-way-to-trade-moving-averages-a-study-in-python-266dbb72b9d0

# Digging deeper and modifying the Bollinger Bands.
# https://kaabar-sofien.medium.com/the-normalized-bollinger-indicator-another-way-to-trade-the-range-back-testing-in-python-db22c111cdde

# Predicting Stock Market Dips, Crashes and Corrections with Light Gradient Boosting Machines
# https://peijin.medium.com/predicting-stock-market-dips-crashes-and-corrections-with-light-gradient-boosting-machines-58064f3a193a

# Fusing the RSI with the Stochastic Oscillator. Does It Improve the Trading Results?
# https://kaabar-sofien.medium.com/fusing-the-rsi-with-the-stochastic-osicllator-does-it-improve-the-trading-results-3c000a5ff588

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
    
    def RelativeStrengthIndex(self, timeperiod=14):
        """
        Relative Strength Index
        
        Is a momentum indicator, measuring the magnitude of recent price changes.
        
        70 is overbought
        30 is oversold

        Parameters
        ----------
        timeperiod : Integer, optional
            The time period of the moving average. The default is 14.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
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
        """
        Average True Range
        
        Is a lagging indicator, used to provide insights into volatility.

        Parameters
        ----------
        timeperiod : Integer, optional
            DESCRIPTION. The default is 14.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        return ta.ATR(self.data.high, self.data.low, self.data.close, timeperiod)
    
    def AverageTrueRangeStopLoss(self, timeperiod = 14, multiplier = 2):
        """
        Average True Range
        
        Is a lagging indicator, used to provide insights into volatility.

        Parameters
        ----------
        timeperiod : Integer, optional
            DESCRIPTION. The default is 14.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        stopLoss = ta.ATR(self.data.high, self.data.low, self.data.close, timeperiod)
                
        stopLoss =  -multiplier*stopLoss
        stopLoss = self.data.close + stopLoss   

        stopLoss.dropna(inplace=True)            
        
        return stopLoss
    
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
        
        return cloud
    
    def ParabolicSAR(self, acceleration = 0, maximum = 0):
        
        real = ta.SAR(self.data.high, self.data.low, acceleration, maximum)
        
        return real
    
    def Aroon(self, timeperiod = 14):
        
        aroon = pd.DataFrame()
        
        aroon['down'], aroon['up'] = ta.AROON(self.data.high, self.data.low, timeperiod=14)
        
        return aroon
    
    def VerticalHorizontalFilter(self, timeperiod = 28):
        '''
        Vertical Horizontal Filter (VHF)
        
        The vertical horizontal filter is a measurement of how strong or weak the market
        is trending.
        
        
        Returns
        -------
        None.

        '''
        
        # Determine the highest closing price (HCP) in time period.
        hcp = self.data.close.rolling(timeperiod).max()
                       
        #  Determine the lowest closing price (LCP) in time period.
        lcp = self.data.close.rolling(timeperiod).min()
        
        # Calculate the range of closing prices in n periods:  HCP - LCP
        closingRange = hcp - lcp
        
        # Calculate the movement in closing price for each period: Closing price [today] - Closing price [yesterday]
        # Add up all price movements for time period, disregarding whether they are up or down:
        change = abs(self.data.close.diff(1))
        
        # Sum of absolute values of ( Close [today] - Close [yesterday] ) for time period
        change = change.rolling(timeperiod).sum()
                
        # VHF = (HCP - LCP) / (Sum of absolute values for time period)
        vhf = closingRange / change       
    
        return vhf
    
    def AroonOscillator(self, timeperiod = 14):
        return ta.AROONOSC(self.data.high,self.data.low, timeperiod=14)
    
    def RateOfChange(self, rateOfChange, lookback=1, where=0, what=0):
        
        for i in range(len(self.data)):
            rateOfChange[i, where] = ((rateOfChange[i, what] - rateOfChange[i - lookback, what]) / rateOfChange[i - lookback, what]) * 100
                                                                                                                
        return rateOfChange
        