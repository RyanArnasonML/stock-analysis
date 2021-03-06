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
    
    def OBV_EMA(self, timeperiod=20):
        """        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        obv = ta.OBV(np.log(self.data.close),np.log(self.data.volume))
        obv_ema = obv.ewm(com=timeperiod, adjust=True, min_periods=timeperiod).mean()              
        
        return (obv-obv_ema)/obv
    
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
        
        plus_dm = ta.PLUS_DM(self.data.high,self.data.low, timeperiod)
        minus_dm = ta.MINUS_DM(self.data.high,self.data.low, timeperiod)
                
        if plus_dm > minus_dm:
            stopLoss = self.data.close - multiplier * stopLoss
        else:
            stopLoss = self.data.close + multiplier * stopLoss
            

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
    
    def Schaff(self, shortPeriod=23, longPeriod=50):
        """
        Schaff Trend Cycle (STC)
        
        STC indicator is a forward-looking leading indicator combining moving averages (MACD) with oscillator (stochastic).  

        Buy Signal: 25
        Sell Signal: 75
        
        Drawbacks: Lags in indicating exit positions, by staying in overbought or oversold position too long

        Parameters
        ----------
        shortPeriod : TYPE, optional
            This is the time frame for the shorter expotential moving average (EMA). The default is 23.
        longPeriod : TYPE, optional
            This is the time frame for the longer expotential moving average (EMA). The default is 50.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
       
        shortEMAClose = ta.EMA(self.data.close, timeperiod=shortPeriod)
        longEMAClose = ta.EMA(self.data.close, timeperiod=longPeriod)
        
        macdClose = shortEMAClose - longEMAClose
        
        shortEMALow = ta.EMA(self.data.low, timeperiod=shortPeriod)
        longEMALow = ta.EMA(self.data.low, timeperiod=longPeriod)
        
        macdLow = shortEMALow - longEMALow
        
        shortEMAHigh = ta.EMA(self.data.high, timeperiod=shortPeriod)
        longEMAHigh = ta.EMA(self.data.high, timeperiod=longPeriod)
        
        macdHigh = shortEMAHigh - longEMAHigh
        
        fastk, fastd = ta.STOCHF(macdHigh, macdLow, macdClose, fastk_period=10, fastd_period=10, fastd_matype=0)
                
        return 100 * ((macdClose - fastk) / (fastd - fastk))
    
    def Vortex(self,timeperiod = 14):
                
        vortex = pd.DataFrame()
        
        vortex['trueRange'] = ta.TRANGE(self.data.high, self.data.low, self.data.close)
        
        # Absolute value of current high minus prior low
        vortex['vm+'] = abs(self.data.high - self.data.low.shift(periods=-1))
        
        # Absolute value of current low minus prior high
        vortex['vm-'] = abs(self.data.low - self.data.high.shift(periods=-1))
        
        vortex['vi+'] = vortex['vm+'] / vortex['trueRange']
        vortex['vi-'] = vortex['vm-'] / vortex['trueRange']
        
        vortex['vi+'] = vortex['vi+'].rolling(timeperiod).sum()
        vortex['vi-'] = vortex['vi-'].rolling(timeperiod).sum()
        
        return None
    
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
    
    def Donchian(self, timeperiod = 14):
        
        donchian = pd.DataFrame()
        
        donchian['upper'] = self.data.high.rolling(timeperiod).max()
        donchian['lower'] = self.data.low.rolling(timeperiod).min()
        donchian['middle'] = (donchian['upper'] + donchian['lower']) / 2
        
        return donchian
    
    def AverageDirectionalIndex(self, timeperiod = 14):
        '''
        Average Direction Movement Index (ADX)
        
        ADX is used to determine the stength of a trend.
        
        If -DI is above the +DX, down trend
        If -DI is below the +DX, up trend
        ADX below 20, price is trendless
        ADX above 20, price is trending

        Parameters
        ----------
        timeperiod : TYPE, optional
            DESCRIPTION. The default is 14.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        
        dmi = pd.DataFrame()
        
        # Need to figure out if it should be PLUS_DM, MINUS_DM, MINUS_DI, PLUS_DI, Directional Movement Index or Average Directional Movement Index
        
        dmi['+di'] = ta.PLUS_DM(self.data.high,self.data.low, timeperiod)
        dmi['-di'] = ta.MINUS_DM(self.data.high,self.data.low, timeperiod)
        dmi['dmi'] = ta.DX(self.data.high,self.data.low, self.data.close, timeperiod)
        
        return dmi
    
    def AroonOscillator(self, timeperiod = 14):
        return ta.AROONOSC(self.data.high,self.data.low, timeperiod=14)
    
    def RateOfChange(self, rateOfChange, lookback=1, where=0, what=0):
        
        for i in range(len(self.data)):
            rateOfChange[i, where] = ((rateOfChange[i, what] - rateOfChange[i - lookback, what]) / rateOfChange[i - lookback, what]) * 100
                                                                                                                
        return rateOfChange
        