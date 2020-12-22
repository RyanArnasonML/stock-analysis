# -*- coding: utf-8 -*-
"""
Gathering select stock data.

Created on Sat Oct 31 11:26:11 2020

@author: ryanar
"""

import datetime
import re

import pandas as pd
#import pandas_datareader.data as web


from stock_analysis.utils import label_sanitizer

class StockReader:
    """Class for reading financial data from websites. """
    
    _index_tickers = {
        'SP500' : '^GSPC',
        'DOW' : '^DJI',
        'NASDAQ' : '^IXIC'
        }
    
    def __init__(self, start, end=None):
        """
        Create a StockReader object for readin across a given date range.

        Parameters
        ----------
        start : datetime object or string in format 'YYYYMMDD'
            The first day of the stock market data.
        end : datetime object or string in format 'YYYYMMDD', optional
            The last day of the stock market data. The default is today.

        Returns
        -------
        A StockReader object.

        """
        self.start, self.end = map(
            lambda x:x.strftime('%Y%m%d') if isinstance(
                x, datetime.datetime
            ) else re.sub(r'\D','',x),
            [start, end or datetime.date.today()]
        )
        if self.start >= self.end:
            raise ValueError('START date must be before END date')
    
    @property
    def avalable_tickers(self):
        """
        Access the names of the indices whose tickers are supported.

        Returns
        -------
        list data type

        """
        return list(self._index_tickers())
    
    @classmethod
    def get_index_ticker(cls, index):
        """
        Class method for getting the ticker of the specified inex, if know.

        Parameters
        ----------
        cls : TYPE
            DESCRIPTION.
        index : String
            The name of the index.

        Returns
        -------
        String or None.

        """
        try:
            index = index.upper()
        except AttributeError:
            raise ValueError('Index must be a string')
        return cls._index_tickers.get(index, None)
    
    @label_sanitizer
    def get_ticker_data(self, ticker):
        """
        Get historical OHLC data for a given data range and ticker.
        Tries to get from Investor Exchange (IEX), but falls back to Yahoo! Finance if IEX doesn't have it.'

        Parameters
        ----------
        ticker : String
            The stock symbol to lookup.

        Returns
        -------
        Pandas Dataframe containing stock data.

        """
        # try:
        #     data = web.DataReader(ticker, 'iex',self.start,self.end)
        #     data.index = pd.to_datetime(data.Index)
        # except:
        #     data = web.get_data_yahoo(ticker, self.start, self.end)
        # return data
        df = pd.read_csv('stock_analysis/data/'+ ticker +'.csv')
        df['Date'] = pd.to_datetime(df['Date'])        
        
        df = df[df['Date'] > self.start]
        df = df[df['Date'] < self.end]
        
        df.set_index('Date', drop=True, inplace=True)
            
        return df
        
    @label_sanitizer        
    def get_bitcoin_data(self):
        
        """
        Get bitcoin historical open-high-low-close (OHLC) data from coinmarketcap.com for a given date ranger.

        Returns
        -------
        A pandas dataframe with the bitcoin data.

        """
        # try:
        #     data = pd.read_html('https://coinmarketcap.com/currencies/bitcoin/historical-data/?'
        #                     'start={}&end{}'.format(self.start,self.end), parse_dates=[0], index_col=[0]
        #                     )[0].sort_index()
        # except:
          
        df = pd.read_csv('stock_analysis/data/BTC-USD.csv')
        df['Date'] = pd.to_datetime(df['Date'])        
        
        df = df[df['Date'] > self.start]
        df = df[df['Date'] < self.end]
        
        df.set_index('Date', drop=True, inplace=True)
            
        return df
    
    @label_sanitizer
    def get_index_data(self, index='SP500'):
        """
        Get historical OHLC data from Yahoo! Finance for the chosen index for a given date range.

        Parameters
        ----------
        index : String, optional
            A string representing the Stock index to retrive. The default is 'SP500'.
            Currently support the indexes below:
                'SP500' for S&P 500,
                'DOW' for Dow Jones Industrial Average
                'NASDAQ' for NASDAQ Composite Index

        Returns
        -------
        A pandas dataframe with the index data.

        """
        if index not in self.avalable_tickers:
            raise ValueError('Index not supported. Available tickers are:') 
                             #{',.join(self.available_tickers)})
        
        return web.get_data_yahoo(self.get_index_ticker(index),self.start, self.end)