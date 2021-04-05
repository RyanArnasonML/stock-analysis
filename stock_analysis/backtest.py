# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

https://www.backtrader.com/docu/quickstart/quickstart/
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

# Import the backtrader platform
import backtrader as bt

# Create a Strategy
class TestStrategy(bt.Strategy):
    
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
    
    # Keep a reference to the "close" line in the data[0] dataseries    
    def __init__(self):
        self.dataclose = self.datas[0].close    
    
    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])
        
        # Current close less than previous close
        if self.dataclose[0] < self.dataclose[-1]:
            # Previous close less than the previous close
            if self.dataclose[-1] < self.dataclose[-2]:
                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY Create, %.2f' % self.dataclose[0])
                self.buy()
  
if __name__ == '__main__':
        
    # Instantiate the Cerebro engine
    cerebro = bt.Cerebro()
    
    # Add a strategy
    cerebro.addstrategy(TestStrategy)
    
    # Datas are in a subfolder of the samples. Need to find where the script is
    # because it could have been called from anywhere
    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    datapath = os.path.join(modpath, 'data/AAPL.csv')

    # Create a Data Feed
    data = bt.feeds.YahooFinanceCSVData(
        dataname=datapath,
        # Do not pass values before this date
        fromdate=datetime.datetime(2020, 1, 1),
        # Do not pass values after this date
        todate=datetime.datetime(2020, 12, 31),
        reverse=False)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)
    
    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    
    # Run over everything
    cerebro.run()
    
    # Print out the final result
    print('Final Portfolio ValueL %.2f' % cerebro.broker.getvalue())