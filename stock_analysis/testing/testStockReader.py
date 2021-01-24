# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 10:24:21 2020

@author: ryanar
"""

import matplotlib.pyplot as plt
from stock_analysis import StockReader, StockVisualizer, Technical, AssetGroupVisualizer, StockAnalyzer, AssetGroupAnalyzer, StockModeler 
from stock_analysis.utils import group_stocks, describe_group, make_portfolio


reader = StockReader('2019-01-01','2020-11-21')
apple = reader.get_ticker_data('AAPL')

sp500 = reader.get_index_data()

#apple.to_csv('apple.csv', index = False, header=True)

