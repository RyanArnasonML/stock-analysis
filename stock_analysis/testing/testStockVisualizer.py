# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 10:24:21 2020

@author: ryanar
"""

import matplotlib.pyplot as plt
from stock_analysis import StockReader, StockVisualizer, Technical, AssetGroupVisualizer, StockAnalyzer, AssetGroupAnalyzer, StockModeler 
from stock_analysis.utils import group_stocks, describe_group, make_portfolio
import numpy as np


reader = StockReader('2019-01-01','2020-11-21')
apple = reader.get_ticker_data('AAPL')

sp500 = reader.get_index_data()


apple_viz = StockVisualizer(apple)

ax = apple_viz.evolution_over_time(
    'close', figsize=(10, 4), legend=False, 
    title='Apple closing price over time'
    )

apple_viz.add_reference_line(
     ax, x=apple.high.idxmax(), color='k', linestyle=':', alpha=0.5,
     label=f'highest value ({apple.high.idxmax():%b %d})')

# Cannot have more than one moving average on a chart.
apple_viz.moving_average(
    'close','3D'
    )

apple_viz.moving_average(
    'close','8D'
    )

# Data needs to be shifted to show correctly on the chart.
apple_viz.moving_average(
    'close','200D'
    )

ax.set_ylabel('price ($)')

fb = reader.get_ticker_data('fb')
aapl = reader.get_ticker_data('aapl')
amzn = reader.get_ticker_data('amzn')
nflx = reader.get_ticker_data('nflx')
goog = reader.get_ticker_data('goog')

fb.close = np.log10(fb["close"])
aapl.close = np.log10(aapl["close"])
amzn.close = np.log10(amzn["close"])
nflx.close = np.log10(nflx["close"])
goog.close = np.log10(goog["close"])


fb.close = fb.close - fb.close.min()
aapl.close = aapl.close - aapl.close.min()
amzn.close = amzn.close - amzn.close.min()
nflx.close = nflx.close - nflx.close.min()
goog.close = goog.close - goog.close.min()


faang = group_stocks({'Facebook' : fb, 'Apple' : aapl, 'Amazon': amzn, 'Netflix' : nflx, 'Google': goog})

# Need to workng on making portfolio useful. 
faang_portfolio = make_portfolio(faang)

faang_viz = AssetGroupVisualizer(faang)

faang_viz.evolution_over_time('close')
faang_viz.show()

"""
faang_viz.heatmap(True)
faang_viz.show()
"""


faang_viz.boxplot('close')
faang_viz.show()

faang_viz.histogram('close')
faang_viz.show()
