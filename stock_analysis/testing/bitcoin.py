# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 10:24:21 2020

@author: ryanar
"""

import matplotlib.pyplot as plt
from stock_analysis import StockReader, StockVisualizer, Technical, AssetGroupVisualizer, StockAnalyzer, AssetGroupAnalyzer, StockModeler 
from stock_analysis.utils import group_stocks, describe_group, make_portfolio


reader = StockReader('2019-01-01','2020-11-21')
bitcoin = reader.get_bitcoin_data()

sma_bitcoin = Technical(bitcoin).SimpleMovingAverage()
atr_bitcoin = Technical(bitcoin).AverageTrueRange()
obv_bitcoin = Technical(bitcoin).KalmanAverage()

bitcoin_viz = StockVisualizer(bitcoin)

ax = bitcoin_viz.evolution_over_time(
    'close', figsize=(10, 4), legend=False, 
    title='Bitcoin closing price over time'
    )

bitcoin_viz.add_reference_line(
     ax, x=bitcoin.high.idxmax(), color='k', linestyle=':', alpha=0.5,
     label=f'highest value ({bitcoin.high.idxmax():%b %d})')

ax.set_ylabel('price ($)')

ay = bitcoin_viz.trade_volume()

#az = bitcoin_viz.candle_stick()

#aa = bitcoin_viz.renko()

#ab = bitcoin_viz.qqplot()

ac = bitcoin_viz.histogram(column='close')