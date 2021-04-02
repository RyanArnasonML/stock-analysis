# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 20:47:55 2021

http://theautomatic.net/2019/04/17/how-to-get-options-data-with-python/

@author: ryanar
"""
import pandas as pd
from yahoo_fin import options
from yahoo_fin import stock_info as si

"""
nflx_dates = options.get_expiration_dates('nflx')

chain = options.get_options_chain("nflx")

calls = chain["calls"]

puts = chain["puts"]

nflx_dates= options.get_expiration_dates("nflx")
 
info = {}
for date in nflx_dates:
    info[date] = options.get_options_chain("nflx")
    
"""
# dow_tickers = si.tickers_dow()

# nasdaq = si.tickers_nasdaq()

# sp = si.tickers_sp500()


aapl_earnings=si.get_next_earnings_date("aapl")


aapl_earnings_hist = si.get_earnings_history("aapl")

data = pd.DataFrame.from_dict(aapl_earnings_hist)

