# -*- coding: utf-8 -*-
"""
Classe for technical analysis of assets.
Created on Sat Oct 31 19:35:28 2020

@author: ryanar
"""

import math
import scipy.stats as scs
import numpy as np
import pandas as pd

from pylab import mpl , plt
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

class BacktestBase(object):
    
    def __init__(self, symbol, startDate, endDate, amount, ftc=0.0, ptc=0.0, verbose=True):
        self.symbol = symbol
        self.start = startDate
        self.end = endDate
        self.initial_amount = amount
        self.amount = amount
        self.ftc = ftc
        self.ptc = ptc
        self.units = 0
        self.trades = 0
        self.verbose = verbose
        self.get_data()
        
    def get_data(self):
        raw = pd.read_csv('http://hilpisch.com/pyalgo_eikon_eod_data.csv', index_col = 0 , parse_dates = True ).dropna() 
        raw = pd.DataFrame(raw[self.symbol])
        raw = raw.loc[self.start:self.end] 
        raw.rename (columns = {self.symbol:'price'}, inplace = True)
        raw ['return'] = np.log(raw/raw.shift(1)) 
        self.data = raw.dropna ()
        
    def plot_data(self, cols=None):
        if cols is None:
            cols = ['price']
        self.data['price'].plot(figsize=(10,6), title = self.symbol)
        
    def get_date_price(self, bar):
        date = str(self.data.index[bar])[:10]
        price  = self.data.price.iloc[bar]
        return date, price
     
    def print_balance(self, bar):
        date, price = self.get_date_price(bar)   
        print(f'{date}|current amount{self.amount:.2f}')
        
    def print_net_wealth(self, bar):
        date, price = self.get_date_price(bar)
        net_wealth = self.units * price + self.amount
        print(f'{date}|current net wealth {net_wealth:.2f}')
        
    def place_buy_order(self, bar, units=None, amount=None):
        
        date, price = self. get_date_price(bar)
        
        if units is None:
            units = int(amount/price)
            
        self.amount -=(units * price) * (1   + self.ptc) + self.ftc
        self.units += units
        self.trades += 1
        
        if self.verbose:
            print(f'{date}| buying {units} units at {price:.2f}')
            self.print_balance(bar)
            self.print_net_wealth(bar)
            
    def place_sell_order(self, bar, units=None, amount=None):
        
        date, price = self.get_date_price(bar)
        
        if units is None:
            units = int(amount/price)
        
        self.amount += (units * price) * (1 - self.ptc) - self.ftc
        self.units -= units
        self.trades += 1
        
        if self.verbose:
            print(f'{date}| selling {units} units at {price:.2f}')
            self.print_balance(bar)
            self.print_net_wealth(bar)
        
    def close_out(self, bar):
        date, price = self. get_date_price(bar)
        self.amount += self.units * price
        self.units = 0
        self.trades += 1
        
        if self.verbose:
            print(f'{date}| inventory {self.units} units at {price:.2f}')
            print('=' * 55)
            
        print('Final balance [$] {:.2f}'.format(self.amount))    
        
        perf = ((self.amount - self.initial_amount) / self.initial_amount * 100)
        
        print('Net Performance [%] {:.2f}'.format(perf))
        print('Trades Executed [#] {:.2f}'.format(self.trades))
        print('=' * 55)
        
if __name__ == '__main__':
    bb = BacktestBase('AAPL.O','2010-1-1','2019-12-31',10000)
    print(bb.data.info())
    print(bb.data.info())
    bb.plot_data()
        
        
            
            
            
            
        
        
            
        
            
        
        
         
         
        
            
   