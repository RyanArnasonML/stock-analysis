# -*- coding: utf-8 -*-
"""
Classe for technical analysis of assets.
Created on Sat Oct 31 19:35:28 2020

@author: ryanar
"""

import numpy as np
import pandas as pd

from pylab import mpl , plt
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

class BacktestBase(object):

    #Constants 
    POSITION_LONG = 1
    POSITION_NONE = 0
    POSITION_SHORT = -1
    
    def __init__(self, symbol, startDate, endDate, cash, ftc=0.0, ptc=0.0, verbose=True):
        self.symbol = symbol
        self.startDate = startDate
        self.endDate = endDate
        self.initial_cash = cash
        self.cash = cash
        self.ftc = ftc
        self.ptc = ptc
        self.shares = 0
        self.trades = 0
        self.verbose = verbose
        self.get_data()
        
    def get_data(self):
        raw = pd.read_csv('http://hilpisch.com/pyalgo_eikon_eod_data.csv', index_col = 0 , parse_dates = True ).dropna() 
        raw = pd.DataFrame(raw[self.symbol])
        raw = raw.loc[self.startDate:self.endDate] 
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
        net_wealth = self.shares * price + self.cash  
        print(f'{date} | Cash: ${self.cash:.2f} Net wealth: ${net_wealth:.2f}')    
        
    def go_long(self, bar, shares=None, cash=None):
        if self.position == self.POSITION_SHORT:
            self.place_buy_order(bar, shares=-self.shares)
        if shares:
            self.place_buy_order(bar, shares=shares)
        elif cash:
            if cash == 'all':
                cash = self.cash
            self.place_buy_order(bar, cash=cash)
            
    def go_short(self, bar, shares=None, cash=None):
        if self.position == self.POSITION_LONG:
            self.place_sell_order(bar, shares=self.shares)
        if shares:
            self.place_sell_order(bar, shares=shares)
        elif cash:
            if cash == 'all':
                cash = self.cash
            self.place_sell_order(bar, cash=cash)    
        
    def place_buy_order(self, bar, shares=None, cash=None):
        
        date, price = self. get_date_price(bar)
        
        if shares is None:
            shares = int(cash/price)
            
        self.cash -=(shares * price) * (1 + self.ptc) + self.ftc
        self.shares += shares
        self.trades += 1
        
        if self.verbose:
            print(f'{date} | Buying {self.shares} shares @ ${price:.2f}')
            self.print_balance(bar)            
            
    def place_sell_order(self, bar, shares=None, cash=None):
        
        date, price = self.get_date_price(bar)
        
        if shares is None:
            shares = int(cash/price)
        
        self.cash += (shares * price) * (1 - self.ptc) - self.ftc
        self.shares -= shares
        self.trades += 1
        
        if self.verbose:
            print(f'{date} | Selling {shares} shares @ ${price:.2f}')
            self.print_balance(bar)            
        
    def close_out(self, bar):
        date, price = self. get_date_price(bar)
        self.cash += self.shares * price
        self.shares = 0
        self.trades += 1
        
        if self.verbose:
            print(f'{date} | inventory {self.shares} shares @ ${price:.2f}')
            print('=' * 55)
            
        print('Final balance [$] {:.2f}'.format(self.cash))    
        
        perf = ((self.cash - self.initial_cash) / self.initial_cash * 100)
        
        print('Net Performance [%] {:.2f}'.format(perf))
        print('Trades Executed [#] {:.2f}'.format(self.trades))
        print('=' * 55)


class BacktestLongOnly(BacktestBase):
    def run_sma_strategy(self, SMA1, SMA2):
        msg = f'\n\nRunning SMA strategy | SMA1 = { SMA1} & SMA2 = {SMA2}'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        print(msg)
        print('=' * 55)

        # Initialize the variables
        self.trades = 0
        self.cash = self.initial_cash
        self.position = self.POSITION_NONE
        
        # Get the data for the trading strategy   
        self.data['SMA1'] = self.data['price'].rolling(SMA1).mean()
        self.data['SMA2'] = self.data['price'].rolling(SMA2).mean()
        
        # The logic for the trading strategy
        for bar in range(SMA2, len(self.data)):
            
            if self.position == self.POSITION_NONE:
                
                if self.data['SMA1'].iloc[bar] > self.data['SMA2'].iloc[bar]:
                    self.place_buy_order(bar, cash=self.cash)
                    self.position = self.POSITION_LONG

            elif self.position == self.POSITION_LONG:
                
                if self.data['SMA1'].iloc[bar] < self.data['SMA2'].iloc[bar]:
                    self.place_sell_order(bar, shares=self.shares)
                    self.position = self.POSITION_NONE

        self.close_out(bar)
    
    def run_momentum_strategy(self, momentum):
        
        msg = f'\n\nRunning momentum stratedy | {momentum} days'
        msg += f'\nfixed costs {self.ftc} |'
        msg += f'proportional costs {self.ptc}'
        print(msg)
        print('=' * 55)
        
        # Initialize the variables
        self.trades = 0
        self.cash = self.initial_cash
        self.position = self.POSITION_NONE
        
        # Get the data for the trading strategy 
        self.data['momentum'] = self.data['return'].rolling(momentum).mean()
        
        # The logic for the trading strategy
        for bar in range(momentum, len(self.data)):
            
            if self.position == self.POSITION_NONE:
            
                if self.data['momentum'].iloc[bar] > 0:
                    self.place_buy_order(bar, cash=self.cash)
                    self.position = self.POSITION_LONG

            elif self.position == self.POSITION_LONG:

                if self.data['momentum'].iloc[bar] < 0:
                    self.place_sell_order(bar, shares=self.shares)
                    self.position = self.POSITION_NONE
        
        self.close_out(bar)
        
    def run_mean_reversion_strategy(self, SMA, threshold):
        
        msg = '\n\nRunning mean reversion strategy | '
        msg += f'SMA = {SMA} & Threshold = {threshold}'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        print(msg)
        print('=' * 55)

        # Initialize the variables
        self.trades = 0
        self.cash = self.initial_cash
        self.position = self.POSITION_NONE

        # Get the data for the trading strategy 
        self.data['SMA'] = self.data['price'].rolling(SMA).mean()
        
        # The logic for the trading strategy
        for bar in range(SMA, len(self.data)):
            
            if self.position == self.POSITION_NONE:

                if(self.data['price'].iloc[bar] < self.data['SMA'].iloc[bar] - threshold):
                    self.place_buy_order(bar, cash=self.cash)
                    self.position = self.POSITION_LONG
            
            elif self.position == self.POSITION_LONG:

                if(self.data['price'].iloc[bar] < self.data['SMA'].iloc[bar] - threshold):
                    self.place_sell_order(bar, shares=self.shares)
                    self.position = self.POSITION_NONE

        self.close_out(bar) 

class BacktestLongShort(BacktestBase):       
        
    def run_sma_strategy(self, SMA1, SMA2):
        msg = f'\n\nRunning SMA strategy | SMA1 = { SMA1} & SMA2 = {SMA2}'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        print(msg)
        print('=' * 55)

        # Initialize the variables
        self.trades = 0
        self.cash = self.initial_cash
        self.position = self.POSITION_NONE

        # Get the data for the trading strategy 
        self.data['SMA1'] = self.data['price'].rolling(SMA1).mean()
        self.data['SMA2'] = self.data['price'].rolling(SMA2).mean()
        
        # The logic for the trading strategy
        for bar in range(SMA2, len(self.data)):

            if self.position in [self.POSITION_NONE, self.POSITION_SHORT]:

                if self.data['SMA1'].iloc[bar] > self.data['SMA2'].iloc[bar]:
                    self.go_long(bar, cash='all')
                    self.position = self.POSITION_LONG # Long position

            elif self.position in[self.POSITION_NONE, self.POSITION_LONG]:

                if self.data['SMA1'].iloc[bar] < self.data['SMA2'].iloc[bar]:
                    self.go_short(bar, cash='all')
                    self.position = self.POSITION_SHORT # Short position
        
        self.close_out(bar)
    
    def run_momentum_strategy(self, momentum):
        
        msg = f'\n\nRunning momentum strategy | {momentum} days'
        msg += f'\nfixed costs {self.ftc} |'
        msg += f'proportional costs {self.ptc}'
        print(msg)
        print('=' * 55)

        # Initialize the variables        
        self.trades = 0
        self.cash = self.initial_cash
        self.position = self.POSITION_NONE

        # Get the data for the trading strategy         
        self.data['momentum'] = self.data['return'].rolling(momentum).mean()

        # The logic for the trading strategy
        for bar in range(momentum, len(self.data)):
            if self.position in [0, self.POSITION_SHORT]:
                if self.data['momentum'].iloc[bar] > 0:
                    self.go_long(bar, cash='all')
                    self.position = self.POSITION_LONG
            elif self.position in [0, self.POSITION_LONG]:
                if self.data['momentum'].iloc[bar] <= 0:
                    self.go_short(bar, cash='all')
                    self.position = self.POSITION_SHORT
        
        self.close_out(bar)
        
    def run_mean_reversion_strategy(self, SMA, threshold):
        
        msg = '\n\nRunning mean reversion strategy | '
        msg += f'SMA = {SMA} & Threshold = {threshold}'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        print(msg)
        print('=' * 55)

        # Initialize the variables        
        self.trades = 0
        self.cash = self.initial_cash
        self.position = self.POSITION_NONE
        
        # Get the data for the trading strategy 
        self.data['SMA'] = self.data['price'].rolling(SMA).mean()
        
        # The logic for the trading strategy
        for bar in range(SMA, len(self.data)):
            
            if self.position == self.POSITION_NONE:
                
                if(self.data['price'].iloc[bar] < self.data['SMA'].iloc[bar] - threshold):
                    self.go_long(bar, cash=self.initial_cash)
                    self.position = self.POSITION_LONG

                elif(self.data['price'].iloc[bar] > self.data['SMA'].iloc[bar] + threshold):
                    self.go_short(bar, cash=self.initial_cash)
                    self.position = self.POSITION_SHORT

            elif self.position == self.POSITION_LONG:

                if self.data['price'].iloc[bar] >= self.data['SMA'].iloc[bar]:
                    self.place_sell_order(bar, shares=self.shares)
                    self.position = self.POSITION_NONE

            elif self.position == self.POSITION_SHORT:

                if self.data['price'].iloc[bar] <= self.data['SMA'].iloc[bar]:
                    self.place_buy_order(bar, shares=-self.shares)
                    self.position = self.POSITION_NONE
       
        self.close_out(bar)                       
                     
                
if __name__ == '__main__':

    def run_strategies():
        lsbt.run_sma_strategy(42,252)
        # lsbt.run_momentum_strategy(60)
        # lsbt.run_mean_reversion_strategy(50,5)
    
lsbt = BacktestLongShort('AAPL.O','2010-1-1','2019-12-31',10000, verbose=True)
run_strategies()
    
# print(bb.data.info())
# print(bb.data.info())
lsbt.plot_data()                    
        
        
           
            
            
            
        
        
            
        
            
        
        
         
         
        
            
   