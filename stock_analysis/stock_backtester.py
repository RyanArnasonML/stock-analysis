# -*- coding: utf-8 -*-
"""
Classe for technical analysis of assets.
Created on Sat Oct 31 19:35:28 2020

@author: ryanar
"""

import numpy as np
import pandas as pd
import talib as ta 

from pylab import mpl , plt
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'


class BacktestBase(object):

    #Constants 
    POSITION_LONG = 1
    POSITION_NONE = 0
    POSITION_SHORT = -1
    
    def __init__(self, symbol, startDate, endDate, cash, ftc=0.0, ptc=0.0, allowShorting = False, verbose=True):
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
        self.allowShorting = allowShorting
        self.positions = []
        self.pnls = []
        self.get_data()
        
    def get_data(self):
        raw = pd.read_csv('http://hilpisch.com/pyalgo_eikon_eod_data.csv', index_col = 0 , parse_dates = True ).dropna() 
        raw = pd.DataFrame(raw[self.symbol])
        raw = raw.loc[self.startDate:self.endDate] 
        raw.rename (columns = {self.symbol:'price'}, inplace = True)
        raw['return'] = np.log(raw/raw.shift(1)) 
        self.data = raw.dropna()
        
    def plot_price(self):
        self.data['price'].plot(figsize=(10,6), title = self.symbol)  
        
    def plot_positions(self):
        # 
        date = self.data.index.values[-len(self.positions):]
        df = pd.DataFrame(data=[date,self.positions]) 
        df = df.transpose(copy=True)
        df.rename(columns={0:"Date", 1:"Position"}, inplace=True)
        df.set_index('Date',drop=True, inplace=True)                
        df.plot(figsize=(10,6), title = self.symbol)
        
    def plot_pnl(self):
        # 
        date = self.data.index.values[-len(self.pnls):]
        df = pd.DataFrame(data=[date,self.pnls]) 
        df = df.transpose(copy=True)
        df.rename(columns={0:"Date", 1:"PNL"}, inplace=True)
        df.set_index('Date',drop=True, inplace=True)                
        df.plot(figsize=(10,6), title = self.symbol)     
        
    def plot_data(self):
        strategyPlot = self.data.copy()
        strategyPlot.drop('return', axis=1, inplace=True)
        strategyPlot.plot(figsize=(10,6), title = self.symbol)       
        
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
        
        date, price = self.get_date_price(bar)
        
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
        self.position = self.POSITION_NONE
        self.positions.append(self.position) 
        
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
        self.positions = []
        self.position = self.POSITION_NONE
        self.positions.append(self.position)
        
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
            
            # Track the profit and loss of the stratagy.    
            date, price = self.get_date_price(bar)
            self.pnls.append(self.shares * price + self.cash)  
            
            # Track if the position in long or short. 
            self.positions.append(self.position)           

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
        self.positions = []
        self.position = self.POSITION_NONE
        self.positions.append(self.position)
        
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
            
            # Track the profit and loss of the stratagy    
            date, price = self.get_date_price(bar)
            self.pnls.append(self.shares * price + self.cash)  
            
            # Track if the position in long or short. 
            self.positions.append(self.position)           
        
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
        self.positions = []
        self.position = self.POSITION_NONE
        self.positions.append(self.position)

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
            
            # Track the profit and loss of the stratagy    
            date, price = self.get_date_price(bar)
            self.pnls.append(self.shares * price + self.cash)  
            
            # Track if the position in long or short. 
            self.positions.append(self.position)           

        self.close_out(bar) 

class BacktestLongShort(BacktestBase):
    
    def run_buy_hold_strategy(self):
        msg = '\n\nBuy and Hold strategy'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        print(msg)
        print('=' * 55)

        # Initialize the variables
        self.trades = 0
        self.cash = self.initial_cash
        self.positions = []
        self.position = self.POSITION_NONE
        self.positions.append(self.position)

        # Get the data for the trading strategy 
        self.data['Hold'] = self.data['price'].rolling(1).mean()
        
        # The logic for the trading strategy
        for bar in range(1, len(self.data)):

            if self.position in [self.POSITION_NONE, self.POSITION_SHORT]:
                self.go_long(bar, cash='all')
                self.position = self.POSITION_LONG # Long position                        
                 
            # Track the profit and loss of the stratagy    
            date, price = self.get_date_price(bar)
            self.pnls.append(self.shares * price + self.cash)      
                    
            # Track if the position in long or short. 
            self.positions.append(self.position)           
        
        self.close_out(bar)    
       
    def run_sma_strategy(self, SMA1, SMA2):
        msg = f'\n\nRunning SMA strategy | SMA1 = { SMA1} & SMA2 = {SMA2}'
        msg += f'\nFixed costs: {self.ftc} | '
        msg += f'Proportional costs: {self.ptc} |'
        msg += f' Shorting Allowed: {self.allowShorting}'
        print(msg)
        print('=' * 55)

        # Initialize the variables
        self.trades = 0
        self.cash = self.initial_cash
        self.positions = []
        self.position = self.POSITION_NONE
        self.positions.append(self.position)

        # Get the data for the trading strategy 
        self.data['SMA1'] = self.data['price'].rolling(SMA1).mean()
        self.data['SMA2'] = self.data['price'].rolling(SMA2).mean()
        
        # The logic for the trading strategy
        for bar in range(SMA2, len(self.data)):
            if self.allowShorting:
                
                if self.position in [self.POSITION_NONE, self.POSITION_SHORT]:
                    
                    if self.data['SMA1'].iloc[bar] > self.data['SMA2'].iloc[bar]:
                        self.go_long(bar, cash='all')
                        self.position = self.POSITION_LONG # Long position               

                elif self.position in[self.POSITION_NONE, self.POSITION_LONG]:
                    
                    if self.data['SMA1'].iloc[bar] < self.data['SMA2'].iloc[bar]:
                        self.go_short(bar, cash='all')
                        self.position = self.POSITION_SHORT # Short position                
            
            else:
                
                if self.position == self.POSITION_NONE:
                
                    if self.data['SMA1'].iloc[bar] > self.data['SMA2'].iloc[bar]:
                        self.place_buy_order(bar, cash=self.cash)
                        self.position = self.POSITION_LONG               

                elif self.position == self.POSITION_LONG:
                    
                    if self.data['SMA1'].iloc[bar] < self.data['SMA2'].iloc[bar]:
                        self.place_sell_order(bar, shares=self.shares)
                        self.position = self.POSITION_NONE
                 
            # Track the profit and loss of the stratagy    
            date, price = self.get_date_price(bar)
            self.pnls.append(self.shares * price + self.cash)      
                    
            # Track if the position in long or short. 
            self.positions.append(self.position)           
        
        self.close_out(bar)
        
    def run_ema_strategy(self, EMA1, EMA2):
        msg = f'\n\nRunning EMA strategy | EMA1 = { EMA1} & EMA2 = {EMA2}'
        msg += f'\nFixed costs: {self.ftc} | '
        msg += f'Proportional costs: {self.ptc}'
        msg += f' Shorting Allowed: {self.allowShorting}'
        print(msg)
        print('=' * 55)

        # Initialize the variables
        self.trades = 0
        self.cash = self.initial_cash
        self.positions = []
        self.position = self.POSITION_NONE
        self.positions.append(self.position)

        # Get the data for the trading strategy 
        self.data['EMA1'] = self.data['price'].ewm(span=EMA1).mean()
        self.data['EMA2'] = self.data['price'].ewm(span=EMA2).mean()
        
        # The logic for the trading strategy
        for bar in range(EMA2, len(self.data)):

            if self.allowShorting:
                
                if self.position in [self.POSITION_NONE, self.POSITION_SHORT]:
                    
                    if self.data['EMA1'].iloc[bar] > self.data['EMA2'].iloc[bar]:
                        self.go_long(bar, cash='all')
                        self.position = self.POSITION_LONG # Long position               

                elif self.position in[self.POSITION_NONE, self.POSITION_LONG]:
                    
                    if self.data['EMA1'].iloc[bar] < self.data['EMA2'].iloc[bar]:
                        self.go_short(bar, cash='all')
                        self.position = self.POSITION_SHORT # Short position                
            
            else:
                
                if self.position == self.POSITION_NONE:
                
                    if self.data['EMA1'].iloc[bar] > self.data['EMA2'].iloc[bar]:
                        self.place_buy_order(bar, cash=self.cash)
                        self.position = self.POSITION_LONG               

                elif self.position == self.POSITION_LONG:
                    
                    if self.data['EMA1'].iloc[bar] < self.data['EMA2'].iloc[bar]:
                        self.place_sell_order(bar, shares=self.shares)
                        self.position = self.POSITION_NONE                
                 
            # Track the profit and loss of the stratagy    
            date, price = self.get_date_price(bar)
            self.pnls.append(self.shares * price + self.cash)      
                    
            # Track if the position in long or short. 
            self.positions.append(self.position)           
        
        self.close_out(bar) 
        
    def run_dema_strategy(self, DEMA1, DEMA2):
        msg = f'\n\nRunning DEMA strategy | DEMA1 = {DEMA1} & DEMA2 = {DEMA2}'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        print(msg)
        print('=' * 55)

        # Initialize the variables
        self.trades = 0
        self.cash = self.initial_cash
        self.positions = []
        self.position = self.POSITION_NONE
        self.positions.append(self.position)

        # Get the data for the trading strategy 
        self.data['DEMA1'] = 2 * self.data['price'].ewm(span=DEMA1).mean() - self.data['price'].ewm(span=DEMA1).mean()
        self.data['DEMA2'] = 2 * self.data['price'].ewm(span=DEMA2).mean() - self.data['price'].ewm(span=DEMA2).mean()
        
        # The logic for the trading strategy
        for bar in range(DEMA2, len(self.data)):

            if self.position in [self.POSITION_NONE, self.POSITION_SHORT]:

                if self.data['DEMA1'].iloc[bar] > self.data['DEMA2'].iloc[bar]:
                    self.go_long(bar, cash='all')
                    self.position = self.POSITION_LONG # Long position               

            elif self.position in[self.POSITION_NONE, self.POSITION_LONG]:

                if self.data['DEMA1'].iloc[bar] < self.data['DEMA2'].iloc[bar]:
                    self.go_short(bar, cash='all')
                    self.position = self.POSITION_SHORT # Short position                
                 
            # Track the profit and loss of the stratagy    
            date, price = self.get_date_price(bar)
            self.pnls.append(self.shares * price + self.cash)      
                    
            # Track if the position in long or short. 
            self.positions.append(self.position)           
        
        self.close_out(bar)
       
    def run_rsi_strategy(self, timeperiod=14):
        msg = f'\n\nRunning Relative Strength Index strategy | RSI = {timeperiod}'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        print(msg)
        print('=' * 55)

        # Initialize the variables
        self.trades = 0
        self.cash = self.initial_cash
        self.positions = []
        self.position = self.POSITION_NONE
        self.positions.append(self.position)

        # Get the data for the trading strategy 
        self.data['RSI'] = ta.RSI(self.data['price'], timeperiod)
        
        # The logic for the trading strategy
        for bar in range(timeperiod, len(self.data)):

            if self.position in [self.POSITION_NONE, self.POSITION_SHORT]:

                if self.data['RSI'].iloc[bar] > 30:
                    self.go_long(bar, cash='all')
                    self.position = self.POSITION_LONG # Long position               

            elif self.position in[self.POSITION_NONE, self.POSITION_LONG]:

                if self.data['RSI'].iloc[bar] < 70:
                    self.go_short(bar, cash='all')
                    self.position = self.POSITION_SHORT # Short position                
                 
            # Track the profit and loss of the stratagy    
            date, price = self.get_date_price(bar)
            self.pnls.append(self.shares * price + self.cash)      
                    
            # Track if the position in long or short. 
            self.positions.append(self.position)           
        
        self.close_out(bar)     
        
    def run_macd_strategy(self, MACD1, MACD2):
        msg = f'\n\nRunning MACD strategy | MACD1 = {MACD1} & MACD2 = {MACD2}'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        print(msg)
        print('=' * 55)

        # Initialize the variables
        self.trades = 0
        self.cash = self.initial_cash
        self.positions = []
        self.position = self.POSITION_NONE
        self.positions.append(self.position)

        # Get the data for the trading strategy 
        self.data['ShortEMA'] = self.data['price'].ewm(span=MACD1).mean()
        self.data['LongEMA'] = self.data['price'].ewm(span=MACD2).mean()
        
        self.data['MACD'] = self.data['ShortEMA'] - self.data['LongEMA']
        
        # Calcualte the signal line
        self.data['signal'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
        
        plt.plot(self.data.index[-365:], self.data['MACD'][-365:])
        plt.plot(self.data.index[-365:], self.data['signal'][-365:])
        
        # The logic for the trading strategy
        for bar in range(1, len(self.data)):

            if self.position in [self.POSITION_NONE, self.POSITION_SHORT]:

                if self.data['signal'].iloc[bar] < self.data['MACD'].iloc[bar]:
                    self.go_long(bar, cash='all')
                    self.position = self.POSITION_LONG # Long position               

            elif self.position in[self.POSITION_NONE, self.POSITION_LONG]:

                if self.data['signal'].iloc[bar] > self.data['MACD'].iloc[bar]:
                    self.go_short(bar, cash='all')
                    self.position = self.POSITION_SHORT # Short position                
                 
            # Track the profit and loss of the stratagy    
            date, price = self.get_date_price(bar)
            self.pnls.append(self.shares * price + self.cash)      
                    
            # Track if the position in long or short. 
            self.positions.append(self.position)           
        
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
        self.positions = []
        self.position = self.POSITION_NONE
        self.positions.append(self.position)

        # Get the data for the trading strategy         
        self.data['momentum'] = self.data['return'].rolling(momentum).mean()

        # The logic for the trading strategy
        for bar in range(momentum, len(self.data)):

            if self.allowShorting:

                if self.position in [0, self.POSITION_SHORT]:
                    if self.data['momentum'].iloc[bar] > 0:
                        self.go_long(bar, cash='all')
                        self.position = self.POSITION_LONG               
                 
                elif self.position in [0, self.POSITION_LONG]:
                    if self.data['momentum'].iloc[bar] <= 0:
                        self.go_short(bar, cash='all')
                        self.position = self.POSITION_SHORT
                    
                if self.position == self.POSITION_NONE:
                    if self.data['momentum'].iloc[bar] > 0:
                        self.place_buy_order(bar, cash=self.cash)
                        self.position = self.POSITION_LONG

                elif self.position == self.POSITION_LONG:
                    if self.data['momentum'].iloc[bar] < 0:
                        self.place_sell_order(bar, shares=self.shares)
                        self.position = self.POSITION_NONE

            else:
                
                if self.position == self.POSITION_NONE:

                    if self.data['momentum'].iloc[bar] > 0:
                        self.place_buy_order(bar, cash=self.cash)
                        self.position = self.POSITION_LONG

                elif self.position == self.POSITION_LONG:

                    if self.data['momentum'].iloc[bar] < 0:
                        self.place_sell_order(bar, shares=self.shares)
                        self.position = self.POSITION_NONE        
                    
                
            # Track the profit and loss of the stratagy    
            date, price = self.get_date_price(bar)
            self.pnls.append(self.shares * price + self.cash)  
            
            # Track if the position in long or short.        
            self.positions.append(self.position)   
            
        self.close_out(bar)
       
    # def run_mean_reversion_strategy(self, SMA, threshold):
        
    #     msg = '\n\nRunning mean reversion strategy | '
    #     msg += f'SMA = {SMA} & Threshold = {threshold}'
    #     msg += f'\nfixed costs {self.ftc} | '
    #     msg += f'proportional costs {self.ptc}'
    #     print(msg)
    #     print('=' * 55)

    #     # Initialize the variables        
    #     self.trades = 0
    #     self.cash = self.initial_cash
    #     self.positions = []
    #     self.position = self.POSITION_NONE
    #     self.positions.append(self.position)
        
    #     # Get the data for the trading strategy 
    #     self.data['SMA'] = self.data['price'].rolling(SMA).mean()
        
    #     # The logic for the trading strategy
    #     for bar in range(SMA, len(self.data)):

    #         if AllowShorting:

    #             if self.position == self.POSITION_NONE:
                
                    
    #                 if(self.data['price'].iloc[bar] < self.data['SMA'].iloc[bar] - threshold):
    #                     self.go_long(bar, cash=self.initial_cash)
    #                     self.position = self.POSITION_LONG
                    
    #                 elif(self.data['price'].iloc[bar] > self.data['SMA'].iloc[bar] + threshold):
    #                     self.go_short(bar, cash=self.initial_cash)
    #                     self.position = self.POSITION_SHORT                

    #             elif self.position == self.POSITION_LONG:

    #                 if self.data['price'].iloc[bar] >= self.data['SMA'].iloc[bar]:
    #                     self.place_sell_order(bar, shares=self.shares)
    #                     self.position = self.POSITION_NONE                 

    #             elif self.position == self.POSITION_SHORT:

    #                 if self.data['price'].iloc[bar] <= self.data['SMA'].iloc[bar]:
    #                     self.place_buy_order(bar, shares=-self.shares)
    #                     self.position = self.POSITION_NONE

    #         else:

    #             if self.position == self.POSITION_NONE:

    #                 if(self.data['price'].iloc[bar] < self.data['SMA'].iloc[bar] - threshold):
    #                     self.place_buy_order(bar, cash=self.cash)
    #                     self.position = self.POSITION_LONG
            
    #             elif self.position == self.POSITION_LONG:

    #                 if(self.data['price'].iloc[bar] < self.data['SMA'].iloc[bar] - threshold):
    #                     self.place_sell_order(bar, shares=self.shares)
    #                     self.position = self.POSITION_NONE
                   
                        
    #         # Track the profit and loss of the stratagy    
    #         date, price = self.get_date_price(bar)
    #         self.pnls.append(self.shares * price + self.cash) 
            
    #         # Track if the position in long or short. 
    #         self.positions.append(self.position) 
       
    #     self.close_out(bar)                       
                    
                
if __name__ == '__main__':

    def run_strategies():
        # lsbt.run_buy_hold_strategy()
        # lsbt.run_sma_strategy(10,50)
        #lsbt.run_sma_strategy(20,50)
        #lsbt.run_sma_strategy(50,200)
        #lsbt.run_sma_strategy(42,252)
        # lsbt.run_ema_strategy(12,26)
        # lsbt.run_ema_strategy(50,200)
        # lsbt.run_dema_strategy(20,50)
        #lsbt.run_rsi_strategy(14)
        # lsbt.run_macd_strategy(12,26)
        # lsbt.run_dema_strategy(50,200) 
        lsbt.run_momentum_strategy(60)        
        # lsbt.run_mean_reversion_strategy(50,5)
    
lsbt = BacktestLongShort('AAPL.O','2010-1-1','2019-12-31', 10000,allowShorting=True, verbose=False)
run_strategies()
    
# print(bb.data.info())
# print(bb.data.info())
lsbt.plot_data()                    
lsbt.plot_positions() 
lsbt.plot_pnl()      
        
           
            
            
            
        
        
            
        
            
        
        
         
         
        
            
   