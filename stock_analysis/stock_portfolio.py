# -*- coding: utf-8 -*-
"""
Classe for technical analysis of assets.
Created on Sat Oct 31 19:35:28 2020

@author: ryanar
"""

import math
import matplotlib.pyplot as plt
from stock_analysis import StockReader, StockVisualizer, Technical, AssetGroupVisualizer, StockAnalyzer, AssetGroupAnalyzer, StockModeler 
from stock_analysis.utils import group_stocks, describe_group, make_portfolio
import numpy as np
import pandas as pd
import scipy.optimize as sco


class Asset:
    
    def __init__(self,name, ticker, assetClass='Stock', data=None):
        
        """
        Create a StockAnalyzer by passing in a pandas DataFrame of OHLC data.

        Parameters
        ----------
        df : Dataframe
            Contains the stock open, high, low and close and index is the Date.

        Returns
        -------
        None.

        """        
        self.name = name
        self.ticker = ticker
        self.assetClass = assetClass
        
        if(data):
            self.data = data()
        else:
            reader = StockReader('2019-01-01','2020-11-21')
            
            if(assetClass=='Stock'):
                self.data = reader.get_ticker_data(self.ticker)
            else:
                self.data = reader.get_bitcoin_data()
                    
        self.technical = Technical(self.data)
        self.analyzer = StockAnalyzer(self.data)
        self.visualizer = StockVisualizer(self.data)
    
    def getName(self):
        return self.name 
    
    def getTicker(self):
        return self.ticker  
    
    def getAssetClass(self):
        return self.assetClass  
    
    def getData(self):
        return self.data
    
    def getClose(self):
        return self.data.close
    
    def getLogReturn(self):
        
        logReturn = np.log(self.data / self.data.shift(1))
        
        return logReturn.dropna()
    
    def getNormalityTests(self):
        print("Normality Test for {}".format(self.name))
        print(30 * '-')
        return self.analyzer.normalityTests()      
    
    def getAnnualizedVolatility(self):
        return self.analyzer.annualized_volatility()
    
    def get52WeekHigh(self):
        return self.analyzer.get52WeekHigh()
    
    def get52WeekLow(self):
        return self.analyzer.get52WeekLow()
    
    def getCumulativeReturns(self):
        return self.analyzer.cumulative_returns()
    
    def getCV(self):
        return self.analyzer.cv()
    
    def getQCD(self):
        return self.analyzer.qcd()
    
    def getVHF(self):
        return self.technical.VerticalHorizontalFilter()
    
    def getADX(self):
        return self.technical.AverageDirectionalIndex()
    
    def getStopLoss(self):
        return self.technical.AverageTrueRangeStopLoss().iloc[-1]
    
    def plotTimeFrame(self,column='close'):
        
        ax = self.visualizer.evolution_over_time(
            column, 
            figsize=(10, 4), 
            legend=False, 
            title='%s closing price over time' % (self.name),
            )
        
        
        self.visualizer.add_reference_line(
            ax, 
            x=self.data.high.idxmax(), 
            color='k', 
            linestyle=':', 
            alpha=0.5,
            label=f'highest value ({self.data.high.idxmax():%b %d})'
            )
        
        ax.set_ylabel('Price ($)')
        
    def plotMovingAverage(self,column='close', average=['60D','200D']):
        
        ax = self.visualizer.moving_average(column, average, title='%s closing price over time' % (self.name))
              
        ax.set_ylabel('Price ($)')    

    def plotRenko(self):
        self.visualizer.renko()
        
    def plotAverageTrueRange(self):
        self.visualizer.AverageTrueRange()    

    def plotIchimokuCloud(self):
        self.visualizer.IchimokuCloud(self.name)           

    def plotVolume(self):
        self.visualizer.trade_volume()
        
    def plotAfterHours(self):
        self.visualizer.after_hours_trades()    

    def plotCandleStick(self):
        self.visualizer.candle_stick()  
        
    def plotQQPlot(self):
        self.visualizer.qqplot()
        
    def plotOnBalanceVolume(self,ewm):
        self.visualizer.OnBalanceVolume(ewm)
        
    def plotBollingerBands(self):
        self.visualizer.bollingerbands()  
        
    def plotATRTrainingStops(self, timeframe=14):
        self.visualizer.ATRTrainingStops(timeframe)    
        
    def plotRelativeStrengthIndex(self, timeframe=14):
        self.visualizer.RelativeStrengthIndex(timeframe)        
        
    def plotMACD(self):
        self.visualizer.MACD()       
    
    def plotHistogram(self, column='close'):
        self.visualizer.histogram(column)
        
    def plotOpenToClose(self):
        print("plotOpenToClose currently doesn't work.")        
        self.visualizer.open_to_close()
        plt.show()
        

class Portfolio:
    """ Class for providing analysis of a stock portfolio. """    
    
    def __init__(self):
        """
        Create a StockAnalyzer by passing in a pandas DataFrame of OHLC data.

        Parameters
        ----------
        df : Dataframe
            Contains the stock open, high, low and close and index is the Date.

        Returns
        -------
        None.

        """
        self.holdings = []
        self.groupVisualizer=AssetGroupVisualizer(self.holdings)
        self.logMeanReturn = 0
        self.logAnnualCov = 0
        
    def addAsset(self, name, ticker, assetClass='Stock', data=None):
        self.holdings.append(Asset(name, ticker,assetClass, data)) 
        
    def numberOfHoldings(self):
        return len(self.holdings)
         
    def getHoldings(self):        
        return self.holdings  

    def listHoldings(self):        
        for holding in self.holdings:            
            print("Name: %s, Ticker: %s, Asset Type: %s, 52 Week Low: %.2f, 52 Week High: %.2f" % (holding.getName(), holding.getTicker(), holding.getAssetClass(), holding.get52WeekLow(), holding.get52WeekHigh()))
    
    def _getLogClose(self):
        
        df = pd.DataFrame()
        
        for holding in self.holdings:
            df[holding.getName()] = holding.getClose()
            
        logReturn = np.log(df / df.shift(1))
        
        return logReturn.dropna()       
    
    def logReturnMean(self):               
        rets = self._getLogClose()
        self.logMeanReturn = rets.mean()
       
    
    def logAnnualVolatility(self):               
        rets = self._getLogClose()
        self.logAnnualCov = rets.cov() * 252
        
    
    # def min_func_sharpe(weights):
    #     return -port_ret(weights) / port_vol(weights)
    
    def efficientFrontier(self):
        cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
        print(cons)
        
        # Set the bounds for the 0-100%
        bnds = tuple((0, 1) for x in range(self.numberOfHoldings()))
                
        # Create a equal weight of all of the holdings
        equalWeighting = np.array(self.numberOfHoldings() * [1. / self.numberOfHoldings(),])  
        
        opts = sco.minimize(min_func_sharpe, equalWeighting, method='SLSQP', bounds=bnds, constraints=cons)
    
    # def volatility(self):
        # return np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))                
            
    def printStatistics(self):
        for holding in self.holdings:
            print("Name: %s, CV: %s, Annual Volatility: %.2f, qcd: %.2f" % (holding.getName(), holding.getCV(), holding.getAnnualizedVolatility(), holding.getQCD())) 
    
    def getStatistics(self):
        
        df = pd.DataFrame()
        
        for holding in self.holdings:
            
            data = {
                'Name': holding.getName(),
                'CV':holding.getCV(),
                'Annual Volatility':holding.getAnnualizedVolatility(),
                'qcd':holding.getQCD()
                }               
            
            stock = pd.Series(data)            
            df = df.append(stock, ignore_index=True)
            
        df.set_index('Name', drop=True, inplace=True)    
            
        return df

    def port_ret(self, weights):
        return np.sum(self.logMeanReturn * weights) * 252

    def port_vol(self, weights):
        return np.sqrt(np.dot(weights.T, np.dot(self.logAnnualCov, weights)))    
            
    def plotRenkoForHolding(self,ticker):        
        for holding in self.holdings:
            if (holding.getTicker() == ticker):
                holding.plotRenko()
                
    def plotSummary(self):
        self.logReturnMean()
        self.logAnnualVolatility()
       
        
        # Random Weighting of Holdings
        
        prets = []
        pvols = []
        
        for p in range (2500):  
            weights = np.random.random(self.numberOfHoldings())  
            weights /= np.sum(weights)
            prets.append(self.port_ret(weights))
            pvols.append(self.port_vol(weights))  
            
        prets = np.array(prets)
        pvols = np.array(pvols)
        
        # Efficient Frontier
        #self.efficientFrontier()


        tvols = []
        # This is the y-axis for the efficient frontier
        trets = np.linspace(0.10, 0.40, 50)
        
        bnds = tuple((0, 1) for x in range(self.numberOfHoldings()))
        
        cons = ({'type': 'eq', 'fun': lambda x:  self.port_ret(x) - tret}, {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
        
        equalWeighting = np.array(self.numberOfHoldings() * [1. / self.numberOfHoldings(),])  
                
        for tret in trets:
            res = sco.minimize(self.port_vol, equalWeighting, method='SLSQP', bounds=bnds, constraints=cons)  
            tvols.append(res['fun'])
        tvols = np.array(tvols)
        
        ind = np.argmin(tvols)  
        evols = tvols[ind:]  
        erets = trets[ind:]

        self.groupVisualizer.portfolioSummary(prets, pvols, evols, erets)
   
           

# Need to implement Portfolio Optimization        
# https://github.com/yhilpisch/py4fi2nd/blob/master/code/ch13/13_a_statistics.ipynb

# Constructing a Killer Investment Portfolio with Python
#https://medium.com/analytics-vidhya/constructing-a-killer-investment-portfolio-with-python-51f4f0d344be

# We could have got 41% from FTSE, here’s how — Python Backtesting #1
# I like that it had a buy and hold vs. algorithm
# https://medium.com/analytics-vidhya/backtesting-of-trading-strategy-with-technical-indicator-1-f782b252d873

#https://medium.com/datadriveninvestor/teaching-a-machine-to-trade-3ef31d5918b3
apple = Asset('Apple','AAPL')
#vhf = apple.getADX()

# apple.getNormalityTests()
# apple.plotTimeFrame()
# apple.plotAverageTrueRange()
# apple.plotRelativeStrengthIndex()
# apple.plotRenko() 
# apple.plotQQPlot()
# apple.plotTimeFrame()
# apple.plotMovingAverage()
# apple.plotOpenToClose()  
apple.plotOnBalanceVolume(20) 
# apple.plotBollingerBands() 
# apple.plotMACD() 
# apple.plotIchimokuCloud()  
# temp=apple.getStopLoss()  
# apple.plotATRTrainingStops()  


# bitcoin = Asset('Bitcoin', 'BTC-USD', assetClass='Crypto')
# bitcoin.plotMACD() 
# bitcoin.plotIchimokuCloud()
# bitcoin.plotRenko()     

        
# myPortfolio = Portfolio()
# myPortfolio.addAsset('Facebook','FB')
# myPortfolio.addAsset('Apple' ,'AAPL')
# myPortfolio.addAsset('Amazon','AMZN')
# myPortfolio.addAsset('Netflix','NFLX')
# myPortfolio.addAsset('Google','GOOG')

# myPortfolio.listHoldings()
# print(myPortfolio.numberOfHoldings())

# print(myPortfolio.getStatistics())
# myPortfolio.printStatistics()

# myPortfolio.plotRenkoForHolding('FB')        
