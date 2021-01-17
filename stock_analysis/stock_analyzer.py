# -*- coding: utf-8 -*-
"""
Classe for technical analysis of assets.
Created on Sat Oct 31 19:35:28 2020

@author: ryanar
"""

import math
import scipy.stats as scs
import numpy as np
from .utils import validate_df

class StockAnalyzer:
    """ Class for providing metrics for technical analysis of a stock. """
    
    @validate_df(columns={'open','high','low','close'})
    def __init__(self,df):
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
        self.data = df
        
    @property
    def _max_periods(self):
        """
        Get the maximum number of trading periods that can be used in calculations.

        Returns
        -------
        TYPE
                DESCRIPTION.

            """
        return self.data.shape[0]
        
    @property
    def close(self):
        """
            

        Returns
        -------
        TYPE
                Return the close column of the data.

        """
        return self.data.close
        
    @property
    def pct_change(self):
        """
    
        Returns
        -------
        TYPE
            Return the percent change of the close column.
        """
        return self.close.pct_change()
    
    def get52WeekHigh(self):
        """
        

        Returns
        -------
        Float
            Returns the hightest price in the last year.

        """
        return self.data[-252:].close.max()
    
    
    def get52WeekLow(self):
        """
        

        Returns
        -------
        Float
            Returns the lowest price in the last year.

        """
        return self.data[-252:].close.min()
    
    def normalityTests(self):
        
        # Convert the close price into log valves
        logReturn = np.log(self.data.close / self.data.close.shift(1))
        
        # The shift command will add a NaN to the beginning of the data, this will need to be removed.
        logReturn = logReturn.dropna()
        
        print('Skew of data set  %14.3f' % scs.skew(logReturn))
        print('Skew test p-value %14.3f' % scs.skewtest(logReturn)[1])
        print('Kurt of data set  %14.3f' % scs.kurtosis(logReturn))
        print('Kurt test p-value %14.3f' % scs.kurtosistest(logReturn)[1])
        print('Norm test p-value %14.3f' % scs.normaltest(logReturn)[1])     
        
    @property
    def pivot_point(self):
        """
            

        Returns
        -------
        Return the pivot point for support/resistance calculations.

        """
        return (self.last_close + self.last_high + self.last_low) / 3
        
    @property
    def last_close(self):
        """
            

        Returns
        -------
        TYPE
            Return the value of the last close in the data.

        """
        return self.data.last('1D').close.iat[0]
        
    @property
    def last_high(self):
        """
            

        Returns
        -------
        TYPE
            Return the value of the last high in the data.

        """
        return self.data.last('1D').high.iat[0]
        
    @property
    def last_low(self):
        """
            

        Returns
        -------
        TYPE
            Return the value of the last low in the data.

        """
        return self.data.last('1D').low.iat[0]
        
    def resistance(self, level=1):
        """
        Calculate the resistance at the given level.

        Parameters
        ----------
        level : TYPE, optional
            The resistance level. The default is 1.

        Raises
        ------
        ValueError
            Not a valid level. Must be 1, 2, or 3

        Returns
        -------
        res : TYPE
            The resistance value.

        """
        if level == 1:
            res = (2 * self.pivot_point) - self.last_low
        elif level == 2:
            res = self.pivot_point + (self.last_high - self.last_low)
        elif level == 3:
            res = self.last_high + 2 * (self.pivot_point - self.last_low)
        else:
            raise ValueError('Not a valid level. Must be 1, 2, or 3')
            
        return res
        
    def support(self, level=1):
        """
        Calculate the support at the given level.

        Parameters
        ----------
        level : TYPE, optional
            DESCRIPTION. The default is 1.

        Raises
        ------
        ValueError
            Not a valid level. Must be 1, 2, or 3.

        Returns
        -------
        sup : TYPE
            The support value.

        """
        if level == 1:
            sup = (2 * self.pivot_point) - self.last_high
        elif level == 2:
            sup = self.pivot_point - (self.last_high - self.last_low)
        elif level == 3:
            sup = self.last_low -2*(self.last_high - self.pivot_point)
        else:
            raise ValueError('Not a valid level. Must be 1, 2, or 3')
                
        return sup
        
    def daily_std(self, periods=252):
        """
        Calculate the daily standard deviation of percent change.

        Parameters
        ----------
        periods : TYPE, optional
            The number of periods to use for the calculation. The default is 252 for the trading days in a year.

        Returns
        -------
        TYPE
            The standard deviation

        """
        return self.pct_change[min(periods, self._max_periods) * -1:].std()
        
    def annualized_volatility(self):
        """
        Calculate the annualized volatility.

        Returns
        -------
        TYPE
            yearly volatility.

        """
        return self.daily_std() * math.sqrt(252)
        
    def volatility(self, periods=252):
        """
        Calculate the rolling volatility.

        Parameters
        ----------
        periods : TYPE, optional
            The number of periods to use for the calculation. The default is 252 for the trading days in a year.

        Returns
        -------
        Series
            A pandas series containing the rolling volatility.

        """
        periods = min(periods, self._max_periods)
        return self.close.rolling(periods).std() / math.sqrt(periods)
        
    def corr_with(self, other):
        """
        Calculate the correlations between this dataframe and another for matching columns.

        Parameters
        ----------
        other : Dataframe
            The other Dataframe used for the comparision.

        Returns
        -------
        Series
            A pandas series containing the correlations.

        """
        return self.data.corrwith(other)
        
    def cv(self):
        """
        Calculate the coefficient of variation for the asset. Note
        that the lower this is, the better the risk/return tradeoff.

        Returns
        -------
        Float
            The coefficient of variation for the asset.

        """
        return self.close.std()/self.close.mean()
        
    def qcd(self):
        """
        Calculate the quantile coefficient of dispersion.

        Returns
        -------
        None.

        """
        q1, q3 = self.close.quantile([0.25, 0.75])
        return(q3 - q1) / (q3 + q1)
        
    def beta(self, index):
        """
                        
        Beta is a measure of the volatility—or systematic risk—of a security or portfolio compared to the market as a whole.
            
        Beta < 1 is theoretically less volatile than the market, and tends to move more slowly than the market.
        Beta = 1 is strongly correlated with the market.
        Beta > 1 is theoretically more volatile than the market, but with increased return.

        Parameters
        ----------
        index : Dataframe
            The dataframe for the index to compare to.

        Returns
        -------
        beta : Float
            Return the calculated beta of the asset.

        """
        index_change = index.close.pct_change()
        beta = self.pct_change.cov(index_change) / index_change.var()
        return beta
        
    def cumulative_returns(self):
        """
        Calculate the series of cumulative returns for plotting.

        Returns
        -------
        Series
            cumulative returns

        """
        return (1 + self.pct_change).cumprod()
        
    @staticmethod
    def port_return(df):
        """
        Calculate the return assuming no distribution per share.

        Parameters
        ----------
        df : Dataframe
            Contains the stock close and index is the Date.

        Returns
        -------
        Float
            DESCRIPTION.

        """
        start, end = df.close[0], df.close[-1]
        return (end - start) / start
        
    def alpha(self, index, r_f):
        """
        Calculates the asset's alpha.
        Parameters:
            - index: The index to compare to.
            - r_f: The risk-free rate of return.
        Returns:
            Alpha, as a float.
        """
        r_f /= 100
        r_m = self.port_return(index)
        beta = self.beta(index)
        r = self.port_return(self.data)
        alpha = r - r_f - beta * (r_m - r_f)
        return alpha

    def is_bear_market(self):
        """
        Determine if a stock is in a bear market, meaning its
        return in the last 2 months is a decline of 20% or more.
        """
        return self.port_return(self.data.last('2M')) <= -.2

    def is_bull_market(self):
        """
        Determine if a stock is in a bull market, meaning its
        return in the last 2 months is a increase of 20% or more.
        """
        return self.port_return(self.data.last('2M')) >= .2

    def sharpe_ratio(self, r_f):
        """
        Calculates the asset's sharpe ratio.
        Parameters:
            - r_f: The risk-free rate of return.
        Returns:
            The sharpe ratio, as a float.
        """
        return (
            self.cumulative_returns().last('1D').iat[0] - r_f
        ) / self.cumulative_returns().std()
        
class AssetGroupAnalyzer:
    """ Analyzes many assets in a dataframe."""     
    @validate_df(columns={'open','high','low','close'})
    def __init__(self, df, group_by='name'):
        """
        Create a AssetGroupAnalyzer by passing in a pandas DataFrame and column to group by.

        Parameters
        ----------
        df : Dataframe
            Contains the stock open, high, low and close and index is the Date.
        group_by : String, optional
            Is used to sort the data, by a column name. The default is 'name'.

        Raises
        ------
        ValueError
            group_by not in dataframe.

        Returns
        -------
        None.

        """
        self.data = df
        if group_by not in self.data.columns:
            raise ValueError(f'`group_by` column "{group_by}" not in dataframe.')
        self.group_by = group_by
        self.analyzers = self._composition_handler()
        
    def _composition_handler(self):
        """
        Create a dictionary mapping each group to its analyzer, taking advantage of composition instead of inheritance.

        Returns
        -------
        dict
            Returns a dictionary mapping each group to its analyzer.

        """
        return {
            group : StockAnalyzer(data) \
            for group, data in self.data.groupby(self.group_by)
        }
    
    def analyze(self, func_name, **kwargs):
        """
        Run a StockAnalyzer method on all assets in the group.

        Parameters
        ----------
        func_name : TYPE
            The name of the method to run.
        **kwargs : TYPE
            Additional keyword arguments to pass to the function.

        Raises
        ------
        ValueError
            StockAnalyzer has no method with that name.

        Returns
        -------
        dict
            A dictionary mapping each asset to the result of the calculation of that function.

        """
        if not hasattr(StockAnalyzer, func_name):
            raise ValueError(f'StockAnalyzer has no "{func_name}" method.')
        if not kwargs:
            kwargs = {}
        return {
            group : getattr(StockAnalyzer, func_name)(analyzer, **kwargs) \
            for group, analyzer in self.analyzers.items()
        }