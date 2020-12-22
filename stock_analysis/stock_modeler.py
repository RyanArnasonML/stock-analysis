# -*- coding: utf-8 -*-
"""
Simple time series modeling for stocks.

Created on Sat Oct 31 15:16:24 2020

@author: ryanar
"""

import matplotlib.pyplot as plt
import pandas as pd

from statsmodels.tsa.seasonal import seasonal_decompose

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

import itertools
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .utils import validate_df

class StockModeler:
    """ 
    Static method for modeling stocks.
    """
    
    def __init__(self):
        raise NotImplementedError("This class is to be used statically, don't instantiate it!")
    
    @staticmethod
    @validate_df(columns={'close'},instance_method=False)    
    def decompose(df, period=20, model="additive"):
        """
        Decompose the closing price of the stock into trend, seasonal, and remainder components.

        Parameters
        ----------
        df : Dateframe
            Contains the stock closing price and index is the Date.
        period : Unsigned Integer
            The number of periods in the frequency.
        model : String, optional
            How to compute the decomposition ('additive', 'multiplicative'). The default is "additive".

        Returns
        -------
        A statsmodels decomposition object.

        """
        return seasonal_decompose(df.close, model=model, period=period)
    
    @staticmethod
    @validate_df(columns={'close'}, instance_method=False)
    def arima(df, *, ar=10, i=1, ma=5, fit=True):
        """
        Create an ARIMA object for modeling time series.

        Parameters
        ----------
        df : Dataframe
            The dataframe should have the index as datetime and a column with stock closing prices.
        * : TYPE
            DESCRIPTION.
        ar : TYPE
            The autoregressive order (p).
        i : TYPE
            The differenced order (q).
        ma : TYPE
            The moving average order (d).
        fit : TYPE, optional
            Whether or not to return the fitted model. The default is True.

        Returns
        -------
        A statsmodels ARIMA object which you can use to fit and predict.

        """
        arima_model = ARIMA(df.close.asfreq('B').fillna(method='ffill'), order=(ar, i, ma))
        return arima_model.fit() if fit else arima_model
    
    @staticmethod
    @validate_df(columns={'close'}, instance_method=False)
    def arima_predictions(df, arima_model_fitted, start, end, plot=True, **kwargs):
        """
        Get ARIMA predictions as pandas Series or plot.

        Parameters
        ----------
        df : Dataframe
            The dataframe should have the index as datetime and a column with stock closing prices.
        arima_model_fitted : TYPE
            The fitted ARIMA model.
        start : TYPE
            The start date for the predictions.
        end : TYPE
            The end date for the predictions.
        plot : BOOLEAN, optional
            Selects the return type of the function. 
             False is pandas Series containing the predictions. 
             The default is True which will return a plot.
        **kwargs : Dictionary
            Additional keyword arguments to pass to the pandas plot() method.

        Returns
        -------
        A matplotlib Axes object or predictions as a Series depending on the value of the `plot` argument.

        """
        predicted_changes = arima_model_fitted.predict(start=start, end=end)
        
        predictions = pd.Series(predicted_changes, name='close').cumsum( ) + df.last('1D').close.iat[0]
        
        if plot:
            ax = df.close.plot(**kwargs)
            predictions.plot(ax=ax, style='r:', label='arima predictions')
            ax.legend()
            
        return ax if plot else predictions
    
    
    @validate_df(columns={'close'}, instance_method=False)
    def arima_grid_search(df, s):
        p = d = q = range(2)
        
        param_combinations = list(itertools.product(p, d, q))
        
        lowest_aic, pdq, pdqs = None, None, None
        
        total_interations = 0
        
        for order in param_combinations:
            seasonal_order = (p, q, d, s)
            total_interations += 1
            try:
                model = SARIMAX(df,
                                order=order,
                                seasonal_order = seasonal_order,
                                enforce_stationarity = False,
                                enforce_invertibility=False,
                                disp=False)
                model_result = model.fit(maxiter=200, disp=False)
                
                if not lowest_aic or model_result.aic < lowest_aic:
                    lowest_aic = model_result.aic
                    pdq, pdqs = order, seasonal_order
                    
            except Exception as ex:
                continue
        
        return lowest_aic, pdq, pdqs
        
    @validate_df(columns={'close'}, instance_method=False)
    def sarimax(df, s = 12):

        lowest_aic, order, seasonal_order = StockModeler.arima_grid_search(df.close, s)

        model = SARIMAX(
            df.close,
            order = order,
            seasonal_order = seasonal_order,
            enforce_stationarity = False,
            enforce_invertibility = False,
            disp = False
        )

        model_results = model.fit(maxiter=200, disp = False)

        print('Lowest AIC: %.3f'%lowest_aic)
        print(model_results.summary())
        print(model_results.resid.describe())

        model_results.plot_diagnostics(figsize=(12,8))
        
    #      n = len(df_settle.index)
    #      prediction = model_results.get_prediction(
    #      start=n-12*5, 
    #      end=n+5
    #      )
    # prediction_ci = prediction.conf_int()
        
         
    #     plt.figure(figsize=(12, 6))

    # ax = df_settle['2008':].plot(label='actual')
    # prediction_ci.plot(
    #     ax=ax, style=['--', '--'],
    #     label='predicted/forecasted')

    # ci_index = prediction_ci.index
    # lower_ci = prediction_ci.iloc[:, 0]
    # upper_ci = prediction_ci.iloc[:, 1]

    # ax.fill_between(ci_index, lower_ci, upper_ci,
    #     color='r', alpha=.1)

    # ax.set_xlabel('Time (years)')
    # ax.set_ylabel('Prices')

    # plt.legend()
    # plt.show()
    
    @staticmethod
    @validate_df(columns={'close'}, instance_method=False)
    def regression(df):
        """
        Create linear regression of time series data with a lag of 1.

        Parameters
        ----------
        df : Dataframe
            The dataframe should have the index as datetime and a column with stock closing prices.

        Returns
        -------
        X : datetime
            Dates
        Y : FLOAT
            The closing price of a stock or security.
        Object
            A fitted statsmodels linear regression.

        """
        X = df.close.shift().dropna()
        Y = df.close[1:]
        return X, Y, sm.OLS(Y, X).fit()
    
    @staticmethod
    @validate_df(columns={'close'},instance_method=False)
    def regression_predictions(df, model, start, end, plot=True, **kwargs):
        """
         Get linear regression predictions as pandas Series or plot.

        Parameters
        ----------
        df : Dataframe
            The dataframe should have the index as datetime and a column with stock closing prices.
        model : TYPE
            The fitted linear regression model.
        start : datetime
            The start date for the predictions.
        end : datetime
            The end date for the predictions.
        plot : Boolean, optional
            False is a pandas Series containing the predictions being return. The default is True, and is a plot being returned.
        **kwargs : Dictionary
            Additional keyword arguments to pass to the pandas plot() method.

        Returns
        -------
        A matplotlib Axes object or predictions as a Series depending on the value of the plot argument.

        """
        predictions = pd.Series(index=pd.date_range(start,end),name='close')
        last = df.last('1D').close
        for i, date in enumerate(predictions.index):
            
            if i == 0:
                pred = model.predict(last)
            else:
                pred = model.predict(predictions.iloc[i-1])
              
            predictions.loc[date] = pred[0]
                        
        if plot:
            ax = df.close.plot(**kwargs)
            predictions.plot(ax=ax, style='r', label='regression predictions')
            ax.legend()
            
        return ax if plot else predictions
    
    @staticmethod
    def plot_residuals(model_fitted):
        fig, axes = plt.subplots(1,2, figsize=(15, 5))
        residuals = pd.Series(model_fitted.resid, name='residuals')
        residuals.plot(style='bo', ax=axes[0], title='Residuals')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Residual')
        residuals.plot(kind='kde', ax=axes[1], title='Residuals KDE')
        axes[1].set_xlabel('Residual')
        return axes
        
            
    
                           
        