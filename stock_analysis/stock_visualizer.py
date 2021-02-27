# -*- coding: utf-8 -*-
"""
Visualize financial instruments

Created on Sat Oct 31 14:15:43 2020

@author: ryanar
"""

import math

from stock_analysis import Technical
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import probplot

import scipy.interpolate as sci
import scipy.optimize as sco

from .utils import validate_df


class Visualizer:
    """ 
    Base visualizer class not intended for direct use.
    """
    
    @validate_df(columns={'open','high','low','close'})
    def __init__(self, df):
        """
        Visualizer Initializer

        Parameters
        ----------
        df : Dataframe
            The Dataframe contains the stock market data.

        Returns
        -------
        None.

        """
        self.data = df
        self.technical = Technical(df)
        
    @staticmethod
    def add_reference_line(ax, x=None, y=None, **kwargs):
        """
        Staic method for adding reference lines to plots

        Parameters
        ----------
        ax : matplotlib Axes Object
            The matplotlib Axes object to add teh reference line to.
        x : TYPE, optional
            For sloped line (x and y). The default is None for a horizontal line with a y.
        y : TYPE, optional
            For sloped line (x and y). The default is None for a vertical line with a x.
        **kwargs : TYPE
            Additional keyword arguments to pass to the plotting function.

        Returns
        -------
        matplotlib Axes Object.

        """
        try: 
            # In case a numpy array-like structures are passed -> AB Line
            if x.shape and y.shape:
                ax.plot(x,y, **kwargs)
        except:
            # A error triggers if at lease one isn't a numpy array
            try:
                if not x and not y:
                    raise ValueError('You must provide a x or a y value at a minimum!')
                elif x and not y:
                    # A vertical line
                    ax.axvline(x, **kwargs)
                elif not x and y:
                    # A horizontal line
                    ax.axhline(y, **kwargs)
            except:
                raise ValueError('If providing only x or y, it must be a single value.')
        ax.legend()
        return ax
    
    @staticmethod
    def shade_region(ax, x=tuple(),y=tuple(), **kwargs):
        """
        Static method for shading a region on a plot.

        Parameters
        ----------
        ax : matplotlib Axes Object
            A matplotlib Axes object to add the shaded region to.
        x : tuple, optional
            Tuple with the `xmin` and `xmax` bounds for the rectangle drawn vertically. The default is empty tuple().
        y : tuple, optional
            Tuple with the `ymin` and `ymax` bounds for the rectangle drawn vertically.. The default is empty tuple().
        **kwargs : TYPE
            Additional keyword arguments to pass to the plotting function.

        Returns
        -------
        The updated matplotlib Axes Object passed in.

        """
        if not x and not y:
            raise ValueError('You must provide an x or a y min/max tuple at a minimum!')
        elif x and y:
            raise ValueError('You can only provide x or y.')
        elif x and not y:
            # vertical span
            ax.axvspan(*x, **kwargs)
        elif not x and y:
            # horizontal span
            ax.axhspan(*y, **kwargs)
        return ax

    @staticmethod
    def _iter_handler(items):
        """
        Static method for making a list out of a item if it isn't a list or tuple already.        

        Parameters
        ----------
        items : TYPE
            The variable to make sure it is a list.

        Returns
        -------
        items : list or tuple
            DESCRIPTION.

        """        
        if not isinstance(items, (list, tuple)):
            items = [items]
        return items

    def _window_calc_func(self, column, periods, name, func, named_arg, **kwargs):
        """
        To be implemented by subclasses. Defines how to add lines resulting
        from window calculations.
        """
        raise NotImplementedError('To be implemented by subclasses!')

    def moving_average(self, column, periods, **kwargs):
        """
        Add line(s) for the moving average of a column.
        Parameters:
            - column: The name of the column ('close','open', 'high','low' and 'volume') to plot.
            - periods: The rule or list of rules for resampling,
                       like '20D' for 20-day periods.
            - kwargs: Additional arguments to pass down to the plotting function.
        Returns:
            A matplotlib Axes object.
        """
        return self._window_calc_func(column, periods, name='MA',func=pd.DataFrame.resample, named_arg='rule', **kwargs)

    def exp_smoothing(self, column, periods, **kwargs):
        """
        Add line(s) for the exponentially smoothed moving average of a column.
        Parameters:
            - column: The name of the column to plot.
            - periods: The span or list of spans for smoothing,
                       like 20 for 20-day periods.
            - kwargs: Additional arguments to pass down to the plotting function.
        Returns:
            A matplotlib Axes object.
        """
        return self._window_calc_func(column, periods, name='EWMA', func=pd.DataFrame.ewm, named_arg='span', **kwargs)

    # abstract methods for subclasses to define
    def evolution_over_time(self, column, **kwargs):
        """To be implemented by subclasses for generating line plots."""
        raise NotImplementedError('To be implemented by subclasses!')

    def boxplot(self, **kwargs):
        """To be implemented by subclasses for generating boxplots."""
        raise NotImplementedError('To be implemented by subclasses!')

    def histogram(self, column, **kwargs):
        """To be implemented by subclasses for generating histograms."""
        raise NotImplementedError('To be implemented by subclasses!')

    def after_hours_trades(self):
        """To be implemented by subclasses."""
        raise NotImplementedError('To be implemented by subclasses!')

    def pairplot(self, **kwargs):
        """To be implemented by subclasses for generating pairplots."""
        raise NotImplementedError('To be implemented by subclasses!')
                     
class StockVisualizer(Visualizer):
    """Visualizer for a single stock."""

    def evolution_over_time(self, column, **kwargs):
        """
        Visualize the evolution over time of a column.
        Parameters:
            - column: The name of the column to visualize.
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.
        Returns:
            A matplotlib Axes object.
        """
        return self.data.plot.line(y=column, **kwargs)

    def boxplot(self, **kwargs):
        """
        Generate boxplots for all columns.
        Parameters:
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.
        Returns:
            A matplotlib Axes object.
        """
        return self.data.plot(kind='box', **kwargs)
    
    def qqplot(self, **kwargs):
        """
        Generate boxplots for all columns.
        Parameters:
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.
        Returns:
            A matplotlib Axes object.
        """
        daily_changes = self.data.close.pct_change(periods=1).dropna()
        figure  = plt.figure(figsize=(8,4))
        ax = figure.add_subplot(111)
        stats.probplot(daily_changes,dist='norm', plot=ax)
        plt.show()
        #return self.data.plot(kind='box', **kwargs)

    def histogram(self, column, **kwargs):
        """
        Generate the histogram of a given column.
        Parameters:
            - column: The name of the column to visualize.
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.
        Returns:
            A matplotlib Axes object.
        """
        return self.data.plot.hist(y=column, **kwargs)

    def trade_volume(self, tight=False, **kwargs):
        """
        Visualize the trade volume and closing price.
        Parameters:
            - tight: Whether or not to attempt to match up the resampled
                     bar plot on the bottom to the line plot on the top.
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.
        Returns:
            A matplotlib Axes object.
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 15))
        self.data.close.plot(ax=axes[0], title='Closing Price').set_ylabel('price')
        monthly = self.data.volume.resample('1M').sum()
        monthly.index = monthly.index.strftime('%b\n%Y')
        monthly.plot(
            kind='bar', ax=axes[1], color='blue', rot=0, title='Volume Traded'
        ).set_ylabel('volume traded')
        if tight:
            axes[0].set_xlim(self.data.index.min(), self.data.index.max())
            axes[1].set_xlim(-0.25, axes[1].get_xlim()[1] - 0.25)
        return axes

    def after_hours_trades(self):
        """
        Visualize the effect of after hours trading on this asset.
        Returns:
            A matplotlib Axes object.
        """
        after_hours = (self.data.open - self.data.close.shift())

        monthly_effect = after_hours.resample('1M').sum()
        fig, axes = plt.subplots(1, 2, figsize=(15, 3))

        after_hours.plot(
            ax=axes[0],
            title='After hours trading\n(Open Price - Prior Day\'s Close)'
        ).set_ylabel('price')

        monthly_effect.index = monthly_effect.index.strftime('%b')
        monthly_effect.plot(
            ax=axes[1],
            kind='bar',
            title='After hours trading monthly effect',
            color=np.where(monthly_effect >= 0, 'g', 'r'),
            rot=90
        ).axhline(0, color='black', linewidth=1)
        axes[1].set_ylabel('price')
        return axes

    def open_to_close(self, figsize=(10, 4)):
        """
        Visualize the daily change from open to close price.
        Parameters:
            - figsize: A tuple of (width, height) for the plot dimensions.
        Returns:
            A matplotlib Figure object.
        """
        is_higher = self.data.close - self.data.open > 0

        fig = plt.figure(figsize=figsize)

        for exclude_mask, color, label in zip(
            (is_higher, np.invert(is_higher)),
            ('g', 'r'),
            ('price rose', 'price fell')
        ):
            plt.fill_between(
                self.data.index,
                self.data.open,
                self.data.close,
                figure=fig,
                where=exclude_mask,
                color=color,
                label=label
            )
        plt.suptitle('Daily price change (open to close)')
        plt.xlabel('date')
        plt.ylabel('price')
        plt.legend()
        plt.close()
        return fig
    
    def OnBalanceVolume(self, ewm=None,figsize=(10, 4)):         
        fig = plt.figure(figsize=figsize)
        if ewm is None:
            plt.plot(self.technical.OnBalanceVolume())
            plt.suptitle('On Balance Volume')
        else:
            plt.plot(self.technical.OBV_EMA(ewm))
            plt.axhline(y=0, color='black', linestyle='-.')
            plt.suptitle('On Balance Volume')
        plt.xlabel('date')
        plt.ylabel('OBV')
        plt.legend()
        plt.show()
        plt.close()
        return fig
    
    def AverageTrueRange(self, timeframe=14, figsize=(10, 4)):         
        """
        https://kaabar-sofien.medium.com/the-power-of-the-average-true-range-indicator-in-trading-5de3dcc811a9

        Parameters
        ----------
        timeframe : TYPE, optional
            DESCRIPTION. The default is 14.
        figsize : TYPE, optional
            DESCRIPTION. The default is (10, 4).

        Returns
        -------
        fig : TYPE
            DESCRIPTION.

        """
        
        fig = plt.figure(figsize=figsize)
        
        # Could add a moving average.
        plt.plot(self.technical.AverageTrueRange(timeframe), label='%s-period' % (timeframe))
        plt.suptitle('Average True Range')
        plt.xlabel('date')
        plt.ylabel('ATR')
        plt.legend()
        plt.show()
        plt.close()
        return fig
    
    def RelativeStrengthIndex(self, timeframe=14, figsize=(10, 4)):         
        fig = plt.figure(figsize=figsize)
        plt.plot(self.technical.RelativeStrengthIndex(timeframe), label='%s-period' % (timeframe))
        plt.suptitle('Relative Strength Index')
        plt.axhline(y=70, color='r', linestyle='--')
        plt.axhline(y=50, color='black', linestyle='-.')
        plt.axhline(y=30, color='g', linestyle='--')
        plt.xlabel('Date')        
        plt.legend()
        plt.show()
        plt.close()
        return fig
    
    def Schaff(self, timeframe=14, figsize=(10, 4)):                
        fig = plt.figure(figsize=figsize)
        plt.plot(self.technical.Schaff())
        plt.suptitle('Schaff Trend Cycle')        
        plt.axhline(y=50, color='r', linestyle='--')
        plt.axhline(y=25, color='g', linestyle='--')
        plt.xlabel('Date')        
        plt.legend()
        plt.show()
        plt.close()
        return fig
    
    def bollingerbands(self, figsize=(10, 4)):         
        fig = plt.figure(figsize=figsize)
        plt.plot(self.technical.BollingerBands())
        plt.suptitle('Bollinger Bands')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        plt.close()
        return fig
    
    def monthlyReturnsBoxplot(self, monthlyReturn, name, figsize=(10, 10)):         
        #fig = plt.figure(figsize=figsize)
       
        monthlyReturn=pd.DataFrame(monthlyReturn)
        
        monthlyReturn.boxplot(column='close', by='Month')
        ax = plt.gca()
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels=['Jan.','Feb.','Mar.','Apr.','May','June','July','Aug.','Sep.','Oct.','Nov.','Dec.']

        ax.set_xticklabels(labels)
        plt.title('of Percent Closing for %s'% (name))        
        plt.ylabel('Percent [%]')
        
        plt.autoscale(True, axis='x', tight=True)
        plt.show()
    
    def MACD(self):         
        
        results = self.technical.MovingAverageConvergenceDivergence()
        
        # Convert to string, to remove gaps caused by weekend in the datetime. 
        results.reset_index(inplace=True)
        results["date"] = results["date"].dt.strftime("%d-%b")
        
        
        fig, ax = plt.subplots(figsize=(10, 4))               
        ax.bar(results["date"], results['history'])
        ax.plot(results['signal'], label='signal')
        ax.plot(results['macd'], label='macd')
        fig.suptitle('MACD')
        plt.xlabel('Date')
        fig.autofmt_xdate()
        
        
        plt.autoscale(True, axis='x', tight=True)
        
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        plt.close()
        #return fig        
    
    def candle_stick(self, **kwargs):
        """
        Candlestick charts originated in Japan over 100 years before the West developed the bar and point-and-figure charts. In the 1700s, a Japanese man named Homma discovered that, while there was a link between price and the supply and demand of rice, the markets were strongly influenced by the emotions of traders.

        Parameters
        ----------
         : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        A candle stick plot of the last 30 days.

        """
        oneMonth = self.data[-30:]
        mpf.plot(oneMonth,type='candle',style='yahoo')
        
    def IchimokuCloud(self, assetName):
        
        results = self.technical.IchimokuCloud() 
        
        results = results[-90:]
                
        fig, ax = plt.subplots(figsize=(10, 4))        
        
        plt.plot(self.data.index.values[-90:], self.data.close[-90:],color='black', label='Close')
        plt.plot(results.index.values, results['conversionLine'],'--' ,label='Conversion Line')
        plt.plot(results.index.values, results['baseLine'], '--', label='Base Line', )
        
        green = results.copy()
        red = results.copy()
        
        green = green.query('leadingSpanA > leadingSpanB')
        red = red.query('leadingSpanA < leadingSpanB')
        
        plt.fill_between(
                green.index.values,
                green['leadingSpanA'],
                green['leadingSpanB'], color='green',
                alpha=0.2                
            )
        
        plt.fill_between(
                red.index.values,
                red['leadingSpanA'],
                red['leadingSpanB'], color='red',
                alpha=0.2                
            )
        
        plt.title('Ichimoku Cloud for %s' % (assetName))
        plt.ylabel('Price')
        plt.xlabel('Date')
        fig.autofmt_xdate()
        
        # plt.autoscale(True, axis='x', tight=True)
        
        plt.legend()
        plt.show()
        plt.close()
        
    def ATRTrainingStops(self,timeframe, figsize=(10, 4)):  
        fig = plt.figure(figsize=figsize)
        
        # Could add a moving average.
        plt.plot(self.data.close, label='Close')
        plt.plot(self.technical.AverageTrueRangeStopLoss(timeframe), label='%s-period' % (timeframe))
        plt.suptitle('Average True Range')
        plt.xlabel('date')
        plt.ylabel('ATR')
        plt.legend()
        plt.show()
        plt.close()
        return fig
        
    def renko(self, **kwargs):
        """
        A Renko chart is a type of chart, developed by the Japanese, that is built using price movement rather than both price and standardized time intervals like most charts are. It is thought to be named after the Japanese word for bricks, "renga," since the chart looks like a series of bricks. A new brick is created when the price moves a specified price amount, and each block is positioned at a 45-degree angle (up or down) to the prior brick. An up brick is typically colored white or green, while a down brick is typically colored black or red.

        Parameters
        ----------
         : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        A candle stick plot of the last 30 days.

        """        
        mpf.plot(self.data,type='renko',style='yahoo')    

    def fill_between_other(self, other_df, figsize=(10, 4)):
        """
        Visualize the difference in closing price between assets.
        Parameters:
            - other_df: The dataframe with the other asset's data.
            - figsize: A tuple of (width, height) for the plot dimensions.
        Returns:
            A matplotlib Figure object.
        """
        is_higher = self.data.close - other_df.close > 0

        fig = plt.figure(figsize=figsize)

        for exclude_mask, color, label in zip(
            (is_higher, np.invert(is_higher)),
            ('g', 'r'),
            ('asset is higher', 'asset is lower')
        ):
            plt.fill_between(
                self.data.index,
                self.data.close,
                other_df.close,
                figure=fig,
                where=exclude_mask,
                color=color,
                label=label
            )
        plt.suptitle(
            'Differential between asset closing price (this - other)'
        )
        plt.legend()
        plt.ylabel('price')
        plt.show()
        plt.close()
        return fig

    def _window_calc_func(self, column, periods, name, func, named_arg, **kwargs):
        """
        Helper method for plotting a series and adding reference lines using
        a window calculation.
        Parameters:
            - column: The name of the column to plot.
            - periods: The rule/span or list of them to pass to the
                       resampling/smoothing function, like '20D' for 20-day periods
                       (for resampling) or 20 for a 20-day span (smoothing)
            - name: The name of the window calculation (to show in the legend).
            - func: The window calculation function.
            - named_arg: The name of the argument `periods` is being passed as.
            - kwargs: Additional arguments to pass down to the plotting function.
        Returns:
            A matplotlib Axes object.
        """
        ax = self.data.plot(y=column, **kwargs)
        for period in self._iter_handler(periods):
            self.data[column].pipe(
                func, **{named_arg: period}
            ).mean().plot(
                ax=ax,
                linestyle='--',
                label=f'{period if isinstance(period, str) else str(period) + "D"} {name}'
            )
        plt.legend()
        return ax

    def correlation_heatmap(self, other):
        """
        Plot the correlations between the same column between this asset and
        another one with a heatmap.
        Parameters:
            - other: The other dataframe.
        Returns:
            A seaborn heatmap
        """
        corrs = self.data.corrwith(other)
        corrs = corrs[~pd.isnull(corrs)]
        size = len(corrs)
        matrix = np.zeros((size, size), float)
        for i, corr in zip(range(size), corrs):
            matrix[i][i] = corr

        # create mask to only show diagonal
        mask = np.ones_like(matrix)
        np.fill_diagonal(mask, 0)

        return sns.heatmap(
            matrix,
            annot=True,
            xticklabels=self.data.columns,
            yticklabels=self.data.columns,
            center=0,
            mask=mask
        )

    def pairplot(self, **kwargs):
        """
        Generate a seaborn pairplot for this asset.
        Parameters:
            - kwargs: Keyword arguments to pass down to `sns.pairplot()`
        Returns:
            A seaborn pairplot
        """
        return sns.pairplot(self.data, **kwargs)

    def jointplot(self, other, column, **kwargs):
        """
        Generate a seaborn jointplot for given column in asset compared to
        another asset.
        Parameters:
            - other: The other asset's dataframe
            - column: The column name to use for the comparison.
            - kwargs: Keyword arguments to pass down to `sns.pairplot()`
        Returns:
            A seaborn jointplot
        """
        return sns.jointplot(
            x=self.data[column],
            y=other[column],
            **kwargs
        )

class AssetGroupVisualizer(Visualizer):
    """Class for visualizing groups of assets in a single dataframe."""

    # override for group visuals
    def __init__(self, df, group_by='name'):
        """This object keeps track of which column it needs to group by."""
        # super().__init__(df)
        # self.group_by = group_by
        self.tck = 0

    def evolution_over_time(self, column, **kwargs):
        """
        Visualize the evolution over time of a column for all assets in group.
        Parameters:
            - column: The name of the column to visualize.
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.
        Returns:
            A matplotlib Axes object.
        """
        if 'ax' not in kwargs:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        else:
            ax = kwargs.pop('ax')
        return sns.lineplot(
            x=self.data.index,
            y=column,
            hue=self.group_by,
            data=self.data,
            ax=ax,
            **kwargs
        )

    def boxplot(self, column, **kwargs):
        """
        Generate boxplots for a given column in all assets.
        Parameters:
            - column: The name of the column to visualize.
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.
        Returns:
            A matplotlib Axes object.                       
        """ 
        return sns.boxplot(
            x=self.group_by,
            y=column,
            data=self.data,
            **kwargs
        )

    def _get_layout(self):
        """
        Helper method for getting an autolayout of subplots (1 per group).
        Returns:
            The matplotlib Figure and Axes objects to plot with.
        """
        subplots_needed = self.data[self.group_by].nunique()
        rows = math.ceil(subplots_needed / 2)
        fig, axes = plt.subplots(rows, 2, figsize=(15, 5*rows))
        if rows > 1:
            axes = axes.flatten()
        if subplots_needed < len(axes):
            # remove excess axes from autolayout
            for i in range(subplots_needed, len(axes)):
                # can't use comprehension here
                fig.delaxes(axes[i])
        return fig, axes

    def histogram(self, column, **kwargs):
        """
        Generate the histogram of a given column for all assets in group.
        Parameters:
            - column: The name of the column to visualize.
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.
        Returns:
            A matplotlib Axes object.
        """

        return sns.displot(self.data, x=column, col="name", col_wrap = 3, height=10, aspect = 1, legend=True, kde=True, kind='hist')
               

    def _window_calc_func(self, column, periods, name, func, named_arg, **kwargs):
        """
        Helper method for plotting a series and adding reference lines using
        a window calculation.
        Parameters:
            - column: The name of the column to plot.
            - periods: The rule/span or list of them to pass to the
                       resampling/smoothing function, like '20D' for 20-day periods
                       (for resampling) or 20 for a 20-day span (smoothing)
            - name: The name of the window calculation (to show in the legend).
            - func: The window calculation function.
            - named_arg: The name of the argument `periods` is being passed as.
            - kwargs: Additional arguments to pass down to the plotting function.
        Returns:
            A matplotlib Axes object.
        """
        fig, axes = self._get_layout()
        for ax, asset_name in zip(axes, self.data[self.group_by].unique()):
            subset = self.data[self.data[self.group_by] == asset_name]
            ax = subset.plot(y=column, ax=ax, label=asset_name, **kwargs)
            for period in self._iter_handler(periods):
                subset[column].pipe(
                    func, **{named_arg: period}
                ).mean().plot(
                    ax=ax,
                    linestyle='--',
                    label=f'{period if isinstance(period, str) else str(period) + "D"} {name}'
                )
            ax.legend()
        return ax

    def after_hours_trades(self):
        """
        Visualize the effect of after hours trading on this asset.
        Returns:
            A matplotlib Axes object.
        """
        num_categories = self.data[self.group_by].nunique()
        fig, axes = plt.subplots(
            num_categories,
            2,
            figsize=(15, 8*num_categories)
        )

        for ax, (name, data) in zip(axes, self.data.groupby(self.group_by)):
            after_hours = (data.open - data.close.shift())

            monthly_effect = after_hours.resample('1M').sum()

            after_hours.plot(
                ax=ax[0],
                title=f'{name} Open Price - Prior Day\'s Close'
            )
            ax[0].set_ylabel('price')

            monthly_effect.index = monthly_effect.index.strftime('%b')
            monthly_effect.plot(
                ax=ax[1],
                kind='bar',
                title=f'{name} after hours trading monthly effect',
                color=np.where(monthly_effect >= 0, 'g', 'r'),
                rot=90
            ).axhline(0, color='black', linewidth=1)
            ax[1].set_ylabel('price')
        return axes

    def pairplot(self, **kwargs):
        """
        Generate a seaborn pairplot for this asset group.
        Parameters:
            - kwargs: Keyword arguments to pass down to `sns.pairplot()`
        Returns:
            A seaborn pairplot
        """
        return sns.pairplot(self.data.pivot_table(values='close', index=self.data.index, columns='name'), diag_kind='kde', **kwargs        )

    def heatmap(self, pct_change=False, **kwargs):
        """
        Generate a seaborn heatmap for correlations between assets.
        Parameters:
            - pct_change: Whether or not to show the correlations of the
                          daily percent change in price or just use
                          the closing price.
            - kwargs: Keyword arguments to pass down to `sns.heatmap()`
        Returns:
            A seaborn heatmap
        """
        pivot = self.data.pivot_table(
            values='close', index=self.data.index, columns='name'
        )
        if pct_change:
            pivot = pivot.pct_change()
        return sns.heatmap(pivot.corr(), annot=True, center=0, **kwargs)
    
    def f(self, x):
        ''' Efficient frontier function (splines approximation). '''
        return sci.splev(x, self.tck, der=0)
    def df(self, x):
        ''' First derivative of efficient frontier function. '''
        return sci.splev(x, self.tck, der=1)
    
    def equations(self, p, rf=0.01):
        eq1 = rf - p[0]  
        eq2 = rf + p[1] * p[2] - self.f(p[2])  
        eq3 = p[1] - self.df(p[2])  
        return eq1, eq2, eq3
    
    def portfolioSummary(self, prets, pvols, evols, erets):
        plt.figure(figsize=(10, 6))
        
       
        plt.scatter(pvols, prets, c=(prets - 0.01) / pvols, marker='.', cmap='coolwarm')
        
        #Efficient Frontier
        plt.plot(evols, erets, 'b', lw=4.0)
                
        # Capital Market Line
        
        self.tck = sci.splrep(evols, erets)
        opt = sco.fsolve(self.equations, [0.01, 0.5, 0.15])
        cx = np.linspace(0.0, 0.35)
        plt.plot(cx, opt[0] + opt[1] * cx, 'r', lw=1.5)
        plt.plot(opt[2], self.f(opt[2]), 'y*', markersize=15.0) 
        
        # Background of the graph
        plt.grid(True)
        plt.axhline(0, color='k', ls='--', lw=2.0)
        plt.axvline(0, color='k', ls='--', lw=2.0)
        plt.xlabel('expected volatility')
        plt.ylabel('expected return')
        plt.colorbar(label='Sharpe ratio')

    def show(self):
        plt.show()
                      
            
            
