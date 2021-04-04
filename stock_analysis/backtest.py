# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import backtrader as bt

if __name__ == '__main__':
    
    cerebro = bt.Cerebro()
    
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    
    cerebro.run()
    
    print('Final Portfolio ValueL %.2f' % cerebro.broker.getvalue())