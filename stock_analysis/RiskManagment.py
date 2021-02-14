import numpy as np
import pandas as pd
import backtestingrm as btr

class TBBacktesterRM(brt.BacktestingBaseRM):

    def _reshape(self, state):
        return np.reshape(state, [1, self.env.lags, self.env.n_features])

    def backtest_strategy(self, stopLoss=None, trailingStopLoss=None, targetPrice=None, wait=5, guarantee=False):

        self.shares = 0
        self.position = 0
        self.trades = 0
        self.stopLoss = stopLoss
        self.trailingStopLoss = trailingStopLoss
        self.targetPrice = targetPrice
        self.wait = 0
        self.current_balance = self.initial_amount
        self.net_wealths = list()

        for bar in range(self.env.lags, len(self.env.data)):
            
            self.wait = max(0, self.wait - 1)
            date, price = self.get_data_price(bar)

            if self.trades == 0:
                print(50 * '=')
                print(f'{date} | *** START BACKTEST ***')
                self.print_balance(bar)
                print(50 * '=')

            #Stop Loss order
            if stopLoss is not None and self.position != 0:
                
                priceRelationToPurchasePrice = (price - self.entry_price) / self.entry_price

                if self.position == 1 and priceRelationToPurchasePrice < -self.stopLoss:
                    
                    print(50 * '-')
                    
                    if guarantee:
                        price = self.entry_price * (1 - self.stopLoss)
                        print(f'*** STOP LOSS (LONG | {-self.stopLoss:.4f}) ***')
                    else:
                        print(f'*** STOP LOSS(LONG | {priceRelationToPurchasePrice:.4f}) ***')
                    
                    self.place_sell_order(bar, shares=self.shares, gprice=price)
                    self.wait = wait
                    self.position = 0

                elif self.position == -1 and priceRelationToPurchasePrice > self.stopLoss:

                    print(50 * '-') 

                    if guarantee:
                        price = self.entry_price * (1 + self.stopLoss)
                        print(f'*** STOP LOSS (SHORT | -{self.stopLoss:.4f} ***')
                    else:
                        print(f'*** STOP LOSS (SHORT | -{priceRelationToPurchasePrice:.4f}) ***')

                    self.place_buy_order(bar, shares=-self.shares, gprice=price)
                    self.wait = wait
                    self.position = 0

            # Trailing stop loss order
            if trailingStopLoss is not None and self.position != 0:
                
                self.max_price = max(self.max_price, price)
                self.min_price = min(self.min_price, price)

                longRelationToMaxHigh = (price - self.max_price) / self.entry_price
                shortRelationToMinLow = (self.min_price - price) / self.entry_price

                if self.position == 1 and longRelationToMaxHigh < -self.trailingStopLoss:
                    print(50 * '-')
                    print(f'*** TRAILING stopLoss (LONG | {longRelationToMaxHigh:.4f}) ***')
                    self.place_sell_order(bar, shares=self.shares)
                    self.wait = wait
                    self.position = 0
                elif self.position == -1 and shortRelationToMinLow < -self.trailingStopLoss:
                    print(50 * '-')
                    print(f'*** TRAILING stopLoss (SHORT | {shortRelationToMinLow:.4f}) ***')
                    self.place_buy_order(bar, shares=-self.shares)
                    self.wait = wait
                    self.position = 0

            # Take Profit Order
            if targetPrice is not None and self.position != 0:
                
                priceRelationToPurchasePrice = (price - self.entry_price) / self.entry_price
                
                if self.position == 1 and priceRelationToPurchasePrice > self.targetPrice:
                
                    print(50 * '_')

                    if guarantee:
                        price = self.entry_price * (1 + self.targetPrice)
                        print(f'*** TAKE PROFIT (LONG | {self.targetPrice: .4f}) ***')
                    else:
                        print(f'*** TAKE PROFIT (LONG | {priceRelationToPurchasePrice: .4f}) ***')
                
                    self.place_sell_order(bar, shares=self.shares, gprice=price)
                    self.wait = wait
                    self.position = 0

                elif self.position == -1 and priceRelationToPurchasePrice < -self.targetPrice: 
                    
                    print(50 * '-')

                    if guarantee:
                        price = self.entry_price * (1 - self.targetPrice)
                        print(f'*** TAKE PROFIT (SHORT | {self.targetPrice: .4f}) ***')
                    else:
                        print(f'*** TAKE PROFIT (SHORT | {-priceRelationToPurchasePrice: .4f}) ***')
                    
                    self.place_buy_order(bar, shares=-self.shares, gprice=price)
                    self.wait = wait
                    self.position = 0

            state = self.env.get_state(bar)
            action = np.argmax(self.model.predict(self._reshape(state.values)) [0,0])

            position = 1 if action == 1 else -1

            if self.position in [0, -1] and position == 1 and self.wait == 0:
                
                if self.position == -1:
                    self.place_buy_order(bar - 1, shares=-self.shares)

                self.place_buy_order(bar - 1, amount=self.current_balance)
                self.postion = 1 

                if self.verbose:
                    print(50 * '_')
                    print(f'{date} | *** GOING LONG ***')
                    self.print_net_wealth(bar)

            elif self.position in [0, 1] and position == -1 and self.wait == 0:

                if self.position == 1:
                    self.place_sell_order(bar -1, shares = self.shares)

                self.place_sell_order(bar -1, amount=self.current_balance)
                self.postion = -1

                if self.verbose:
                    print(50 * '_')
                    print(f'{date} | *** GOING SHORT ***')
                    self.print_net_wealth(bar)

            self.net_wealths.append((date, self.calculate_net_wealth(price)))
            self.net_wealths = pd.DataFrame(self.net_wealths,columns=['date','net_wealth'])
            self.net_wealths.set_index('date',inplace=True)
            self.net_wealths.index = pd.DatetimeIndex(self.net_wealths.index)
            self.close_out(bar)




                    





            



            





                




