"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
from datetime import timedelta
import QLearner as ql
import pandas as pd
import numpy as np
import util as ut
import os
import csv
from util import get_data, plot_data

def author(self):
    return 'lyang338'
        
def Force_Index(df, periods=1):
    ForceIndex = pd.Series(df['Close'].diff(periods)*df['Volume'], name = 'ForceIndex')
    return ForceIndex

def SMA(df, periods=12):
    return pd.rolling_mean(df, window=periods)

def get_rolling_mean(values, window):
    # Return rolling mean of given values, using specified window size.
    return pd.rolling_mean(values, window=window)


def get_rolling_std(values, window):
    # Return rolling standard deviation of given values, using specified window size.
    return pd.rolling_std(values, window=window)


def get_bollinger_bands(rm, rstd):
    # Return upper and lower Bollinger Bands.
    upper_band = rm + rstd*2
    lower_band = rm - rstd*2
    return upper_band, lower_band

def get_bollinger_bands_B_indicator(price, rm, rstd):
    # Return upper and lower Bollinger Bands.
    upper_band = rm + rstd*2
    lower_band = rm - rstd*2
    # %B = (Price - Lower Band)/(Upper Band - Lower Band)
    B = (price - lower_band)/(upper_band - lower_band)
    return B


class AAPL_player(object):
    def __init__(self, market_state, share_holdings = 0):
        self.market_state = market_state
        self.share_holdings = share_holdings
        
def update_shareholdings(A, action):
    # update share_holdings according to action
    if action == 0: # LONG
        A.share_holdings = 200
    elif action == 1: # SHORT
        A.share_holdings = -200
    else: # keep zero
        A.share_holdings = 0
    
    return A.share_holdings

        
    
class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False):
        self.verbose = verbose

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = 'AAPL', \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,12,31), \
        sv = 100000): 
        
        
        long_dates = pd.date_range(sd - timedelta(days=30), ed + timedelta(days=30))
        dates = pd.date_range(sd, ed)
        syms=[symbol]
        df = pd.DataFrame(index=long_dates)
        
        prices_all = ut.get_data(syms, long_dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices.columns = ['Close']
        df = df.join(prices)
        
        volume_all = ut.get_data(syms, long_dates, colname = "Volume")  # automatically adds SPY
        volume = volume_all[syms]  # only portfolio symbols
        volume.columns = ['Volume']
        df = df.join(volume)
        #print volume.head()
        
        df = df.dropna()
        #print df.head()
        
        # compute and discretize force index
        forceIndex_period = 1
        ForceIndex = Force_Index(df, forceIndex_period)
        df = pd.concat([df, ForceIndex], axis=1)
        df['forceIndex_state'] = pd.cut(df.ForceIndex, bins=10, labels=False)
        
        # compute and discretize price/sma
        sma_window = 7
        sma = SMA(df['Close'], sma_window)
        mask = (sma.index >= sd) & (sma.index <= ed)
        sma = sma.loc[mask]
        # compute price_sma
        df['priceSMA_state'] = df['Close']/sma
        df['priceSMA_state'] = pd.cut(df['priceSMA_state'], bins=10, labels=False)
        
        # compute and discretize BB
        # Compute Bollinger Bands
        # 1. Compute rolling mean
        rm = get_rolling_mean(df['Close'], window=20)

        # 2. Compute rolling standard deviation
        rstd = get_rolling_std(df['Close'], window=20)

        # 3. Compute upper and lower bands
        upper_band, lower_band = get_bollinger_bands(rm, rstd)

        # get B
        bollinger_B = get_bollinger_bands_B_indicator(df['Close'], rm, rstd)
        #print bollinger_B.head()
        
        #df = pd.concat([df, bollinger_B], axis=1)
        df['bollinger_B_state'] = pd.cut(bollinger_B, bins=10, labels=False)
        
        mask = (df.index >= sd) & (df.index <= ed)
        df = df.loc[mask]
        
        
        # convert dataframe to array to improve speed
        df_forceIndex_state_array = df.loc[:,('forceIndex_state')].as_matrix()
        df_priceSMA_state_array = df.loc[:,('priceSMA_state')].as_matrix()
        df_bollinger_B_state_array = df.loc[:,('bollinger_B_state')].as_matrix()
        df_Close_array = df.loc[:,('Close')].as_matrix()
        
        #df_action_array = df.loc[:,('action')].as_matrix()
        #df_trades_array = df.loc[:,('trades')].as_matrix()
        
        #state0 = int(df_forceIndex_state_array[0])
        #state0 = int(df_forceIndex_state_array[0] + df_bollinger_B_state_array[0]*10)
        state0 = int(df_forceIndex_state_array[0] + df_priceSMA_state_array[0]*10 + df_bollinger_B_state_array[0]*100)
        self.qlearner = ql.QLearner(num_states=1000, num_actions = 3, alpha = 0.2, rar = 1.0, radr = 0.9999)
        
        for i in range(1000):
            
            # init player
            A = AAPL_player(state0, 0)
            action = self.qlearner.querysetstate(state0)
            
            
            #print action
            reward = 0
            share_holdings = 0
            day = 1
            #for day in range(df.shape[0]-1):
            while (day < df.shape[0]-1):
                
                # not long or short in last day
                # calculate reward
                if A.share_holdings != 0:
                    diff_price = df_Close_array[day] - df_Close_array[day-1]
                    reward = diff_price * A.share_holdings
                
                #state = int(df_forceIndex_state_array[day])
                #state = int(df_forceIndex_state_array[day] + df_bollinger_B_state_array[day]*10)
                state = int(df_forceIndex_state_array[day] + df_priceSMA_state_array[day]*10 + df_bollinger_B_state_array[day]*100)
                
                action = self.qlearner.query(state, reward)
                #print state, reward, action
                #df['action'].ix[day] = action # df_action
                #df_action_array[day] = action  
                #pre_share_holdings = share_holdings
                share_holdings = update_shareholdings(A, action)
                #trade = share_holdings - pre_share_holdings
                #df['trades'].ix[day] = trade # df_trades
                #df_trades_array[day] = trade
                day += 1
                
            #Q = self.qlearner.returnQ()   
            #print 'Q sum is', np.sum(Q), i

        #Q = self.qlearner.returnQ()   
        #print 'train Q is:', Q       
        #print df.head()

                    


    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "AAPL", \
        sd=dt.datetime(2010,1,1), \
        ed=dt.datetime(2011,12,31), \
        sv = 100000):

        
        long_dates = pd.date_range(sd - timedelta(days=30), ed + timedelta(days=30))
        dates = pd.date_range(sd, ed)
        syms=[symbol]
        df = pd.DataFrame(index=long_dates)
        trades = pd.DataFrame(index=dates)
        
        prices_all = ut.get_data(syms, long_dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices.columns = ['Close']
        df = df.join(prices)
        
        volume_all = ut.get_data(syms, long_dates, colname = "Volume")  # automatically adds SPY
        volume = volume_all[syms]  # only portfolio symbols
        volume.columns = ['Volume']
        df = df.join(volume)
        
        df = df.dropna()
        
        
        
        # compute and discretize force index
        forceIndex_period = 1
        ForceIndex = Force_Index(df, forceIndex_period)
        
        df = pd.concat([df, ForceIndex], axis=1)
        df['forceIndex_state'] = pd.cut(df.ForceIndex, bins=10, labels=False)
        
        # compute and discretize price/SMA
        sma_window = 7
        sma = SMA(df['Close'], sma_window)
        mask = (sma.index >= sd) & (sma.index <= ed)
        sma = sma.loc[mask]
        df['priceSMA_state'] = df['Close']/sma
        df['priceSMA_state'] = pd.cut(df['priceSMA_state'], bins=10, labels=False)
        
        # compute and discretize BB
        # Compute Bollinger Bands
        # 1. Compute rolling mean
        rm = get_rolling_mean(df['Close'], window=20)

        # 2. Compute rolling standard deviation
        rstd = get_rolling_std(df['Close'], window=20)

        # 3. Compute upper and lower bands
        upper_band, lower_band = get_bollinger_bands(rm, rstd)

        # get B
        bollinger_B = get_bollinger_bands_B_indicator(df['Close'], rm, rstd)
        #print bollinger_B.head()
        
        #df = pd.concat([df, bollinger_B], axis=1)
        df['bollinger_B_state'] = pd.cut(bollinger_B, bins=10, labels=False)
        
        mask = (df.index >= sd) & (df.index <= ed)
        df = df.loc[mask]


        df['trades'] = np.repeat(0, df.shape[0])
        
        # convert dataframe to array to improve speed
        df_forceIndex_state_array = df.loc[:,('forceIndex_state')].as_matrix()
        df_priceSMA_state_array = df.loc[:,('priceSMA_state')].as_matrix()
        df_bollinger_B_state_array = df.loc[:,('bollinger_B_state')].as_matrix()
        df_Close_array = df.loc[:,('Close')].as_matrix()
        df_trade_array = df.loc[:,('trades')].as_matrix()
        #state0 = int(df_forceIndex_state_array[0])
        #state0 = int(df_forceIndex_state_array[0] + df_bollinger_B_state_array[0]*10)
        state0 = int(df_forceIndex_state_array[0] + df_priceSMA_state_array[0]*10 + df_bollinger_B_state_array[0]*100)
        
        #rar = self.qlearner.reset_rar()
        #print rar
        
        #print df
        # init player
        B = AAPL_player(state0, 0)
        share_holdings = 0
        for day in range(df.shape[0]-1):
            #state = int(df_forceIndex_state_array[day])
            #state = int(df_forceIndex_state_array[day] + df_bollinger_B_state_array[day]*10)
            state = int(df_forceIndex_state_array[day] + df_priceSMA_state_array[day]*10 + df_bollinger_B_state_array[day]*100)
            
            action = self.qlearner.querysetstate(state)
            #df['action'].ix[day] = action # df_action
            pre_share_holdings = share_holdings
            share_holdings = update_shareholdings(B, action)
            trade = share_holdings - pre_share_holdings
            #df['trades'].ix[day] = trade # df_trades
            df_trade_array[day] = trade
        
        trades = pd.DataFrame(data=df_trade_array, index=df.index)
        #print trades
#         trades = df['trades']
#         trades.columns = ['AAPL']

        #print 'trades end'
        #print trades.head()
        return trades

if __name__=="__main__":
    print "One does not simply think up a strategy"
    learner = StrategyLearner(verbose = False)
    learner.addEvidence()
    learner.testPolicy()
