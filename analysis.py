"""MC1-P1: Analyze a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality

def author(self):
        return 'lyang338'

def calculate_portfolio_value(prices, allocs, sv):
    normalized_prices = prices/prices.ix[0, :] # normalize all the stock prices first
    alloced = normalized_prices * allocs
    pos_vals = alloced * sv
    port_val = pos_vals.sum(axis=1)
    return port_val

def calculate_portfolio_statistics(port_val, rfr, sf):
    cr = (port_val[-1] / port_val[0]) - 1 # Cumulative return
    daily_returns = (port_val / port_val.shift(1)) - 1
    daily_returns[0] = 0 # in the lecture use daily_returns.ix[0, :] = 0, since here it is only one total return
    daily_returns = daily_returns[1:]
    adr = daily_returns.mean() # Average period return 
    sddr = daily_returns.std() # Standard deviation of daily return
    sr = (daily_returns - rfr).mean() / sddr * np.sqrt(sf) # Sharpe ratio
    return cr, adr, sddr, sr

def plot_normalized_prices(df):
    nomalized_df = df / df.ix[0, :]
    plot_data(nomalized_df, title="Normalized prices", xlabel="Date", ylabel="Normalized price")
    
def calculate_end_portfolio_value(port_val, sv):
    return (port_val[-1] / port_val[0]) * sv
    

def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get daily portfolio value
    port_val = calculate_portfolio_value(prices, allocs, sv)

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr, adr, sddr, sr = calculate_portfolio_statistics(port_val, rfr, sf)
    

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        plot_normalized_prices(df_temp)

    # Add code here to properly compute end value
    ev = calculate_end_portfolio_value(port_val, sv)

    return cr, adr, sddr, sr, ev

def test_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2009,1,1)
    end_date = dt.datetime(2010,1,1)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000  
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, \
        gen_plot = False)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    test_code()
