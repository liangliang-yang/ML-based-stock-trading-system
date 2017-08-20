"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def author():
    return 'lyang338'

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000):
    # this is the function the autograder will call to test your code

    df_orders = pd.read_csv(orders_file)  # read csv file
    df_orders = df_orders.sort(columns='Date')  # sort orders if it is disordered

    # extract date infor from the order file
    start_date = df_orders['Date'].min()
    end_date = df_orders['Date'].max()

    # step1: construct the prices df, no need for SPY
    stock_symbols = list(set(df_orders['Symbol']))
    df_prices = get_data(stock_symbols, pd.date_range(start_date, end_date))
    df_prices = df_prices.drop('SPY', axis=1)
    cash = start_val # add last column cash

    # step 2: construct/init the trades df, initialize fill with zero
    index_trades = pd.date_range(start_date, end_date)
    df_trades = pd.DataFrame(index=index_trades, columns=[stock_symbols])
    df_trades = df_trades.fillna(0)

    #step 3: contsruct the holdings df
    df_holdings = np.cumsum(df_trades)

    #step 4: run the order in order book
    for orderNumber in range (0, len(df_orders)):
        order = df_orders.ix[orderNumber, ]
        df_trades, df_holdings, cash = run_orderBook(order, df_prices, df_trades, df_holdings, cash)


    # step 5: construct the Values df for stock after update of
    df_stockValues = df_prices * df_holdings
    df_stockValues = df_stockValues.dropna() # drop na value
    df_stockValues['Value'] = np.sum(df_stockValues, axis=1)

    # step 6: update the df_trades
    df_trades['tradeValue'] = np.sum(df_prices * df_trades, axis=1)
    df_trades = df_trades.dropna(subset=['tradeValue'])
    df_trades['Cash'] = 0
    df_trades.ix[0, 'Cash'] = start_val - df_trades.ix[0, 'tradeValue']
    for row in range(1, len(df_trades)):
        df_trades.ix[row, 'Cash'] = df_trades.ix[row-1, 'Cash'] - df_trades.ix[row, 'tradeValue']

    df_trades['stockValue'] = df_stockValues['Value']
    # total value = cash + stockValue
    df_trades['Portvals'] = df_trades['Cash'] + df_trades['stockValue']

    portvals = df_trades[['Portvals']]
    return portvals

def  run_orderBook(order, df_prices, df_trades, df_holdings, cash):

    df_trades_old = df_trades.copy()
    df_holdings_old = df_holdings.copy()
    cash_old = cash

    date = order['Date']
    symbol = order['Symbol']

    #update df_trades acoordering to the order book, 'BUY' or 'SELL'
    if order['Order'] == 'BUY':
        df_trades.ix[date, symbol] = df_trades.ix[date, symbol] + order['Shares']
        trade_value = np.sum(df_prices.ix[date, symbol] * order['Shares'])
    if order['Order'] == 'SELL':
        df_trades.ix[date, symbol] = df_trades.ix[date, symbol] + order['Shares'] * (-1)
        trade_value = np.sum(df_prices.ix[date, symbol] * order['Shares'] * (-1))

    # update the stock holdings after trading
    df_holdings = np.cumsum(df_trades)

    # update the cash still owned
    cash = cash - trade_value

    # caluculate the leverage
    stockValues = df_prices.ix[date, ] * df_holdings.ix[date, ]
    sum_long = np.sum(stockValues[stockValues > 0])
    sum_short = np.sum(stockValues[stockValues < 0])
    leverage = (sum_long + (-1)*sum_short) / ((sum_long - (-1)*sum_short) + cash)



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


def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders2.csv"
    sv = 1000000

    df_orders = pd.read_csv(of)  # read csv file
    df_orders = df_orders.sort(columns='Date')  # sort orders if it is disordered
    #df_orders = df_orders.reset_index()


    start_date = df_orders['Date'].min()
    end_date = df_orders['Date'].max()

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = calculate_portfolio_statistics(portvals, 0.0, 252)

    # Get portfolio stats for SPY
    prices_SPY = get_data(['$SPX'], pd.date_range(start_date, end_date))
    prices_SPY = prices_SPY[['$SPX']]
    portvals_SPY = calculate_portfolio_value(prices_SPY, [1.0], sv)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = calculate_portfolio_statistics(portvals_SPY, 0.0, 252)


    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()
