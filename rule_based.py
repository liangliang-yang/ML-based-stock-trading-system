import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data, plot_data
from analysis import calculate_portfolio_value, calculate_portfolio_statistics, plot_normalized_prices

import numpy as np
%pylab
%matplotlib inline

def author(self):
        return 'lyang338'



def symbol_to_path(symbol, base_dir=os.path.join("..", "data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

#  simple moving average
def SMA(df, periods=12):
    return pd.rolling_mean(df, window=periods)

def STO_KD(df, periods=14):
    # %K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
    # %D = 3-day SMA of %K

    current = df['Close'] # Current Close
    low = df['Low'] # Lowest Low
    high = df['High']   # Highest High
    length = len(df)-periods + 1

    STO_K = pd.DataFrame(index=df.index, columns=['%K'])
    STO_K[0:] = np.float('nan')

    for i in range(0, length):
        Highest_High = np.max(high[i: i+periods])
        Lowest_Low = np.min(low[i: i+periods])
        position = i + periods - 1
        current_close = current.ix[position]
        STO_K.ix[position] = (current_close - Lowest_Low) /(Highest_High - Lowest_Low) *100

    STO_D = SMA(STO_K, periods=3)
    STO_D.rename(columns={'%K':'%D'},inplace=True)
    result = STO_K.join(STO_D)
    return result


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


# better for low <0.9, not for high
def test_Basket_strategy_run():
    # Read data
    long_dates = pd.date_range('2007-11-1', '2009-12-31')
    dates = pd.date_range('2008-1-1', '2009-12-31')
    start_date = '2008-1-1'
    end_date = '2009-12-31'
    symbols = ['AAPL']
    df = get_data(symbols, long_dates)
    df_AAPL = df
    mask_AAPL = (df_AAPL.index >= start_date) & (df_AAPL.index <= end_date)
    df_AAPL = df_AAPL.loc[mask_AAPL]
    print df_AAPL.head()

    df_STO = pd.DataFrame(index=long_dates)
    for symbol in symbols:
        df_STO_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                    parse_dates=True, na_values=['nan'])

        df_STO = df_STO.join(df_STO_temp)
    df_STO = df_STO.dropna()



    # compute SMA
    sma_window = 12
    sma = SMA(df[symbols], sma_window)
    price_sma = df[symbols]/sma
    # generate orders_file and signal for price_sma
    price_sma.columns = ['Price/SMA']
    price_sma_mask = (price_sma.index >= start_date) & (price_sma.index <= end_date)
    price_sma = price_sma.loc[price_sma_mask]


    # compute STO_KD
    sto_kd = STO_KD(df_STO, periods=14)
    df_STO = pd.concat([df_STO, sto_kd], axis=1)
    mask_sto_kd = (sto_kd.index >= start_date) & (sto_kd.index <= end_date)
    sto_kd = sto_kd.loc[mask_sto_kd]

    # compute BB and %B
    # 1. Compute rolling mean
    rm = get_rolling_mean(df[symbols], window=20)

    # 2. Compute rolling standard deviation
    rstd = get_rolling_std(df[symbols], window=20)

    # 3. Compute upper and lower bands
    upper_band, lower_band = get_bollinger_bands(rm, rstd)

    # get B
    bollinger_B = get_bollinger_bands_B_indicator(df, rm, rstd)
    bollinger_B = bollinger_B[symbols]
    mask_BB = (bollinger_B.index >= start_date) & (bollinger_B.index <= end_date)
    bollinger_B = bollinger_B.loc[mask_BB]
    bollinger_B.columns = ['bollinger_B']


    ########################## need a basket here, store all price_sma, sto_kd, B, Order and Signal, used for add vertical line
    df_basket = pd.concat([price_sma, bollinger_B, sto_kd], axis=1)
    # add 'Order' and 'Signal column to basket df
    df_basket['Order'] = np.repeat('NoAction',df_basket.shape[0])
    df_basket['Signal'] = np.repeat('NoAction',df_basket.shape[0])
    df_basket['rule_label'] = np.repeat(0,df_basket.shape[0])

    orders = []
    share_hold = 0
    sv = 100000
    print df['AAPL'].ix[df.shape[0]-1]
    print price_sma.index[0].date()


    day=0
    print df_basket['Price/SMA'].ix[day].item()
    while(day < df_basket.shape[0]-1):
        # close the position first
        if(share_hold != 0):
            cash += df['AAPL'].ix[day]*share_hold
            if(share_hold > 0):
                # need to sell first
                orders.append([df_basket.index[day].date(), 'AAPL', 'SELL', 200])
            elif(share_hold < 0):
                # need to buy first
                orders.append([df_basket.index[day].date(), 'AAPL', 'BUY', 200])
            share_hold = 0

        #Overboght
        if(((df_basket['Price/SMA'].ix[day] > 1.05) & (df_basket['bollinger_B'].ix[day] > 1)) |
           ((df_basket['%D'].ix[day] > 80) & (df_basket['bollinger_B'].ix[day] > 1)) |
           ((df_basket['Price/SMA'].ix[day] > 1.05) & (df_basket['%D'].ix[day] > 80))):
            orders.append([df_basket.index[day].date(), 'AAPL', 'SELL', 200])
            df_basket.loc[df_basket.index[day],'Order']='SELL'
            df_basket.loc[df_basket.index[day], 'Signal']='SHORT'
            df_basket.loc[df_basket.index[day], 'rule_label']=-1
            share_hold -= 200
            # hold for 21 days
            if (day < df_basket.shape[0]-21):
                day += 21
            else:
                day = df_basket.shape[0]-1

        elif(((df_basket['Price/SMA'].ix[day] < 0.95) & (df_basket['bollinger_B'].ix[day] < 0)) |
           ((df_basket['%D'].ix[day] < 20) & (df_basket['bollinger_B'].ix[day] < 0)) |
           ((df_basket['Price/SMA'].ix[day] < 0.95) & (df_basket['%D'].ix[day] < 20))):
            orders.append([df_basket.index[day].date(), 'AAPL', 'BUY', 200])
            df_basket.loc[df_basket.index[day],'Order']='BUY'
            df_basket.loc[df_basket.index[day], 'Signal']='LONG'
            df_basket.loc[df_basket.index[day], 'rule_label']=1
            share_hold += 200
            # hold for 21 days
            if (day < df_basket.shape[0]-21):
                day += 21
            else:
                day = df_basket.shape[0]-1

        else:
            day += 1

    # last day close the position
    if(share_hold != 0):
            cash += df['AAPL'].ix[day]*share_hold
            if(share_hold > 0):
                # need to sell first
                orders.append([df_basket.index[day].date(), 'AAPL', 'SELL', 200])
            elif(share_hold < 0):
                # need to buy first
                orders.append([df_basket.index[day].date(), 'AAPL', 'BUY', 200])
            share_hold = 0


    df_basket.to_csv('./df_basket_with_label.csv')
    with open("./basket_orders.csv",'wb') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        wr.writerow(['Date','Symbol', 'Order', 'Shares'])
        wr.writerows(orders)


    of = "./basket_orders.csv"
    df_orders = pd.read_csv(of)
    start_date = df_orders['Date'].min()
    end_date = df_orders['Date'].max()

    benchmark = "./benchmark_order.csv"
    benchmark_orders = pd.read_csv(benchmark)
    start_date_benchmark = benchmark_orders['Date'].min()
    end_date_benchmark = benchmark_orders['Date'].max()

    long_days=df_basket[df_basket['Signal']=='LONG'].index.values

    short_days=df_basket[df_basket['Signal']=='SHORT'].index.values



    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    portvals.columns = ['strategy']
    nomalized_portvals = portvals / portvals.ix[0, :]
    nomalized_portvals.to_csv('./manual_method.csv')
    #print portvals

    benchmarkvals = compute_portvals(orders_file = benchmark, start_val = sv)
    benchmarkvals.columns = ['benchmark']
    nomalized_benchmarkvals = benchmarkvals / benchmarkvals.ix[0, :]
    print benchmarkvals.head()

    nomalized_AAPL = df_AAPL / df_AAPL.ix[0, :]
    print nomalized_AAPL.head()

    #df_AAPL = df['AAPL']
    print df['AAPL'].head()
    #print sv
    #print portvals
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    #cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = calculate_portfolio_statistics(portvals, 0.0, 252)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = calculate_portfolio_statistics(portvals, 0.0, 252)
    print 'Best cum_ret, avg_daily_ret, std_daily_ret are', cum_ret, avg_daily_ret, std_daily_ret

    # Get portfolio stats for SPY
    prices_SPY = get_data(['SPY'], pd.date_range(start_date, end_date))
    #print prices_SPY
    prices_SPY = prices_SPY[['SPY']]
    portvals_SPY = calculate_portfolio_value(prices_SPY, [1.0], sv)
    #cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = calculate_portfolio_sta



    #plot_normalized_prices(df_temp)

    fig, ax1 = plt.subplots()
#     ax2 = ax1.twinx()

    ax1.plot(nomalized_benchmarkvals.index, nomalized_benchmarkvals['benchmark'], label='Benchmark', color='black')
    ax1.plot(nomalized_portvals.index, nomalized_portvals['strategy'], label='rule-based portfolio', color='blue')
    ax1.plot(nomalized_AAPL.index, nomalized_AAPL['AAPL'], label='AAPL', color='orange')

#     ax2.plot(df.index, df['Price_SMA'], label='Price/SMA', color='red', linestyle='--')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Normalized Price')
    #ax.set_xticks((10,100,1000))
#     ax1.set_ylim([50, 250])

#     ax2.set_ylabel('Price/SMA')
    ax1.set_ylim([0, 2.0])
    plot_start_date = '2007-12-25'
    plot_end_date = '2009-12-31'
    ax1.set_xlim([plot_start_date, plot_end_date])
    ax1.axhline(y = 1.0, linestyle='--', color='purple')
#     ax2.axhline(y = 1.1)
    for ld in long_days:
        plt.axvline(x=ld,color='g')
    for sd in short_days:
        plt.axvline(x=sd,color='red')

    ax1.legend(loc='upper right')
#     ax2.legend(loc='lower right')
    fig.autofmt_xdate()
    plt.title('Manual Rule-Based Trader plot')
    plt.show()



if __name__ == "__main__":
    test_Basket_strategy_run()


def compute_portvals(orders_file, start_val):
    # this is the function the autograder will call to test your code

    df_orders = pd.read_csv(orders_file)  # read csv file
    #df_orders = df_orders.sort(columns='Date')  # sort orders if it is disordered

    # extract date infor from the order file
    start_date = df_orders['Date'].min()
    end_date = df_orders['Date'].max()

    # step1: construct the prices df, no need for SPY
    stock_symbols = list(set(df_orders['Symbol']))
    df_prices = get_data(stock_symbols, pd.date_range(start_date, end_date))
    df_prices = df_prices.drop('SPY', axis=1)
    cash = start_val # add last column cash
    #print cash

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
    return df_trades, df_holdings, cash
