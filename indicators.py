import os
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data, plot_data

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


def test_SMA_run():
    # Read data
    dates = pd.date_range('2008-1-1', '2009-12-31')
    symbols = ['AAPL']
    df = get_data(symbols, dates)


    # compute SMA
    sma = SMA(df[symbols], 12)

    # compute price_sma
    price_sma = df[symbols]/sma


    df = pd.concat([df, sma, price_sma], axis=1)
    df.columns = ['SPY', symbols[0], 'SMA', 'Price_SMA']

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(df.index, df[symbols], label='Price', color='black')
    ax1.plot(df.index, df['SMA'], label='SMA', color='green')

    ax2.plot(df.index, df['Price_SMA'], label='Price/SMA', color='red', linestyle='--')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.set_ylim([50, 250])

    ax2.set_ylabel('Price/SMA')
    ax2.set_ylim([0.7, 1.2])
    ax2.axhline(y = 0.95)
    ax2.axhline(y = 1.05)

    ax1.legend(loc='lower left')
    ax2.legend(loc='lower right')
    fig.autofmt_xdate()
    plt.title('Price/SMA indicator plot')
    plt.show()



if __name__ == "__main__":
    test_SMA_run()


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

def test_STO_KD_run():
    # Read data
    dates = pd.date_range('2008-1-1', '2009-12-31')
    symbols = ['AAPL']
    df = pd.DataFrame(index=dates)
    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                    parse_dates=True, na_values=['nan'])

        df = df.join(df_temp)


    # compute SMA
    sto_kd = STO_KD(df, periods=14)
    print sto_kd.head()


    df = pd.concat([df, sto_kd], axis=1)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(df.index, df['Low'], label='Low', color='orange')
    ax1.plot(df.index, df['High'], label='High', color='green')
    ax1.plot(df.index, df['Close'], label='Close', color='black')

    ax2.plot(df.index, df['%D'], label='%D (3-day SMA of %K)', color='red', linestyle='--')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax2.set_ylabel('%D')

    ax1.set_ylim([0, 250])
    ax2.set_ylim([-30, 130])
    ax2.axhline(y = 80)
    ax2.axhline(y = 20)

    ax1.legend(loc='lower left')
    ax2.legend(loc='lower right')
    fig.autofmt_xdate()
    plt.title('Stochastic Oscillator %D indicator plot')
    plt.show()



if __name__ == "__main__":
    test_STO_KD_run()


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


def test_bollinger_B_run():
    # Read data
    start_date = '2008-1-1'
    end_date = '2009-12-31'
    long_dates = pd.date_range('2007-11-1', '2009-12-31')
    dates = pd.date_range('2008-1-1', '2009-12-31')
    symbols = ['AAPL']
    df = get_data(symbols, dates)
    df_long = get_data(symbols, long_dates)


    # Compute Bollinger Bands
    # 1. Compute rolling mean
    rm = get_rolling_mean(df_long[symbols], window=20)

    # 2. Compute rolling standard deviation
    rstd = get_rolling_std(df_long[symbols], window=20)

    # 3. Compute upper and lower bands
    upper_band, lower_band = get_bollinger_bands(rm, rstd)

    # get B
    bollinger_B = get_bollinger_bands_B_indicator(df_long, rm, rstd)
    print bollinger_B.head()
    mask_BB = (bollinger_B.index >= start_date) & (bollinger_B.index <= end_date)
    bollinger_B = bollinger_B.loc[mask_BB]

    print bollinger_B.head()

    df = pd.concat([df, rm, upper_band, lower_band, bollinger_B], axis=1)
    df.columns = ['SPY', symbols[0], 'SMA', 'UB', 'LB', 'B', 'B_SPY']
    mask_df = (df.index >= start_date) & (df.index <= end_date)
    df = df.loc[mask_df]
    print df.head()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(df.index, df[symbols], label='AAPL', color='black')
    ax1.plot(df.index, df['UB'], label='Upper Band', color='green', linewidth=2)
    ax1.plot(df.index, df['LB'], label='Lower Band', color='green', linewidth=2)
    ax1.fill_between(df.index,df['UB'], df['LB'], color = 'grey')

    ax2.plot(df.index, df['B'], label='B', color='red', linestyle='--')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.set_ylim([0, 250])


    ax2.set_ylabel('B indicator')
    ax2.set_ylim([-0.5, 1.5])
    ax2.axhline(y = 0)
    ax2.axhline(y = 1.0)

    ax1.legend(loc='lower left', prop={'size':12})
    ax2.legend(loc='lower right')
    fig.autofmt_xdate()
    plt.title('Bollinger Band %B indicator plot')
    plt.show()

if __name__ == "__main__":
    test_bollinger_B_run()
