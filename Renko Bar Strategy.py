


"""
Buy signal:
-Rekno Bar greater than or equal to 2
-5day OBV slope greater than 30 degrees
-Exit if bar less than 2

Sell Signal:
- Renko bar less than equal to -2
- 5 Day OBV slope less than -30 degrees
- Exit if renko bar is greater than -2

"""
#Renko Bar Strategy




#imports
import numpy as np
import pandas as pd
from stocktrends import Renko
import statsmodels.api as sm
import copy
import matplotlib.pyplot as plt
import yfinance as yf


#get ticker data
tickers = ['AAPL', 'MSFT','GOOG', 'INTC', 'F']
ohlcv_data = {}

# looping over tickers and storing OHLCV dataframe in dictionary
for ticker in tickers:
    temp = yf.download(ticker,period='60d',interval='5m')
    temp.dropna(how="any",inplace=True)
    ohlcv_data[ticker] = temp


#functions for calculating key stats
def slope(ser,n):
    "function to calculate the slope of n consecutive points on a plot"
    slopes = [i*0 for i in range(n-1)] #list of zeros that goes from 0 to n-1, which is the first slope point
    for i in range(n,len(ser)+1): #from the first slope point to last 
        y = ser[i-n:i] #the y value is the dataframe valye of row we are on minus the number of con points, to this row's value, which gives us n data points to make a slope line with
        x = np.array(range(n)) #the x values are just an array from 0-n
        y_scaled = (y - y.min())/(y.max() - y.min()) #scale the y values
        x_scaled = (x - x.min())/(x.max() - x.min()) #scale the x values
        x_scaled = sm.add_constant(x_scaled) #adds a columns of 1s to the array
        model = sm.OLS(y_scaled,x_scaled) #create the model with oridanry least squares
        results = model.fit() #fit the model
        slopes.append(results.params[-1]) #add the slopes to the slope list
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes)))) #find the slope angle using trig
    return np.array(slope_angle)


def renko_DF(DF):
    "function to convert ohlc data into renko bricks"
    df = DF.copy() #copy the dataframe
    df.reset_index(inplace=True)
    df = df.iloc[:,[0,1,2,3,4,5]] 
    df.columns = ["date","open","high","low","close","volume"] #create new column names
    df2 = Renko(df) #create the instance
    df2.brick_size = max(0.5,round(ATR(DF,120)["ATR"][-1],0)) #the brick size is either 0.5, or the last ATR
    renko_df = df2.get_ohlc_data() #get the bricks
    renko_df["bar_num"] = np.where(renko_df["uptrend"]==True,1,np.where(renko_df["uptrend"]==False,-1,0))# if it is uptrend it is 1, if it is downtrend it is -1
    for i in range(1,len(renko_df["bar_num"])):
        if renko_df["bar_num"][i]>0 and renko_df["bar_num"][i-1]>0: #if the bar number is positive, and was positive last time
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1] #add the bar numbers, essentially stacking 1s together
        elif renko_df["bar_num"][i]<0 and renko_df["bar_num"][i-1]<0: #if the bar number is negative, and the last one was too
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1] #add the negative bar numbers together
    renko_df.drop_duplicates(subset="date",keep="last",inplace=True) #drop duplicate dates
    return renko_df #return the renko dataframe
def ATR(DF, n=14):
    df = DF.copy() #make new datagframe
    df['H-L'] = df['High']-df["Low"] #
    df['H-PC'] = abs(df['High'] - df['Adj Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Adj Close'].shift(1))
    df['TR'] = df[['H-L',"H-PC","L-PC"]].max(axis = 1,skipna= False)
    df['ATR'] = df['TR'].ewm(com = n, min_periods=n).mean()
    return df
def OBV(DF):
    """function to calculate On Balance Volume, which is directional volume accumulated"""
    
    df = DF.copy()
    df['daily_ret'] = df['Adj Close'].pct_change()
    df['direction'] = np.where(df['daily_ret']>=0,1,-1)
    df['direction'][0] = 0
    df['vol_adj'] = df['Volume'] * df['direction']
    df['obv'] = df['vol_adj'].cumsum()
    return df['obv']

def CAGR(DF):
    "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
    df = DF.copy()
    df["cum_return"] = (1 + df["return"]).cumprod()
    n = len(df)/(60*78)
    CAGR = (df["cum_return"].tolist()[-1])**(1/n) - 1
    return CAGR

def volatility(DF):
    "function to calculate annualized volatility of a trading strategy"
    df = DF.copy()
    vol = df['return'].std() * np.sqrt(60*78)
    return vol

def sharpe(DF,rf):
    "function to calculate sharpe ratio ; rf is the risk free rate"
    df = DF.copy()
    sr = (CAGR(df) - rf)/volatility(df)
    return sr
    

def max_dd(DF):
    "function to calculate max drawdown"
    df = DF.copy()
    df["cum_return"] = (1 + df['return']).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"]/df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()



#backtesting



ohlc_renko = {}
df = copy.deepcopy(ohlcv_data)


tickers_signal = {}
tickers_return = {}



for ticker in tickers:
    print("merging for {}".format(ticker)) #merge the ohlc dataframe with the renko df
    renko = renko_DF(df[ticker]) #copy the df
    df[ticker]['date'] = df[ticker].index #make a new date column from the index
    ohlc_renko[ticker] = df[ticker].merge(renko.loc[:,['date','bar_num']], how = 'outer', on = 'date') #merge the two dataframes, on the date axis
    ohlc_renko[ticker]["bar_num"].fillna(method='ffill',inplace=True) #fill an nan values using forward fill
    ohlc_renko[ticker]['obv'] = OBV(ohlc_renko[ticker]) #make an obv column
    ohlc_renko[ticker]['obv slope'] = slope(ohlc_renko[ticker]['obv'],5) #make a slope column
    tickers_signal[ticker] ="" #initililzie the two rows
    tickers_return[ticker] = []



for ticker in tickers:
    '''
    check for the signal
    if the signal is empty, assign a signal based on the angle of the slope and the renko bar number
    if it is a buy, calculate returns for the candle, and check whether it isi still a buy or blank signal
    same thing with sell
    
    '''
    print("calculating daily returns for ",ticker)
    for i in range(len(ohlc_renko[ticker])):
        if tickers_signal[ticker] == "":
            tickers_return[ticker].append(0)
            if ohlc_renko[ticker]["bar_num"][i]>=2 and ohlc_renko[ticker]["obv slope"][i]>30:
                tickers_signal[ticker] = "Buy"
            elif ohlc_renko[ticker]["bar_num"][i]<=-2 and ohlc_renko[ticker]["obv slope"][i]<-30:
                tickers_signal[ticker] = "Sell"
        
        elif tickers_signal[ticker] == "Buy":
            tickers_return[ticker].append((ohlc_renko[ticker]["Adj Close"][i]/ohlc_renko[ticker]["Adj Close"][i-1])-1)
            if ohlc_renko[ticker]["bar_num"][i]<=-2 and ohlc_renko[ticker]["obv slope"][i]<-30:
                tickers_signal[ticker] = "Sell"
            elif ohlc_renko[ticker]["bar_num"][i]<2:
                tickers_signal[ticker] = ""
                
        elif tickers_signal[ticker] == "Sell":
            tickers_return[ticker].append((ohlc_renko[ticker]["Adj Close"][i-1]/ohlc_renko[ticker]["Adj Close"][i])-1)
            if ohlc_renko[ticker]["bar_num"][i]>=2 and ohlc_renko[ticker]["obv slope"][i]>30:
                tickers_signal[ticker] = "Buy"
            elif ohlc_renko[ticker]["bar_num"][i]>-2:
                tickers_signal[ticker] = ""
    ohlc_renko[ticker]["return"] = np.array(tickers_return[ticker])


#plotting returns
fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows = 5, ncols = 1, sharex =True)
(ohlc_renko['GOOG']['return']+1).cumprod().plot(ax = ax1)
(ohlc_renko['MSFT']['return']+1).cumprod().plot(ax = ax2)
(ohlc_renko['AAPL']['return']+1).cumprod().plot(ax = ax3)
(ohlc_renko['INTC']['return']+1).cumprod().plot(ax = ax4)
(ohlc_renko['F']['return']+1).cumprod().plot(ax = ax5)
plt.show()




strategy_df = pd.DataFrame()




for ticker in tickers:
    strategy_df[ticker] = ohlc_renko[ticker]['return']




strategy_df['return'] = strategy_df.mean(axis = 1)


#evaluating stats




cagr = {}
sharpe_ratios = {}
max_drawdowns = {}




for ticker in tickers:
    cagr[ticker] = CAGR(ohlc_renko[ticker])
    sharpe_ratios[ticker] = sharpe(ohlc_renko[ticker],0.025)
    max_drawdowns[ticker] = max_dd(ohlc_renko[ticker])






