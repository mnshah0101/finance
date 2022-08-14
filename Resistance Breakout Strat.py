#Resitance breakout strategy

#choose high volume stocks for this strategy
#define a breakout rule+ has to do with volume and price
#define stop loss signals


import numpy as np
import pandas as pd
import yfinance as yf
import copy
import time
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


#indicators


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
    vol = df["return"].std() * np.sqrt(60*78)
    return vol
def sharpe(DF,rf):
    "function to calculate sharpe ratio ; rf is the risk free rate"
    df = DF.copy()
    sr = (CAGR(df) - rf)/volatility(df)
    return sr    
def max_drawdown(DF):
    df = DF.copy()
    df['cum_return'] = (df['return']+1).cumprod()
    df['rolling max'] = df['cum_return'].cummax()
    df['drawdown'] = df['rolling max'] - df['cum_return'] #gives you difference between each point and the rolling max
    return (df['drawdown']/df['rolling max']).max() #gives you percent
def calmar(DF):
    return (CAGR(DF)/max_drawdown(DF))

def ATR(DF, n=14):
    df = DF.copy() #make new datagframe
    df['H-L'] = df['High']-df["Low"] #
    df['H-PC'] = abs(df['High'] - df['Adj Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Adj Close'].shift(1))
    df['TR'] = df[['H-L',"H-PC","L-PC"]].max(axis = 1,skipna= False)
    df['ATR'] = df['TR'].ewm(com = n, min_periods=n).mean()
    return df["ATR"]


def long_returns(DF):
    "function to calculate normal returns based on time frame"
    df = DF.copy()
    df['returns'] = df["Adj Close"].pct_change() +1
    df["cum_return"] = (df["returns"]).cumprod()
    return (df['cum_return'].iloc[-1])-1



# Download historical data for required stocks
tickers = ['STR', 'PRAX', 'SOUN','UONE', 'AERC']
ohlcv_data = {}

    



# looping over tickers and storing OHLCV dataframe in dictionary
for ticker in tickers:
    temp = yf.download(ticker,period='60d',interval='5m')
    temp.dropna(how="any",inplace=True)
    ohlcv_data[ticker] = temp



ohlcv_dict = copy.deepcopy(ohlcv_data)
tickers_signal ={} #tracks whether you should buy or not in the current state
tickers_return = {} #tracks the returns


#back testing

for ticker in tickers:
    print("calculating ATR and rolling max price for", ticker)
    ohlcv_dict[ticker]["ATR"] = ATR(ohlcv_dict[ticker],20)
    ohlcv_dict[ticker]["roll_max_cp"] = ohlcv_dict[ticker]["High"].rolling(20).max()
    ohlcv_dict[ticker]["roll_min_cp"] = ohlcv_dict[ticker]["Low"].rolling(20).min()
    ohlcv_dict[ticker]["roll_max_vol"] = ohlcv_dict[ticker]["Volume"].rolling(20).max()
    ohlcv_dict[ticker].dropna(inplace=True)
    tickers_signal[ticker] = '' #for each candle, have a buy or sell
    tickers_return[ticker] = [] #for each candle, have a return




for ticker in tickers:
    print('calculating returns for', ticker)
    
    for i in range(len(ohlcv_dict[ticker])): #for each row
        
        if tickers_signal[ticker] == '': #if the signal is empty
            tickers_return[ticker].append(0) #the return is 0 for the first iteration
            
            if ohlcv_dict[ticker]['High'][i]>=ohlcv_dict[ticker]['roll_max_cp'][i] and ohlcv_dict[ticker]['Volume'][i]>=1.5*ohlcv_dict[ticker]['roll_max_vol'][i-1]: #if the stock is 1.5 above the rolling max and the volume is over 1.5 times the max for the previous 20 periods, then buy, it will break resistance and go up  
                tickers_signal[ticker] = 'Buy'
            elif ohlcv_dict[ticker]['Low'][i]<=ohlcv_dict[ticker]['roll_min_cp'][i] and ohlcv_dict[ticker]['Volume'][i]>=1.5*ohlcv_dict[ticker]['roll_max_vol'][i-1]: #if its under 1.5 times the rolling min, and volume is 1.5 times less than previous 20 day min, sell, it will break resistance and go down, this is called reversal
                tickers_signal[ticker] = 'Sell'
        
        elif tickers_signal[ticker]== 'Buy': #if we have bought the stock
            
            if ohlcv_dict[ticker]["Low"][i]<ohlcv_dict[ticker]["Close"][i-1]-ohlcv_dict[ticker]['ATR'][i-1]: # if the low for the last 5 mins is less than what we bought it at minus expected variability
                tickers_signal[ticker] = "" #change from a buy to a stop position, we want to sell
                tickers_return[ticker].append(((ohlcv_dict[ticker]['Close'][i-1]-ohlcv_dict[ticker]['ATR'][i-1])/ohlcv_dict[ticker]['Close'][i-1])-1) #calculate percent returns, assuming we sell as stop loss level
            elif ohlcv_dict[ticker]['Low'][i]<=ohlcv_dict[ticker]['roll_min_cp'][i] and ohlcv_dict[ticker]['Volume'][i]>=1.5*ohlcv_dict[ticker]['roll_max_vol'][i-1]: #if the stock goes back into sell range, we sell and calculate returns
                tickers_signal[ticker] = "Sell" #
                tickers_return[ticker].append(((ohlcv_dict[ticker]['Close'][i])/ohlcv_dict[ticker]['Close'][i-1])-1) #calculate percent returns from the last candle
            else: #if none of these are true, lets stay with our buy until we can stop or sell our position
                tickers_return[ticker].append((ohlcv_dict[ticker]['Close'][i]/ohlcv_dict[ticker]['Close'][i-1])-1) #else, if we have a buy
        
        elif tickers_signal[ticker] == "Sell": #if we want to sell
            
            if ohlcv_dict[ticker]["High"][i]>ohlcv_dict[ticker]["Close"][i-1] + ohlcv_dict[ticker]["ATR"][i-1]: #if today's high is is greater than yesterday's close and 1.5 above the ATR, we are in a breakout, and should close our position
                tickers_signal[ticker] = ""
                tickers_return[ticker].append((ohlcv_dict[ticker]["Close"][i-1]/(ohlcv_dict[ticker]["Close"][i-1] + ohlcv_dict[ticker]["ATR"][i-1]))-1)
            elif ohlcv_dict[ticker]["High"][i]>=ohlcv_dict[ticker]["roll_max_cp"][i] and ohlcv_dict[ticker]["Volume"][i]>1.5*ohlcv_dict[ticker]["roll_max_vol"][i-1]: #if we are increasing, we should buy again
                tickers_signal[ticker] = "Buy"
                tickers_return[ticker].append((ohlcv_dict[ticker]["Close"][i-1]/ohlcv_dict[ticker]["Close"][i])-1)
            else: #else, lets just take our profits 
                tickers_return[ticker].append((ohlcv_dict[ticker]["Close"][i-1]/ohlcv_dict[ticker]["Close"][i])-1)                
                
    ohlcv_dict[ticker]['return'] = np.array(tickers_return[ticker]) 
        
             
                
#evaluate the strategy

strategy_df = pd.DataFrame()


for ticker in tickers:
    strategy_df[ticker] = ohlcv_dict[ticker]['return']


strategy_df['return'] = strategy_df.mean(axis = 1)


print(sharpe(strategy_df,0.0025))
print(CAGR(strategy_df))
print(max_drawdown(strategy_df))


(1+strategy_df).cumprod().plot()



cagr = {}
sharpes = {}
max_drawdowns = {}



for ticker in tickers:
    print("calculating kpis for", ticker)
    cagr[ticker] = CAGR(ohlcv_dict[ticker])
    sharpes[ticker] = sharpe(ohlcv_dict[ticker],long_returns(ohlcv_dict[ticker]))
    max_drawdowns[ticker] = max_drawdown(ohlcv_dict[ticker])

#plotting

fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(ncols=1, nrows = 5, figsize = (12,12), sharex = True)
  
(ohlcv_dict['STR']['Adj Close'].pct_change()+1).cumprod().plot(ax = ax1, label = 'Long')
(1+strategy_df['STR']).cumprod().plot(ax= ax1, label ='Strat')
ax1.legend()
plt.show()
(ohlcv_dict['PRAX']['Adj Close'].pct_change()+1).cumprod().plot(ax = ax2, label = 'Long')
(1+strategy_df['PRAX']).cumprod().plot(ax= ax2, label = 'Strat')
ax2.legend()
plt.show()

(ohlcv_dict['SOUN']['Adj Close'].pct_change()+1).cumprod().plot(ax = ax3)
(1+strategy_df['SOUN']).cumprod().plot(ax= ax3)
plt.show()

(ohlcv_dict['UONE']['Adj Close'].pct_change()+1).cumprod().plot(ax = ax4)
(1+strategy_df['UONE']).cumprod().plot(ax= ax4)
plt.show()

(ohlcv_dict['AERC']['Adj Close'].pct_change()+1).cumprod().plot(ax = ax5)
(1+strategy_df['AERC']).cumprod().plot(ax= ax5)
plt.show()



