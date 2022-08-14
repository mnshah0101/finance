#monthly portfolio rebalancing 

#Choose a universe of stocks
#Stick to this group of stocks
#Build fixed individual posistions, meaning all positions of same value by picking m number of stocks
#Rebalance the portfolio by removing X worse stocks and replacing them with X top stocks from the universe- can stocks be picked again?
#Backtest the strategy and compare the KPIs with simple buy and hold strategy of corresponding index - not in Dow Jones right now
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import copy


def CAGR(DF):
    """
    compound cumulative anaual growth rate
    """
    df = DF.copy()
    df['cum_return']= (1+df['mon_return']).cumprod()
    n = len(df)/51
    CAGR = (df['cum_return'].iloc[-1])**(1/n) -1
    return CAGR
        
def volatilty(DF):
    """
    annlualized  volatility
    """
    df=DF.copy()
    avol = df['mon_return'].std()*np.sqrt(51)
    return avol
def sharpe(DF,n):
    """
    sharpe ratio
    """
    return (CAGR(DF)-n)/volatilty(DF)     
def max_drawdown(DF):
    """
    max drawdown
    """
    df = DF.copy()
    df['cum_return'] = (df['mon_return']+1).cumprod()
    df['rolling max'] = df['cum_return'].cummax()
    df['drawdown'] = df['rolling max'] - df['cum_return'] #gives you difference between each point and the rolling max
    return (df['drawdown']/df['rolling max']).max() #gives you percent
def calmar(DF):
    return (CAGR(DF)/max_drawdown(DF))


#get historical ticker data
tickers = ['MMM','AXP', 'AAPL', 'BA','CAT','CVX','CSCO','KO','DD','XOM','GE','GS','HD', "INTC", 'JPM','IBM', 'JNJ','MCD','MRK','MSFT','NKE','PFI','PG','TRV','UNH','RTX','VZ','V','WMT',"DIS"]
ohlcv= {}
for ticker in tickers:
    temp = yf.download(ticker, period = '2y', interval='5d')
    temp.dropna(inplace =True)
    ohlcv[ticker]=temp

ohlcv_dict = copy.deepcopy(ohlcv)
return_df =pd.DataFrame()

for ticker in tickers:
    ohlcv_dict[ticker]['mon_return'] = ohlcv_dict[ticker]["Adj Close"].pct_change()
    return_df[ticker] = ohlcv_dict[ticker]['mon_return']


def pfolio(DF,m,x):
    """
    DF is the monthly return of all stocks
    m is the number of stocks in the portfolio
    x is the number of stocks to be replaced
    """
    df = DF.copy() #create copy of monthly return datagrame
    portfolio = ['AAPL','DIS','BA','CAT','CSCO','KO']# this is your portfolio list
    monthly_ret = [0] #montly returns for portfolio, your first reutnr will be 0
    for i in range(1,len(df)): #for every row from the second to the last row
        if len(portfolio) >0: #if the portfolio is not empty
            monthly_ret.append(df[portfolio].iloc[i,:].mean()) #for this month in the portfolio, calculate the average monthly return, and add that to the list 
            bad_stocks = df[portfolio].iloc[i,:].sort_values(ascending = True)[:x].index.values.tolist() #find the worst stocks in that month from the portfolio,sort the series by ascending, find the x worst stocks, and add them to the worst_stocks list, by index
            portfolio = [t for t in portfolio if t not in bad_stocks] #your portfolio becomes stocks that aren't a bad stock
        fill = m - len(portfolio) # the number you need to fill is the amount you have minus the amount you need
        new_picks = df[[t for t in tickers]].iloc[i,:].sort_values(ascending = False)[:fill].index.values.tolist() #pick the 3 stocks in the month that performed the best
        portfolio = portfolio + new_picks #combine your lists
    monthly_ret_df = pd.DataFrame(np.array(monthly_ret), columns = ['mon_return'])    #make a new dataframe with the array of monthly returns, with the column being called 'mon_return'
    return monthly_ret_df #return this df
                                      
        

#evaluate strategy

print(CAGR(pfolio(return_df,6,3)))
print(volatilty(pfolio(return_df,6,3)))
print(sharpe(pfolio(return_df,6,3),0.025))
print(max_drawdown(pfolio(return_df,6,3)))
print(calmar(pfolio(return_df,6,3)))








