#GOOGLE Recurrent Neural Net Price Movement Predictor


#imports
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from collections import deque
import random



#create list of tickers, we want to predict GOOG price using GOOG stock data and other related companeis
tickers = ['GOOG', 'AAPL', 'AMZN', 'MSFT']




def get_dataframes(tickers_list):
    """
    create dictionary of Close and Volume data for each stock
    """
    stock_dict = {}
    for ticker in tickers_list:
        stock_dict[ticker] = yf.download(ticker, period="7d", interval='1m')
    for ticker in tickers_list:
        df= stock_dict[ticker]
        df.reset_index(inplace = True)
        df.drop(['Datetime','Open','High','Low','Close'], axis = 1, inplace = True)
    return stock_dict
    
    

#create this dictionary
ov_dict = get_dataframes(tickers)


def join_dataframes(dict_to_join, tickers_list):
    """
    create full dictionary of all data from dictionary
    """
    df = pd.DataFrame()
    concat_list = []
    for ticker in tickers_list:
        dict_to_join[ticker].columns = ["{}_close".format(ticker),"{}_volume".format(ticker)]
    for item in tickers_list:
        df = pd.concat([df,dict_to_join[item]], axis = 1)
    df.dropna(inplace= True)
    columns = df.columns
    for col in columns:
        df = df[getattr(df, col) != 0]

    return df


#create this full data frame
df_full = join_dataframes(ov_dict, tickers)


def target_class(current,future):
    """
    target class function, 1 when stock goes up, 0 when stock goes down
    """
    if future>current:
        return 1
    else:
        return 0



def create_target_class(df,stock,shift_by):
    """
    creates this target class
    """
    return list(map(target_class, df[stock], df[stock].shift(shift_by)))
    

#create this target class for Google close price
df_full['Target'] = create_target_class(df_full, 'GOOG_close', -3)




#Show distribution of target classes
plt.hist(df_full['Target'])
plt.show()

# In[229]:


def create_val_set(df,val_size):
    """
    creates a main training and validation set
    """
    length = len(df)
    main = df[:int(length*(1-val_size))]
    val = df[int(length*(1-val_size)):]
    return main, val


#create main and val sets for df_full
main_df, val_df = create_val_set(df_full, 0.05)


from sklearn import preprocessing  


def preprocess_df(df, max_seq):
    """
    normalization, scaling, and rebalancing of data, and create sequences to run through neural net
    """
    for col in df.columns:  
        if col != "Target": 
            df[col] = df[col].pct_change() 
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
    sequential_data = []
    for i in range(len(df)-max_seq):
        sequence = df.drop("Target", axis = 1).iloc[i:i+max_seq]
        target = df['Target'].iloc[i+max_seq]
        sequential_data.append((sequence,target))
        buys = []  
    sells = []  

    for seq, target in sequential_data:  
        if target == 0:  
            sells.append([seq, target])  
        elif target == 1:  

    random.shuffle(buys)  
    random.shuffle(sells)  

    lower = min(len(buys), len(sells)) 

    buys = buys[:lower]  
    sells = sells[:lower]  

    sequential_data = buys+sells  
    random.shuffle(sequential_data)  
    X = []
    y = []

    for seq, target in sequential_data:  
        X.append(seq)  
        y.append(target)  

    return np.array(X), y 



#create training and validaiton sets
X_train, y_train = preprocess_df(main_df, 60)
X_validation, y_validation = preprocess_df(val_df, 60)


#imports for RNN architecture
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization

#create RNNN model (call model.summary() for model summary
model = Sequential()
model.add(LSTM(256, input_shape=(X_train.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))


opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)


#fit model to X_train and y_train
history = model.fit(
    X_train, np.array(y_train),
    batch_size=64,
    epochs=10,
    validation_data=(X_validation, np.array(y_validation))
)


#plot accuracy and loss for validaiton and training sets
pd.DataFrame(model.history.history).plot()
plt.show()

#create predictions from X_Train
predictions = model.predict(X_train)
predictions = np.argmax(predictions, axis=1)


#evaluation of model (around 60%)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_train,predictions))
print(confusion_matrix(y_train,predictions))







