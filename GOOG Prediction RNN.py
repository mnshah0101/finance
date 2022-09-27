#!/usr/bin/env python
# coding: utf-8

# In[203]:


import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from collections import deque
import random


# In[206]:


tickers = ['GOOG', 'AAPL', 'AMZN', 'MSFT']


# In[ ]:





# In[ ]:





# In[209]:


def get_dataframes(tickers_list):
    stock_dict = {}
    for ticker in tickers_list:
        stock_dict[ticker] = yf.download(ticker, period="7d", interval='1m')
    for ticker in tickers_list:
        df= stock_dict[ticker]
        df.reset_index(inplace = True)
        df.drop(['Datetime','Open','High','Low','Close'], axis = 1, inplace = True)
    return stock_dict
    
    


# In[210]:


ov_dict = get_dataframes(tickers)


# In[219]:


ov_dict['GOOG']


# In[221]:


def join_dataframes(dict_to_join, tickers_list):
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
        


# In[222]:


df_full = join_dataframes(ov_dict, tickers)


# In[223]:


df_full


# In[224]:


def target_class(current,future):
    if future>current:
        return 1
    else:
        return 0


# In[ ]:





# In[225]:


def create_target_class(df,stock,shift_by):
    return list(map(target_class, df[stock], df[stock].shift(shift_by)))
    


# In[226]:


df_full['Target'] = create_target_class(df_full, 'GOOG_close', -3)


# In[227]:


df_full


# In[228]:


plt.hist(df_full['Target'])


# In[229]:


def create_val_set(df,val_size):
    length = len(df)
    main = df[:int(length*(1-val_size))]
    val = df[int(length*(1-val_size)):]
    return main, val


# In[230]:


main_df, val_df = create_val_set(df_full, 0.05)


# In[ ]:





# In[ ]:





# In[231]:


from sklearn import preprocessing  # pip install sklearn ... if you don't have it!

def preprocess_df(df, max_seq):
    for col in df.columns:  # go through all of the columns
        if col != "Target":  # normalize all ... except for the target itself!
            df[col] = df[col].pct_change()  # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
    sequential_data = []
    for i in range(len(df)-max_seq):
        sequence = df.drop("Target", axis = 1).iloc[i:i+max_seq]
        target = df['Target'].iloc[i+max_seq]
        sequential_data.append((sequence,target))
        buys = []  # list that will store our buy sequences and targets
    sells = []  # list that will store our sell sequences and targets

    for seq, target in sequential_data:  # iterate over the sequential data
        if target == 0:  # if it's a "not buy"
            sells.append([seq, target])  # append to sells list
        elif target == 1:  # otherwise if the target is a 1...
            buys.append([seq, target])  # it's a buy!

    random.shuffle(buys)  # shuffle the buys
    random.shuffle(sells)  # shuffle the sells!

    lower = min(len(buys), len(sells))  # what's the shorter length?

    buys = buys[:lower]  # make sure both lists are only up to the shortest length.
    sells = sells[:lower]  # make sure both lists are only up to the shortest length.

    sequential_data = buys+sells  # add them together
    random.shuffle(sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.
    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), y  # return X and y...and make X a numpy array!

        
            
            


# In[232]:


X_train, y_train = preprocess_df(main_df, 60)
X_validation, y_validation = preprocess_df(val_df, 60)


# In[240]:


import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization


# In[234]:


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


# In[235]:


model.summary()


# In[236]:


opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)


# In[237]:


history = model.fit(
    X_train, np.array(y_train),
    batch_size=64,
    epochs=10,
    validation_data=(X_validation, np.array(y_validation))
)


# In[241]:


pd.DataFrame(model.history.history).plot()


# In[242]:


predictions = model.predict(X_train)
predictions = np.argmax(predictions, axis=1)


# In[187]:


from sklearn.metrics import classification_report, confusion_matrix


# In[201]:


print(classification_report(y_train,predictions))


# In[202]:


print(confusion_matrix(y_train,predictions))


# In[244]:


next_minute_data = yf.download("GOOG", period = '5m', interval='1m')


# In[246]:


propertynext_minute_data.iloc[1:5]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




