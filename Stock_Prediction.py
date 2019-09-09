# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from pandas_datareader import data as pdr 
import fix_yahoo_finance as yf 

get_ipython().system('pip install yfinance --upgrade --no-cache-dir')
yf.pdr_override()

from google.colab import drive

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import GradientBoostingRegressor

from sklearn import metrics


#%%
df_full = pdr.get_data_yahoo("AMZN", start="2018-01-01").reset_index()
df_full.to_csv('AMZN.csv', index=False)


#%%
df_full.shape


#%%
df_full.head()


#%%
df_full.describe()


#%%
df_full["Adj Close"].plot()


#%%
df_full.set_index("Date", inplace=True)


#%%
window_size=32
num_samples=len(df_full)-window_size

indices=np.arange(num_samples).astype(np.int)[:,None]+np.arange(window_size+1).astype(np.int)
data = df_full['Adj Close'].values[indices]
x = data[:,:-1]
y = data[:,-1]


#%%
split_fraction=0.8
ind_split=int(split_fraction*num_samples)


#%%
x_train=x[:ind_split]
y_train=y[:ind_split]
x_test=x[ind_split:]
y_test=y[ind_split:]


#%%
y_test.shape


#%%
def get_performance (model_pred):
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, model_pred).round(4))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, model_pred).round(4))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, model_pred)).round(4))

def get_plot (model_pred):
    plt.scatter(model_pred, y_test, color="gray")
    plt.plot(y_test, y_test, color='red', linewidth=2)


#%%
y_pred_lag=np.roll(y_test, 1)


#%%
np.roll(y,1)


#%%
get_performance(y_pred_lag)


#%%
get_plot(y_pred_lag)


#%%
#Linear_Regression_Model
model_lr=LinearRegression()
model_lr.fit(x_train, y_train)


#%%
get_ipython().run_line_magic('pinfo', 'model_lr.fit')


#%%
y_pred_lr=model_lr.predict(x_test)


#%%
get_performance(y_pred_lr)


#%%
get_plot(y_pred_lr)


#%%
#Ridge_Regression_Model
model_ridge = Ridge()
model_ridge.fit(x_train, y_train)

y_pred_ridge=model_ridge.predict(x_test)


#%%
get_performance(y_pred_ridge)


#%%
get_plot(y_pred_ridge)


#%%
model_ridge.coef_


#%%
df_comp=pd.DataFrame({"lag":np.absolute(y_test-y_pred_lag), "lr":np.absolute(y_test-y_pred_lr), "ridge":np.absolute(y_test-y_pred_ridge)})


#%%
df_comp.head()


#%%
y_pred_lag


#%%
df_comp.plot.bar(figsize=(16, 6))
#plt.ylim(0, 10)
#plt.xlim(9, 20)


#%%
df_comp.plot.bar(figsize=(16, 6))
plt.ylim(0, 10)
plt.xlim(11, 20)


#%%



