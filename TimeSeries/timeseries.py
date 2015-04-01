import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime
import statsmodels.formula.api as sm
from utils import *
import statsmodels.graphics.tsaplots as  tsaplots


hourly = pd.read_csv('FreemontHourly.csv',index_col='Date',parse_dates=True)
hourly.columns = ['northbound', 'southbound']
hourly['total'] = hourly.northbound + hourly.southbound
# Let us resample at daily and weekkly period
daily = hourly.resample('D', how='sum')
weekly = hourly.resample('W', how='sum')
daily[daily.columns].plot()
# Not clear to identify if there is a positive trend.
#For this reason let us use the Mean Smoothing to check possible positive trend
smooth30 = pd.stats.moments.rolling_mean(daily.total, window=30)
plt.figure()
smooth30.plot()
smooth30 = pd.DataFrame(smooth30)
smooth30.columns = ['smooth']
smooth30['num'] = range(smooth30.shape[0])
model = sm.ols(formula ='smooth ~ num', data = smooth30).fit()
start_row = smooth30.shape[0] - len(model.fittedvalues.values)
reg = pd.Series(model.fittedvalues.values, index = smooth30.index[start_row:])
reg.plot(color='yellow')

#We can see the trend is lightly positive trend
#There is a peack in April probabaly because of the daylights hours
weekly['daylight'] = map(hours_of_daylight, weekly.index)
daily['daylight'] = map(hours_of_daylight, daily.index)
#The trend is likely caused by the  daylight
plt.figure()
plt.scatter(x=weekly.daylight, y=weekly.total)
plt.xlabel('daylight')
plt.ylabel('weekly bicycle trafic')
model = sm.ols(formula ='total ~ daylight', data = weekly).fit()
reg = pd.Series(model.fittedvalues.values, index = weekly['daylight'])
reg.plot(color='red')
# model.params.daylight is the slope
weekly['daylight_trend'] = model.predict(weekly[['daylight']])
weekly['detrend'] = weekly.total - weekly.daylight_trend
plt.figure()
plt.scatter(x=weekly.daylight, y=weekly.detrend)
plt.xlabel('daylight')
plt.ylabel('detrend')
plt.plot(weekly.daylight, np.zeros(weekly.shape[0])) # x = 0 horizonta line
plt.title('weekly traffic detrended')
#iid distribution around 0
#The "adjusted weekly count" plotted here can be thought of as the number of cyclists we'd expect to see if the hours of daylight were not a factor
tsaplots.plot_acf(weekly.detrend, lags=20)
#test for stationarity
weekly[['total', 'daylight_trend']].plot()
plt.figure()
weekly[['detrend']].plot()  
plt.ylabel("adjusted total weekly riders")
# So far we have worked with weekly data. Daylight affect weekly data.
"""Accounting for Day of the Week"""
#For daily data another factor that affect the trend is the day of the week
day = ['mon','tue','wed', 'thu','fri','sat','sun']
