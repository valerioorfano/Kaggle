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
print "Let us resample at daily and weekly period"
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
model = sm.ols(formula ='total  ~daylight', data = weekly).fit()
reg = pd.Series(model.fittedvalues.values, index = weekly['daylight'])
reg.plot(color='red')
# model.params.daylight is the slope
weekly['daylight_trend'] = model.predict(weekly[['daylight']])
weekly['daylight_detrend'] = weekly.total - weekly.daylight_trend
plt.figure()
plt.scatter(x=weekly.daylight, y=weekly.daylight_detrend)
plt.xlabel('daylight')
plt.ylabel('daylight_detrend')
plt.plot(weekly.daylight, np.zeros(weekly.shape[0])) # x = 0 horizonta line
plt.title('weekly traffic detrended')
#iid distribution around 0
#The "adjusted weekly count" plotted here can be thought of as the number of cyclists we'd expect to see if the hours of daylight were not a factor
tsaplots.plot_acf(weekly.daylight_detrend, lags=20)
#test for stationarity
weekly[['total', 'daylight_trend']].plot()
weekly[['daylight_detrend']].plot()  
plt.ylabel("adjusted total weekly riders")
# So far we have worked with weekly data. Daylight affect weekly data.
"""Accounting for Day of the Week
---------------------------------------------------------------------"""
#For daily data another factor that affect the trend is the day of the week
days = ['mon','tue','wed', 'thu','fri','sat','sun']
daily['dayofweek'] = daily.index.dayofweek
group = daily.groupby('dayofweek').total.mean()
group.index = days
plt.figure()
group.plot()
#In Seattle bike is not just for advertisement since on monday the number or biker are 2.5 times higher then sunday. Bicycle is a means of commuting.
#Let's check whether there is a relationship between the bycicle counters and dayofweek + daylight
#One Hot encoding of the dayofweek.
for i in range(7):
    daily[days[i]] = (daily.index.dayofweek ==i).astype(float)
    
X = daily[days + ['daylight']]
y = daily.total
model = sm.ols(formula = 'y ~ X', data=daily).fit()
daily['dayweek_daylight_trend'] = model.predict(X)
daily['dayweek_daylight_detrend'] = daily.total - daily.dayweek_daylight_trend
daily[['total','dayweek_daylight_detrend']].plot()
#'dayweek_daylight_detrend' is the timeseries if day of the week and daylight would not be factors
"""Accounting for temperature and precipitation
---------------------------------------------------------------------"""
weather = pd.read_csv('seatacweather.csv', index_col = 'DATE', parse_dates=True, usecols=[2,3,6,7])
# temperatures are in 1/10 deg C; convert to F
weather['TMIN'] = 0.18 * weather['TMIN'] + 32
weather['TMAX'] = 0.18 * weather['TMAX'] + 32
# precip is in 1/10 mm; convert to inches
weather['PRCP'] /= 254
plt.figure()
weather['TMIN'].resample('w','min').plot(legend = True)
weather['TMAX'].resample('w','max').plot(legend = True)
plt.title('Temperature in Seattle')
plt.ylabel('Temp Fareneiht')
plt.figure()
weather['PRCP'].resample('w','sum').plot(legend = True)
plt.ylabel('Weekly precipitation (in)')
plt.title("Precipitation in Seattle")
#Merge data
daily = daily.join(weather)
X = daily[days + ['daylight','TMIN','TMAX','PRCP']]
y = daily.total
model = sm.ols(formula = 'y ~ X', data = daily).fit()
daily['over_all'] = model.fittedvalues.values
plt.figure()
daily[['total','over_all']].plot()
daily['over_all_detrend'] = daily['total'] - daily['over_all']
plt.figure()
daily['over_all_detrend'].plot()
plt.ylabel('daily bicycle traffic detrended')
#Let's check if there is still a trend as we did initially, threfore we use a Mean Smoothing with 30 days
plt.figure()
pd.stats.moments.rolling_mean(daily.over_all_detrend, window=30).plot()
#Even with detrending there is still positive trend
#We want to add another factor depending the the time moving ahead. Maybe there is a positive trend just because of the time goes by, meaning there has been really an increase number of bikers
daily['daycount'] = np.arange(daily.shape[0])
columns = days + ['daycount','daylight','TMIN','TMAX','PRCP']
X = daily[columns]
y = daily.total
final_model = sm.ols(formula = 'y ~ X', data = daily).fit()
daily['final_trend'] = model.fittedvalues.values
daily[['total','final_trend']].plot()
daily['final_detrend'] = daily['total'] - daily['final_trend']
plt.figure()
daily['final_detrend'].plot()
plt.ylabel('final daily bicycle traffic detrended')
#Again let's see if we still have a trend in our data
pd.stats.moments.rolling_mean(daily.final_detrend,window=30).plot()
"""--------------------------------------------------------------- """
print "Effect of the PRCP"
slope = final_model.params[columns.index('PRCP') + 1]
print("{:.2f}  daily crossings lost per inch of rain".format(slope))


"""--------------------------------------------------------------- """
print "Effect of the TMIN, TMAX" 
slopes = final_model.params[[columns.index('TMIN') + 1, columns.index('TMAX') + 1]]
slopes.index = [0,1]
print("{:.2f} daily crossings lost per each Farenheit of TMIN, {:.0f} daily crossings gained per each Farenheit of TMAX with the presence of the other factors".format(slopes[0], slopes[1]))

"""--------------------------------------------------------------- """
print "Effect of the daylight" 
slope = final_model.params[columns.index('daylight') + 1]
print("{:.2f} daily crossings gained per each hour of daylight  with the presence of the other factors".format(slope))

"""--------------------------------------------------------------- """
print "Effect of the day of the week - monday" 
slope = final_model.params[columns.index('mon') + 1]
print("{:.2f} daily crossings gained if day = monday with the presence of the other factors".format(slope))

print "Effect of the day of the week - sunday" 
slope = final_model.params[columns.index('sun') + 1]
print("{:.2f} daily crossings lost if day = sunday with the presence of the other factors".format(slope))

"""Is ridership increasing?"""
"""--------------------------------------------------------------- """
print "definitely there is an ridership incresing because when we detrend the timeseries from dayofweek, daylight, TMIN,TMAX,PRCP the rollingaverage shows still a positive trend."
print "Effect of the daycount" 
slope = final_model.params[columns.index('daycount') + 1]
print("{:.2f} daily crossings gained for each daycount with the presence of the other factors".format(slope))

