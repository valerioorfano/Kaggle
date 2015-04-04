import statsmodels
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


dta = sm.datasets.sunspots.load_pandas().data
rng = pd.date_range('1700', '2009', freq='A')
dta.index = rng
dta = dta.drop('YEAR',axis=1)
dta.plot(figsize=(12,8))
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)
#Let's check it the ts is stationary using Dickey-Fuller test.
unit_root = sm.tsa.adfuller(dta.SUNACTIVITY)
#The first element if the result cintains the p-value
#H0: ts is not stationary
#HA: ts is stationary
print "P-value {}".format(unit_root[1])
# P-value is 0.05 the limit for a 95% Confidence Level

#Looking at ACF plot, there is  a correlation with time series at the time t-1 and t-2. So we can think to use AR(2) of order 2
#Looking at Partial ACF plot, there is  a sharp cutoff at lag3. 
arma_mod20 = sm.tsa.ARMA(dta, (2,0)).fit()
print arma_mod20.params
arma_mod30 = sm.tsa.ARMA(dta, (3,0)).fit()
print arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic
print arma_mod30.params
print arma_mod30.aic, arma_mod30.bic, arma_mod30.hqic
#ARMA 3 lags is better for AIC metric
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax = arma_mod30.resid.plot(ax=ax)
#residuals of the regression model doesn't look like a whote noise.
#Mean and Variance not constant


resid = arma_mod30.resid
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = statsmodels.graphics.api.qqplot(resid, line='q', ax=ax, fit=True)
#qqplot doesn't show a normal distribution
#Let's show acf and paf of the residuals
"""
The Ljung–Box test is commonly used in autoregressive integrated moving average (ARIMA) modeling. Note that it is applied to the residuals of a fitted ARIMA model, not the original series, and in such applications the hypothesis actually being tested is that the residuals from the ARIMA model have no autocorrelation.
The test is applied to the residuals of a time series after fitting an ARMA(p,q) model to the data. The test examines m autocorrelations of the residuals. If the autocorrelations are very small, we conclude that the model does not exhibit significant lack of fit.
The Ljung-Box Q (LBQ) statistic tests the null hypothesis that autocorrelations up to lag k equal zero (that is, the data values are random and independent up to a certain number of lags--in this case 12). If the LBQ is greater than a specified critical value, autocorrelations for one or more lags might be significantly different from zero, indicating the values are not random and independent over time
H0: 	The model does not exhibit lack of fit. 
Ha: 	The model exhibits lack of fit. The hypothesis of randomness is rejected if

      QLB > CHSPPF((1-alpha),h) 

where CHSPPF is the percent point function of the chi-square distribution. """
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)	

r,q,p = sm.tsa.acf(resid, qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print table.set_index('lag')
"""Q should go to zero, if it goes up then there is autocorrelation."""
#Let's see how predictors differ from the ts to check how it would work with the predictions
predict_sunspots = arma_mod30.predict('1990', '2012', dynamic=True)
print predict_sunspots
ax = dta.ix['1950':].plot(figsize=(12,8))
ax = predict_sunspots.plot(ax=ax, style='r--', label='Dynamic Prediction');
ax.legend();
ax.axis((-20.0, 38.0, -4.0, 200.0))
#Prediction are not very good


def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()
    
mean_forecast_err(dta.SUNACTIVITY, predict_sunspots)

# Exercise: Can you obtain a better fit for the Sunspots model? (Hint: sm.tsa.AR has a method select_order)
ar_mod = sm.tsa.AR(dta)
print ar_mod.select_order(maxlag=20, ic='bic')
#9 is the best order for ARMA model
ar_mod90 = sm.tsa.ARMA(dta,(9,0)).fit()	
statsmodels.graphics.api.qqplot(ar_mod90.resid, line='q', fit=True)
#Not yet a good model fit
predict_sunspots = ar_mod90.predict('1990', '2012', dynamic=True)
print predict_sunspots
ax = dta.ix['1950':].plot(figsize=(12,8))
ax = predict_sunspots.plot(ax=ax, style='r--', label='Dynamic Prediction');
ax.legend();
ax.axis((-20.0, 38.0, -4.0, 200.0))
#predictions are better
#-----------------------------------------------------------------------------
macrodta = sm.datasets.macrodata.load_pandas().data
macrodta.index = pd.Index(sm.tsa.datetools.dates_from_range('1959Q1', '2009Q3'))
cpi = pd.Series(macrodta["cpi"], index= macrodta.index)
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax = cpi.plot(ax=ax);
ax.legend()
#There is definitely a positive trend
print sm.tsa.adfuller(cpi)[1]
#Not a stationary ts
#To transform a no-stationary into stationary ts we need to differentiate
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(cpi, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(cpi, lags=40, ax=ax2)
#Since we need to differentiate we use ARIMA  I stand for integration
arima_mod211 = sm.tsa.ARIMA(cpi,(2,1,1)).fit()
statsmodels.graphics.api.qqplot(arima_mod211.resid, line='q', fit=True)
predict = arima_mod211.predict('1960Q1','2009Q3', dynamic=False, typ='levels')
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
cpi.plot(ax=ax)
predict.plot(ax=ax)
"""WOW very good predictions"""
"""Residuals white nose"""
"""Test for stationarity"""
arima_mod211.plot()
sm.graphics.tsa.plot_acf(arima_mod211.resid)
#-----------------------------------------------------------------------------




	

	
