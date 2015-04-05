import pandas
import matplotlib.pyplot as plt
import statsmodels.api as sm


dta = sm.datasets.macrodata.load_pandas().data
index = pandas.Index(sm.tsa.datetools.dates_from_range('1959Q1', '2009Q3'))
dta.index = index

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
dta.realgdp.plot(ax=ax)
legend = ax.legend(loc = 'upper left')
legend.prop.set_size(20)
"""The Hodrick–Prescott filter (also known as Hodrick–Prescott decomposition) is a mathematical tool used in macroeconomics, especially in real business cycle theory, to remove the cyclical component of a time series from raw data. It is used to obtain a smoothed-curve representation of a time series, one that is more sensitive to long-term than to short-term fluctuations.
The Hodrick-Prescott filter separates a time-series $y_t$ into a trend and a cyclical component"""
gdp_cycle, gdp_trend = sm.tsa.filters.hpfilter(dta.realgdp)

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
gdp_decomp = dta[['realgdp']]
gdp_decomp["trend"] = gdp_trend
gdp_decomp.ix["2000-03-31":,['trend','realgdp']].plot(ax=ax)


"""Baxter-King approximate band-pass filter: Inflation and Unemployment. # The Baxter-King filter is intended to explictly deal with the periodicty of the business cycle. By applying their band-pass filter to a series, they produce a new series that does not contain fluctuations at higher or lower than those of the business cycle. Specifically, the BK filter takes the form of a symmetric moving average """
bk_cycles = sm.tsa.filters.bkfilter(dta[["infl","unemp"]])
fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
bk_cycles.plot(ax=ax, style=['r--', 'b-'])
dta[["infl","unemp"]].plot(ax=ax)

"""Vector Autoregression (VAR), introduced by Nobel laureate Christopher Sims in 1980, is a powerful statistical tool in the macroeconomist's toolkit."""
dta = sm.datasets.macrodata.load_pandas().data
endog = dta[["infl", "unemp", "tbilrate"]]
index = sm.tsa.datetools.dates_from_range('1959Q1', '2009Q3')
dta.index = pandas.Index(index)
del dta['year']
del dta['quarter']
endog.index = pandas.Index(index) # DatetimeIndex or PeriodIndex in 0.8.0
endog.plot(subplots=True, sharex =True, figsize=(14,18))
var_mod = sm.tsa.VAR(endog.ix['1979-12-31':]).fit(maxlags=4, ic=None)
print var_mod.summary()
var_mod.test_normality() 
var_mod.test_whiteness() 

var_mod.test_causality('unemp', 'tbilrate', kind='Wald')
var_mod.model.select_order()
#We should use 2 lags where the Information criteria is minimum
var_mod = sm.tsa.VAR(endog.ix['1979-12-31':]).fit(maxlags=2, ic=None)
	
	
