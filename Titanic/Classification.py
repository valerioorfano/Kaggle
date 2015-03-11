import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric import smoothers_lowess
from pandas import Series, DataFrame
from patsy import dmatrices
from sklearn import datasets, svm
from data import *
from plot import *
from ml import *

train_data = load_data()
#plot_data(train_data)
train_data = exploratory(train_data)
X = train_data.iloc[:,1:]
y = train_data['Survived']

#Let's first use some singolar important algorithm
#Using SearchCVGrid to calculate the best parameters
models = learn(X,y,5)
#Training 
metrica(X,y,models,5)
