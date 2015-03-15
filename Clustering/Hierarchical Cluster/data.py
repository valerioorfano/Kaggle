import pandas as pd
import numpy as np
from sklearn.decomposition import PCA as pca

def load_data():
    data = pd.read_csv('samsungdata.csv')
    data.drop(['Unnamed: 0', 'activity'], axis=1, inplace=True)
    data.dropna(inplace=True)
    return data
    
    
    