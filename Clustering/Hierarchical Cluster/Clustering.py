from data import *
import matplotlib.pyplot as plt
from ml2 import *
import numpy as np


samsung = load_data()
#plt.scatter(data[:,0], data[:,1])
a,b,c,d,e = clustering(samsung)  