import csv
import numpy as np
import pandas as pd
from kprototype import *
from sklearn.preprocessing import StandardScaler
# read the data into a 2-dimensional array
rawdata = pd.read_csv('creditscreen.csv')
rawdata = np.array(rawdata)
x = np.array([row for row in rawdata if '?' not in row])
x = x[:, :-1]
c = x[:,[0,3,4,5,6,8,9,11,12]]
n = [[float(y) for y in z] for z in x[:,[1,2,7,10,13,14]]]
#standardization of numerical values
s = StandardScaler().fit_transform(n)
iteration = 1	
c = KPrototypes([s,c],2,None)
c.Initialization()
while c.reallocation() > 0:
    print "Iteration: {}".format(iteration) 
    iteration += 1
        
#write the clusters to a csv file
#convert cluster values to int
       
print("Writing clusters to Modes file...")
with open ('Centroids.csv','w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([['Centroids for numeric data:']])
    writer.writerows([['    ']])
    writer.writerows(c.centroids)
    writer.writerows([['    ']])
    writer.writerows([['Modes for categorical data:']])
    writer.writerows([['    ']])
    writer.writerows(c.modes)
print("Writing clusters to file...")
with open ('Clusters.csv','w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([[x] for x in c.clustership])
    