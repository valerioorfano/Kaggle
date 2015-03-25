import csv
from kmodes import *
import numpy as np
import pandas as pd

# read the data into a 2-dimensional array
rawdata = pd.read_csv('soybeans.csv')
rawdata = np.array(rawdata)
data = rawdata[:-1, :-1]
iteration = 1	
c = KModes(data,4)
c.BuildInitialClusters()
while c.BuildClusters() > 0:
    print "Iteration: {}".format(iteration) 
    iteration += 1
        
#write the clusters to a csv file
#convert cluster values to int
       
print("Writing clusters to Modes file...")
with open ('Modes.csv','w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([[int(y) for y in x]  for x in c.clustervalues])
print("Writing clusters to file...")
with open ('Clusters.csv','w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([[x] for x in c.clustership])