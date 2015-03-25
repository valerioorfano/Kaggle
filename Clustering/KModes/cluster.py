import csv
import numpy as np
import pandas as pd
from kmodes import *
# read the data into a 2-dimensional array
rawdata = pd.read_csv('soybeans.csv')
rawdata = np.array(rawdata)
x = rawdata[:-1, :-1]
iteration = 1	
c = kmodes(x,4)
c.Initialization()
while c.reallocation() > 0:
    print "Iteration: {}".format(iteration) 
    iteration += 1
        
#write the clusters to a csv file
#convert cluster values to int
       
print("Writing clusters to Modes file...")
with open ('Modes.csv','w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([['    ']])
    writer.writerows([['Modes for categorical data:']])
    writer.writerows([['    ']])
    writer.writerows(c.modes)
print("Writing clusters to file...")
with open ('Clusters.csv','w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([c.clustership])
