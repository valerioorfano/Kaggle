import random
import copy 
import numpy as np
np.set_printoptions(threshold='nan')

class KModes:
	
    def __init__(self,z,k=8,verbose=0):
        self.data = z
        self.k = k
        self.verbose = verbose
        self.numobjects = z.shape[0]
        self.numattributes = z.shape[1]
        self.clustervalues = [['' for x in range(self.numattributes)] for y in range(self.k)]
        self.clustership = [-1 for y in range(self.numobjects)]
        self.clustercount = [0 for y in range(self.k)]
        self.clusterfrequency = [[{} for x in range(self.numattributes)] for y in range(self.k)]
    	
        assert self.k < self.numobjects, "More clusters than data points?"
	
    def BuildInitialClusters(self):
        # Choose random data values for the k clusters
        for i in range(self.k):
            rand = random.randrange(self.numobjects)
            self.clustervalues[i] = self.data[rand]        
        totaldistance = 0
        for i in range(self.numobjects):
            closest = self.ClosestCluster(i)
            cluster = closest[0]
            totaldistance += closest[1]
            self.clustership[i] = cluster
            self.clustercount[cluster] += 1
            for j in range(self.numattributes):
                val = self.data[i,j]
                if val in self.clusterfrequency[cluster][j].keys():
                    self.clusterfrequency[cluster][j][val] += 1
                else:
                    self.clusterfrequency[cluster][j][val] = 1

        
        
        
        # DEBUG CODE
        if self.verbose > 0:
            for i in range(self.k):
                print("Initial modes are: Cluster {} -> Mode {}".format(i, self.clustervalues[i]))
            print("Initial cluster counts are: ")
            for i in range(self.k):
                print("\t{} => {}".format(i,self.clustercount[i]))
            print("Clustership: ")
            for i in range(self.numobjects):
                print("row: {} is in cluster {}".format(i,self.clustership[i]))
        # END DEBUG CODE
    	
        # return total cost and initial clusters built
        return [totaldistance, self.clustervalues]
	
    def BuildClusters(self):
        moves = 0
        # New Modes Centroids. We take the value most frequent
        for i in range(self.k):
            for j in range(self.numattributes):
                self.clustervalues[i][j] = self.HighestFrequency(i,j)
        #Then  we generate again new clusters    
        for i in range(self.numobjects):
            cluster = self.ClosestCluster(i)[0]
            if self.clustership[i] != cluster:
                moves += 1
                oldcluster = self.clustership[i]
                self.clustership[i] = cluster
                self.clustercount[cluster] += 1
                self.clustercount[oldcluster] -= 1
                for j in range(self.numattributes):
                     val = self.data[i,j]
                     if val in self.clusterfrequency[cluster][j].keys():
                         self.clusterfrequency[cluster][j][val] += 1
                     else:
                         self.clusterfrequency[cluster][j][val] = 1
                     self.clusterfrequency[oldcluster][j][val] -= 1

        # DEBUG CODE
        if self.verbose > 0:
            for i in range(self.k):
                print("New Modes are: Cluster {} -> Mode {}".format(i, self.clustervalues[i]))
            print("new cluster counts are: ")
            for i in range(self.k):
                print("\t{} => {}".format(i,self.clustercount[i]))
            print("New Clustership: ")
            for i in range(self.numobjects):
                print("row: {} is in cluster {}".format(i,self.clustership[i]))
        # END DEBUG CODE
        return moves

	
    def ClosestCluster(self,o):
        cluster = 0
        mindistance = self.Distance(o,0)
        for i in range(self.k):
            distance = self.Distance(o,i)
            if distance < mindistance:
                mindistance = distance
                cluster = i
        return [cluster, mindistance]
    
    #This is used to calcualte the dissimilarity matrix    
    def Distance(self,o,c):
        dist = 0
        for i in range(self.numattributes):
            if self.data[o,i] != self.clustervalues[c][i]:
                dist += 1
        return dist
	
    def HighestFrequency(self,c,e):
        keys = [key for key,val in self.clusterfrequency[c][e].iteritems() if val == max(self.clusterfrequency[c][e].values())]
        mode = keys[0]
        return mode