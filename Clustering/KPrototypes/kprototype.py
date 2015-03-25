import numpy as np
import random

class KPrototypes():
	
    def __init__(self,x,k=2,gamma = None):
        """k-protoypes clustering algorithm for mixed numeric and categorical data.
        Huang, Z.: Clustering large data sets with mixed numeric and categorical values,
        Proceedings of the First Pacific Asia Knowledge Discovery and Data Mining Conference,
        Singapore, pp. 21-34, 1997.
	
        Inputs: k = number of clusters
        Attributes: clusters = cluster numbers [no. points]
        centroids = for numeric atributes
        modes = for categorical attributes
        clustership = clustership matrix [k * no. points]
        distance = clustering cost, defined as the sum distance of all points to their respective clusters
        gamma = weighing factor that determines relative importance of num./cat. attributes (see discussion in Huang [1997])
        
        """
       self.k = k
       self.xnum = np.asanyarray(x[0])
       self.xcat = np.asanyarray(x[1])
       self.nnumpoints, self.nnumattrs = self.xnum.shape
       self.ncatpoints, self.ncatattrs = self.xcat.shape
       self.modes = [['' for x in range(self.ncatattrs)] for y in range(self.k)]
       self.centroids = [[[-99 for x in range(self.nnumattrs)] for y in range(self.k)]]       
       self.clustership = [-1 for y in range(self.nnumpoints)]
       self.clustercount = [0 for y in range(self.k)]
       self.clusterfrequency = [[{} for x in range(self.ncatattrs)] for y in range(self.k)]
       self.sumincluster = np.zeros((self.k, self.nnumattrs), dtype='float')
       assert self.nnumpoints == self.ncatpoints, "More numerical points than categorical?"
       assert self.k < self.nnumpoints, "More clusters than data points?"
       self.gamma = 2
    
    def Initialization(self,verbose=1):
        if verbose:
            print("Init: initializing centroids")
            
        initialcat = random.sample(range(self.ncatpoints),self.k)                
        self.modes = self.xcat[initialcat]
                
        initialnum = random.sample(range(self.nnumpoints),self.k)            
        self.centroids = self.xnum[initialnum]
        #self.centroids = [np.mean(self.xnum, axis=0) + np.random.randn(self.k, self.nnumattrs) * np.std(self.xnum, axis=0)]
	
        if verbose:
            print("Init: initializing clusters")
        
        for i in range(self.nnumpoints):
            # initial assigns to clusters
            cluster = 0
            MinDistance = self.Distance(self.centroids[0], self.xnum[i]) + self.gamma * self.Sigma(self.modes[0], self.xcat[i])
            for z in range(self.k):
                distance = self.Distance(self.centroids[z], self.xnum[i]) + self.gamma * self.Sigma(self.modes[z], self.xcat[i])
                if distance < MinDistance:
                    MinDistance = distance
                    cluster = z
            self.clustership[i] = cluster
            self.clustercount[cluster] += 1
            for j in range(self.nnumattrs):
                self.sumincluster[cluster,j] += self.xnum[i,j]
                if (self.clustercount[cluster] != 0):  
                    self.centroids[cluster,j] = self.sumincluster[cluster,j] / self.clustercount[cluster]   
            for j in range(self.ncatattrs):
                val = self.xcat[i,j]
                if val in self.clusterfrequency[cluster][j].keys():
                    self.clusterfrequency[cluster][j][val] += 1
                else:
                    self.clusterfrequency[cluster][j][val] = 1
                self.modes[cluster][j] = self.HighestFrequency(cluster,j)   


    def reallocation(self, verbose=1):
        if verbose:
            print("Reallocation process started ...")

        moves = 0
        for i in range(self.nnumpoints):
            cluster = self.ClosestCluster(i)[0]
            if self.clustership[i] != cluster:
                moves += 1
                oldcluster = self.clustership[i]
                self.clustership[i] = cluster
                self.clustercount[cluster] += 1
                self.clustercount[oldcluster] -= 1
                for j in range(self.nnumattrs):
                    self.sumincluster[cluster,j] += self.xnum[i,j]
                    self.sumincluster[oldcluster,j] -= self.xnum[i,j]
                    if (self.clustercount[cluster] != 0):  
                        self.centroids[cluster,j] = self.sumincluster[cluster,j]/self.clustercount[cluster]                  
                    if (self.clustercount[oldcluster] != 0):      
                        self.centroids[oldcluster,j] = self.sumincluster[oldcluster,j]/self.clustercount[oldcluster]
                for j in range(self.ncatattrs):
                     val = self.xcat[i,j]
                     if val in self.clusterfrequency[cluster][j].keys():
                         self.clusterfrequency[cluster][j][val] += 1
                     else:
                         self.clusterfrequency[cluster][j][val] = 1
                     self.clusterfrequency[oldcluster][j][val] -= 1
                     self.modes[cluster][j] = self.HighestFrequency(cluster,j)
                     self.modes[oldcluster][j] = self.HighestFrequency(oldcluster,j)
        return moves


    def ClosestCluster(self,i):
        cluster = 0
        mindistance = self.Distance(self.centroids[0], self.xnum[i]) + self.gamma * self.Sigma(self.modes[0], self.xcat[i])
        for z in range(self.k):
            distance = self.Distance(self.centroids[z], self.xnum[i]) + self.gamma * self.Sigma(self.modes[z], self.xcat[i])
            if distance < mindistance:
                mindistance = distance
                cluster = z
        return [cluster, mindistance]


    def Distance(self,anum, b):
        # Euclidean distance
        return np.sum((anum - b) ** 2)


    def Sigma(self,a, b):
        # simple matching dissimilarity
        return sum(a != b)
	 
    def HighestFrequency(self,c,e):
        keys = [key for key,val in self.clusterfrequency[c][e].iteritems() if val == max(self.clusterfrequency[c][e].values())]
        mode = keys[0]
        return mode
