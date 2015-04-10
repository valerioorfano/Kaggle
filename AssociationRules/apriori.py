from itertools import combinations
from collections import defaultdict, Iterable
import pandas as pd
import numpy as np
import sys

class Apriori:
    def __init__(self,data, minSup=0.01, minConf=0.5):
        self.dataset = data
        self.numTrans = 0
        self.transList = defaultdict(list)
        self.freqList = defaultdict(int)
        self.itemset = set()
        self.highSupportList = list()
        self.numItems = 0
        self.prepData()
        self.F = defaultdict(list)
        self.minSup = minSup
        self.minConf = minConf
        self.H = []
        

    def genAssociations(self):
        candidate = {}	
        self.F[1] = self.firstPass(self.freqList, 1)
        k=2
        while len(self.F[k-1]) != 0:
            candidate[k] = self.candidateGen(self.F[k-1], k)
            for t in self.transList.iteritems():
                for c in candidate[k]:
                    if set(c).issubset(t[1]):
                        self.freqList[c] += 1
            self.F[k] = self.prune(candidate[k], k)
            k += 1
        prod_freq = {}      
        sort_prod_freq = {}              
        for i in range(1,k):
            for j in self.F[i]:
                prod_freq[j] = self.freqList[j]
        sort_prod_freq = sorted(prod_freq.items(), key= lambda x:x[1], reverse=True)
        for key, count in sort_prod_freq:
            print "itemset {} and relative frequency {}".format(key, count)
        print " over a number of {} transactions".format(self.numTrans)    
        return 

	
    def prune(self, items, k):
        f = []
        for item in items:
            count = self.freqList[item]
            support = self.support(count)
            if support >= .95:
                self.highSupportList.append(item)
            elif support >= self.minSup:
                f.append(item)
        return f
	

    def candidateGen(self, items, k):
        candidate = []
        for x in combinations(items, k):
            candidate.append(x)
        return candidate
	
    def genRules(self):
        for i in range(1,len(self.F)+1):
                if i >= 2:
                    for item in self.F[i]:
                        subsets = self.genSubsets(item)
                        for subset in subsets:
                            if len(subset) == 1:
                                subCount = self.freqList[subset[0]]
                            else:
                                subCount = self.freqList[subset]
                            itemCount = self.freqList[item]
                            if subCount != 0:
                                confidence = self.confidence(subCount, itemCount)
                                if confidence >= self.minConf:
                                    support = self.support(self.freqList[item])
                                    rhs = self.difference(item, subset)
                                    if len(rhs) == 1:
                                        self.H.append((subset, rhs, support, confidence))
        for el in self.H:
            print "{0} => {1} support:{2:.2f} confidence:{3:.2f}".format(el[0][0], el[1][0], el[2], el[3])  
        return 
  
 

    def genSubsets(self, item):
        subsets = []
        for i in range(1,len(item)):
            subsets.extend(combinations(item, i))
        return subsets

    def difference(self, item, subset):
        return tuple(x for x in item if x not in subset)
    	
    def confidence(self, subCount, itemCount):
        return float(itemCount)/subCount
    	
    def support(self, count):
        return float(count)/self.numTrans
    	
    def firstPass(self, items, k):
        f = []
        for item, count in items.iteritems():
            support = self.support(count)
            if support == 1:
                self.highSupportList.append(item)
            elif support >= self.minSup:
                f.append(item)   	
        return f


    def prepData(self):
        for i, transaction in enumerate(self.dataset):
            self.transList[i] = transaction
            self.numTrans +=1    
            for item in transaction:
                if not item in self.itemset:
                    self.itemset.add(item)
                    self.numItems +=1
                self.freqList.setdefault(item,0)
                self.freqList[item] +=1 
            
def main():
    
    num_args = len(sys.argv)
    minSup = minConf = 0
    if num_args != 4 :
        print 'Expected input format: python apriori.py <dataset.csv> <minSup> <minConf>'
        return
    else:
        data = []
        with open('/home/tom/Projects/AssociationRules/Apriori/' + str(sys.argv[1]),'r') as file:
            for i, line in enumerate(file):
                line = line[:-3]
                data.append(list(line.split(' ')))
            data = data[:-1]
        minSup = float(sys.argv[2])
        minConf = float(sys.argv[3])
        print "Dataset: ", sys.argv[1], " MinSup: ", minSup, " MinConf: ", minConf
        apriori = Apriori(data,minSup,minConf)        
        apriori.genAssociations()
        apriori.genRules()
        
        
if __name__ == '__main__':
    main()

