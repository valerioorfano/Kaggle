from matplotlib.pyplot import show
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram,fcluster,fclusterdata,centroid,leaves_list
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import numpy
import random
import sys


#Input: z= linkage matrix, treshold = the treshold to split, n=distance matrix size


def clustering(data):
    thres = 25
    #Create the distance matrix for the array of sample vectors.
    #Look up 'squareform' if you want to submit your own distance matrices as they need to be translated into reduced matrices
    reduced_data = PCA(n_components=2).fit_transform(data)
    data = pd.DataFrame.as_matrix(data)    
    data_dist = pdist(data,metric='euclidean') # computing the distance
    Y = linkage(data_dist, method='complete')    
    fig = plt.figure(figsize=(8,8))
    # x ywidth height
    ax1 = fig.add_axes([0.05,0.1,0.2,0.6])
    Z1 = dendrogram(Y, orientation='right')
    ax1.set_xticks([])
    
    # Compute and plot second dendrogram.
    ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
    Z2 = dendrogram(Y)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    #Compute and plot the heatmap
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    D = squareform(data_dist)
    D = D[idx1,:]
    D = D[:,idx2]
    im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=plt.cm.RdYlGn)
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    # Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
    plt.colorbar(im, cax=axcolor) 
    
    #From the heatmap it is evident there are about 4 clusters.    
    thres = 28
    plt.figure()
    dendrogram(Y, color_threshold=thres, show_leaf_counts=True)
    clusters1 = fcluster(Y, t=thres,criterion='distance')     
    col = np.array(clusters1)
    col = col/float(np.max(col))
    col = np.array([round(x,2) for x in col])
    print col
    plt.scatter(reduced_data[:,0], reduced_data[:,1], c = col)
    plt.show()
    print "cluster1", clusters1
    print "leaves", leaves_list(Y)    
    return
    