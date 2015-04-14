"""Collaborative Filtering:
I like what you like. 1 step is finding similar people by means of distance (manhattan or euclide)"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import manhattan_distances as md
from sklearn.metrics.pairwise import euclidean_distances as ed
from operator import itemgetter
from collections import defaultdict

data = pd.read_csv('beer_reviews2.csv', header=0, sep=',')

class CF:
    def __init__(self,data,dist):
        self.dataset = data
        self.dist = dist
        self.cols_rating = ['review_overall', 'review_aroma', 'review_palate', 'review_taste']

    """Distance between 2 users"""
    def distance(self, user1, user2):
        beers1 = self.dataset[self.dataset.review_profilename == user1].beer_name.unique()
        beers2 = self.dataset[self.dataset.review_profilename == user2].beer_name.unique()
        beers_intersect = [x for x in beers1 if x in beers2]
        beers = self.dataset[self.dataset.review_profilename.isin([user1,user2]) & self.dataset.beer_name.isin(beers_intersect)]
        #cols = ['review_overall', 'review_aroma', 'review_palate', 'review_taste']
        if beers_intersect == []:
            #no ntersection
            pass
        sum = 0    
        for x in beers_intersect:
            lista1 = beers.ix[(beers.review_profilename == user1) & (beers.beer_name == x) , self.cols_rating].values
            lista2 = beers.ix[(beers.review_profilename == user2) & (beers.beer_name == x) , self.cols_rating].values
            sum += self.dist(lista1,lista2)[0][0]
        return [round(sum,2), len(beers_intersect)]
    

    def nearest_neighbour(self,username):
        nearest = defaultdict(list)     
        for x in self.dataset.review_profilename.unique():
            if x != username:            
                nearest[x] = self.distance(username, x)                         
        nearest_sort = sorted(nearest.items(), key = lambda (k,v): v[1], reverse = True)
        max_commons = nearest_sort[0][1][1]
        #we get the element with highest number of common evaluations              
        nearest_max_commons = [(k,v) for (k,v) in nearest.items() if v[1] == max_commons]
        nearest_max_comm_sort = sorted(nearest_max_commons,key = lambda (k,v):v[0])
        nearest = nearest_max_comm_sort[0][0]
        return nearest        
        #we select the records with the highest number of common evaluations and among those that with the lowest distance       
        
                    

    def recommend(self, username):
        """List of recomendations:
        1 Find the nearest neighbour
        2 Find the beers evaluated by the neighbour but not by username 
        3 Return the list of recommendations """
        recommendations = defaultdict(list)        
        nearest = self.nearest_neighbour(username)        
        beers_username = self.dataset[self.dataset.review_profilename == username].beer_name.unique()        
        beers_nearest = self.dataset[self.dataset.review_profilename == nearest].beer_name.unique()                
        beers_nearest_less_uname = [x for x in beers_nearest if x not in beers_username] 
        for b_name in beers_nearest_less_uname:
            recommendations[b_name] = self.dataset.ix[(self.dataset.review_profilename == nearest) & (self.dataset.beer_name == b_name),self.cols_rating].values     
        df,rows = [],[]
        for k,v in recommendations.items():
            df.append(v[0].tolist())
            rows.append(k)
        df = pd.DataFrame(df)
        df.index = rows
        df.columns = self.cols_rating
        print df
        return recommendations




cf = CF(data,ed)
#aa = cf.distance('stcules','seanyfo')
#nn = cf.nearest_neighbour('stcules')
aa = cf.recommend('stcules')