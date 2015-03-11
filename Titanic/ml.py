from sklearn.metrics import make_scorer
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing, grid_search
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from data import *
import os
import numpy as np


models = {'LogisticRegression':LogisticRegression(),'RandomForestClassifier':RandomForestClassifier(),'ExtraTreesClassifier':ExtraTreesClassifier(),'GradientBoostingClassifier':GradientBoostingClassifier()}

PARAM_GRID = {
	'LogisticRegression': {'C': [1.5, 2, 2.5, 3, 3.5, 5, 5.5],
	'class_weight': ['auto']},
	'RandomForestClassifier': {
	'n_jobs': [1], 'max_depth': [15, 20, 25, 30, 35, None],
	'min_samples_split': [1, 3, 5, 7],
	'max_features': [4, 5, 6, 7]
	},
	'ExtraTreesClassifier': {'min_samples_leaf': [2, 3],
	'n_jobs': [1],
	'min_samples_split': [1, 2, 5],
	'bootstrap': [False],
	'max_depth': [15, 20, 25, 30],
	'max_features': [4, 5, 6, 7]},
	'GradientBoostingClassifier': { 'max_features': [4, 5, 6, 7],
	'learning_rate': [.05, .08, .1],
	'max_depth': [8, 10, 13]},
	}
 
def learn(X, y, cv_tries):
    if os.path.exists("saved_params.p"):
        clfs = load_models()
    else:    
        clfs = {}
        for name in models.keys():
            grid = PARAM_GRID[name] 
            print 'Running grid search for model = {}, X shape = {}'.format(name, X.shape)
            clf = GridSearchCV(models[name], grid, cv=cv_tries, scoring='accuracy')
            # Fit the model to the grid and return an object containing the best paramaters
            clfs[name] = clf.fit(X, y)
            print "INFO - {} score = {}, params = {}" .format(name, clf.best_score_, clf.best_params_)
            save_models(clfs)        
    return clfs

 
def metrica(X,y,models, cv_tries):
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    for name in models.keys():
        print name
        clf = models[name]
        scores = cross_validation.cross_val_score(clf, X, y, cv=cv_tries, scoring = 'accuracy',)
        print "Accuracy: {} +/- {}".format(np.mean(scores), np.std(scores))
        
        





 