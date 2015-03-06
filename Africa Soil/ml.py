from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.linear_model import ElasticNet,ElasticNetCV, Lasso, Ridge, SGDRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures 
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import Nystroem 
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split,ShuffleSplit 
from sklearn.metrics import mean_squared_error, make_scorer
import random
import os
from utils import *
from sklearn.base import BaseEstimator, RegressorMixin
import copy
from sklearn.linear_model import ElasticNetCV

def get_clfs():
    return {
        'gbr' : { 
            'est' : Pipeline([ 
                ('clf', GradientBoostingRegressor()) 
            ]),
            'grid' : {
                'clf__loss' :['ls', 'huber'],
                'clf__n_estimators' : [100],    
                'clf__learning_rate': [.1, .01],
                'clf__max_depth' : [1, 5, 9],
                'clf__min_samples_leaf' : [3, 5],
                'clf__max_features': [1.0, .3, .1],
                'clf__subsample': [1.0]
            }
        },
        'svr' : { 
            'est' :  Pipeline([ 
                ('scale', StandardScaler()), 
                ('clf', SVR()) 
            ]), 
            'grid' : { 
                'clf__kernel' : ['rbf'], 
                'clf__gamma' :  [.1, .01, 0.001, 0.0001, 0.00001], 
                'clf__C' : [1, 10, 100, 1000] 
            }
        },
        'ridge' : { 
            'est' : Pipeline([ 
                ('scale', StandardScaler()),
                ('poly', PolynomialFeatures()), 
                ('clf', Ridge()) 
            ]),
            'grid' : {
                'clf__alpha' : [.1, 1.0, 10, 100, 500, 1000],
                'clf__normalize' : [False],
                'clf__fit_intercept' : [False],
                'clf__max_iter' : [15000]
            }
        },
        'lasso' : { 
            'est' : Pipeline([ 
                ('scale', StandardScaler()),
                ('poly', PolynomialFeatures()), 
                ('clf', Lasso()) 
            ]),
            'grid' : {
                'clf__alpha' : [.01, .1, 1.0, 10, 100],
                'clf__normalize' : [False],
                'clf__fit_intercept' : [False],
                'clf__max_iter' : [15000]
            }
        },
        'sgd' : { 
            'est' : Pipeline([ 
                ('scale', StandardScaler()),
                ('clf', SGDRegressor()) 
            ]),
            'grid' : {
                'clf__alpha' : [.0001, .00001, .001],
                'clf__l1_ratio' : [1.0, .75, .5, .25, .1],
                'clf__penalty' : ['elasticnet', 'l2'],
                'clf__fit_intercept' : [False]
            }
        }
      }      

'''
'svr_nyst' : { 
            'est' :  Pipeline([ 
                ('scale', StandardScaler()),
                ('nyst', Nystroem()),
                ('clf', SVR()) 
            ]), 
            'grid' : { 
                'nyst__n_components' : [250, 500, 600],
                'nyst__gamma' : [None],
                'clf__kernel' : ['linear'], 
                'clf__C' : [1, 10, 100, 1000] 
            }
        },
        'knn' : { 
            'est' :  Pipeline([ 
                ('scale', StandardScaler()), 
                ('clf', KNeighborsRegressor()) 
            ]), 
            'grid' : {
                'clf__n_neighbors' : [3, 4, 5, 8, 10, 25, 50],
                'clf__weights' : ['distance', 'uniform'],
                'clf__p' : [1, 2, 3]
            }
        },
        'elasticnet' : { 
            'est' : Pipeline([ 
                ('scale', StandardScaler()),
                ('poly', PolynomialFeatures()), 
                ('clf', ElasticNet()) 
            ]),
            'grid' : {
                'clf__alpha' : [.01, .1, 1.0, 10, 100],
                'clf__l1_ratio' : [1.0, .75, .5, .25, .1],
                'clf__normalize' : [False],
                'clf__fit_intercept' : [False],
                'clf__max_iter' : [15000]
            }
        },

'''

def grid_search2(X, Y, clfs, sample_rate = 1.0, k=5):
    """Determine the optimal tuning parameters for each base learner, returned
    as the result of a GridSearchCV fit"""
    if os.path.exists("saved_params.p"):
        grids = load_params()
    else:    
        # Sample the given data to increase runtime performance
        ix = random.sample(range(len(X)), int(len(X) * sample_rate))
        print len(X), len(ix)
        X, Y = X.iloc[ix], Y.iloc[ix]
        
        # Grid search over the parameters for each model type to determine the best setttings
        grids = {}
        for name, clf in clfs.iteritems():
            grids[name] = {}
            for response in Y.columns.tolist():
                X_train, X_test, y_train, y_test = train_test_split(X, Y[response], test_size=.2)
                print 'Running grid search for response = {}, model = {}, X shape = {}'.format(response, name, X_train.shape)
                
                clf = GridSearchCV(clfs[name]['est'], clfs[name]['grid'], 
                                   cv=k, scoring=make_scorer(mean_squared_error, greater_is_better=False))
                                   
                # Fit the model to the grid and return an object containing the best paramaters
                grids[name][response] = clf.fit(X, Y[response])
                print "INFO - {}:{} score = {}, params = {}" .format(name, response, clf.best_score_, clf.best_params_)
        save_params(grids)        
    return grids
    

def grid_search(X, Y, clfs, sample_rate = 1.0, k=5):
    """Determine the optimal tuning parameters for each base learner, returned
    as the result of a GridSearchCV fit"""
    if os.path.exists("saved_params.p"):
        grids = load_params()
    else:    
        # Sample the given data to increase runtime performance
        ix = random.sample(range(len(X)), int(len(X) * sample_rate))
        print len(X), len(ix)
        X, Y = X.iloc[ix], Y.iloc[ix]
        
        # Grid search over the parameters for each model type to determine the best setttings
        grids = {}
        for name in clfs.keys():
            grids[name] = {}
            for response in Y.columns.tolist():
                X_train, X_test, y_train, y_test = train_test_split(X, Y[response], test_size=.2)
                print 'Running grid search for response = {}, model = {}, X shape = {}'.format(response, name, X_train.shape)
                
                clf = GridSearchCV(clfs[name]['est'], clfs[name]['grid'], 
                                   cv=k, scoring=make_scorer(mean_squared_error, greater_is_better=False))
                                   
                # Fit the model to the grid and return an object containing the best paramaters
                grids[name][response] = clf.fit(X, Y[response])
                print "INFO - {}:{} score = {}, params = {}" .format(name, response, clf.best_score_, clf.best_params_)
        save_params(grids)        
    return grids


def get_models2():
    """Return models previously defined with parameters 
    tuned as shown to provide the best CV scores on the data in question"""
    
    # Fetch best parameters for each model
    gs_params = load_params()

    # Create a dictionary of response -> model sets where the different models
    # for each response are optimally tuned
    clfs = get_clfs()
    models = {}
    for model in gs_params.keys():
        for response in gs_params[model].keys():
            
            if response not in models.keys():
                models[response] = {}
                
            # Clone the estimator and tune it appropriately for this response variable
            clf = cp.deepcopy(clfs[model]['est'])
            params = gs_params[model][response].best_params_
            print 'Setting params for response = {}, model = {} as {}'.format(response, model, params)
            clf.set_params(**params)
            
            models[response][model] = clf
    return models




def get_models(gs_params=None):
    """Return models previously defined with parameters 
    tuned as shown to provide the best CV scores on the data in question"""
    
    # Fetch best parameters for each model
    if gs_params == None:
        gs_params = load_params()
        
    models = {}
    for model in gs_params.keys():
        for response in gs_params[model].keys():
            
            if response not in models.keys():
                models[response] = {}
                
            # Clone the estimator and tune it appropriately for this response variable
            models[response][model] = gs_params[model][response].best_estimator_
    return models





class StackRegressor(BaseEstimator, RegressorMixin):
    """'Stack Generalization' implementation using predictions from base
    learners in cross validation as inputs to a (simple) meta-learner"""
    def __init__(self, base_estimators, meta_estimator, k=10):
        """Stack regression requires a set of 'base_estimators' used to make predictions
        for a 'meta_estimator' through the output of 'k' CV rounds.  The meta_estimator
        is then used to make final predictions"""
        self.base_estimators_ = base_estimators
        # Initialize a dictionary that will contain the base estimators used during each fold
        # (the averages of predictions from each fold regressor will be used as inputs 
        # to the meta learner)
        self.fold_estimators_ = dict([ (model, []) for model in base_estimators.keys() ])
        self.meta_estimator_ = meta_estimator
        self.k_ = k

    def _get_meta_train(self, X, y):
        """Train the base estimators using CV and use the predictions on test
        data in each fold to produce an input dataset to the meta learner"""
        # Initialize parts of meta learner dataset
        X_meta = None
        y_meta = None
        # For each fold, train the base estimators on training data, and apply
        # those estimators to test data to produce a piece of the meta learner 
        # training set
        for i, (train, test) in enumerate(ShuffleSplit(len(X), self.k_)):
            print 'Training stack base estimators, fold {} of {}'.format(i+1, self.k_)
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
            X_train_meta = pd.DataFrame()
            for model in sorted(self.base_estimators_.keys()):
                # Copy, then fit the base estimator since it will be used multiple times
                clf = copy.deepcopy(self.base_estimators_[model])
                clf.fit(X_train, y_train)
                # Assign the predictions from this leaner as inputs to the meta learner
                X_train_meta[model] = clf.predict(X_test)
                # Save the base learner for later use during predictions
                self.fold_estimators_[model].append(clf)
            # Add the resuling training data to the result
            X_meta = np.vstack((X_meta, X_train_meta)) if X_meta is not None else X_train_meta
            y_meta = np.concatenate((y_meta, y_test)) if y_meta is not None else y_test
        # Return the final meta learner training set
        return X_meta, y_meta
            
    def _train_meta(self, X, y):
        """Training the meta learner on unbiased base learner predictions"""
        self.meta_estimator_.fit(X, y)

    def fit(self, X, y):
        """Train the stack regressor through an initial CV round for
        base learners followed by a single fit for the meta learner"""
        # Cast to numpy array, if pandas series/data frame
        X, y = np.array(X), np.array(y)
        # Get the meta learner training data
        X_meta, y_meta = self._get_meta_train(X, y)
        # Fit the meta learner, effectively determining which base
        # learner predictions are most useful
        self.meta_estimator_.fit(X_meta, y_meta)
        
    def predict(self, X):
        """Predict unseen data using averages of predictions from base learners
        as inputs to the meta learner"""
        X_meta = pd.DataFrame()
        for model in sorted(self.base_estimators_.keys()):
            # Get predicted vectors from each fold estimator of this model type
            pred = [clf.predict(X) for clf in self.fold_estimators_[model] ]
            print "pred", pred
            # Use average predicted value from each fold as input to meta learner
            X_meta[model] = np.mean(np.transpose(np.array(pred)), axis=1)
        # Return the predictions of the meta learner
        return self.meta_estimator_.predict(X_meta)



def predict_response(X_train, X_test, Y_train, models, responses):    
    predictions = {}
    regressors = {}
    for response in responses:
        print 'Running predictions for response {}'.format(response)
        # Create stack regressor using regularized linear meta learner with non-negative input attributions
        clf = StackRegressor(models[response], ElasticNetCV(positive=True), k=10)
        # Fit stack regressor and add fit regressor as well as its predictions to the results
        clf.fit(X_train, Y_train[response])
        predictions[response] = clf.predict(X_test)
        regressors[response] = clf
    
    return predictions, regressors
