from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix



    
def plot_correlations(data):
    """Plot pairwise correlations of features in the given dataset"""

    cols = data.columns.tolist()
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    
    # Plot absolute value of pairwise correlations since we don't
    # particularly care about the direction of the relationship,
    # just the strength of it
    cax = ax.matshow(data.corr().abs())#, cmap=cm.YlOrRd)
    
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticklabels(cols)


def scatter_matr(data):
    scatter_matrix(data, figsize=(10, 10), diagonal='kde')



def plot_grid_search_results2(grids):
    """Plot the results of the grid search, showing which model types were most
    accurate for each response variable"""
    scores = []
#    for model in grids.keys():
    for model in grids.keys():
        
        # For each response and model, determine the CV MSE and add it to
        # a final result for plotting
        for response in grids[model].keys():
            print "model" , model
            print "response", response
            cv_scores = []
            for score in grids[model][response].grid_scores_:   
                vals = np.abs(score.cv_validation_scores)  # one score for each cv 5
                cv_scores.append((np.mean(vals), np.std(vals)))
            cv_scores = pd.DataFrame(cv_scores).sort(0)
            print "cv_scores", cv_scores
            scores.append((response, model, cv_scores.iloc[0,0], cv_scores.iloc[0,1]))
            print "scores", scores            
            
    # Plot the accuracy per model type and response variable
    scores = pd.DataFrame(scores, columns=['response', 'model', 'mean', 'std'])
    fig, ax = plt.subplots(nrows=1, ncols=len(scores['response'].unique()), figsize=(20,4))
    ax = np.ravel(ax)
    for i, (response, data) in enumerate(scores.groupby('response')):
        data = data.set_index('model').sort('mean')
        data['mean'].plot(kind='bar', yerr=data['std'], title=response, ax=ax[i])




def plot_grid_search_results(grids):
    """Plot the results of the grid search, showing which model types were most
    accurate for each response variable"""
    scores = []
#    for model in grids.keys():
    for model in grids.keys():
        # For each response and model, determine the CV MSE and add it to
        # a final result for plotting
        for response in grids[model].keys():
            scores.append((response, model,np.abs(grids[model][response].best_score_)))
    # Plot the accuracy per model type and response variable
    scores = pd.DataFrame(scores, columns=['response', 'model', 'MSE'])
    fig, ax = plt.subplots(nrows=1, ncols=len(scores['response'].unique()), figsize=(20,4))
    ax = np.ravel(ax)
    for i, (response, data) in enumerate(scores.groupby('response')):
        data = data.set_index('model').sort('MSE')
        print "data", data
        data['MSE'].plot(kind='bar',  title=response, ax=ax[i])


def show_importance(regressors):
    importances = {}
    for response in regressors.keys():
        imp = regressors[response].meta_estimator_.coef_
        names = regressors[response].base_estimators_.keys()
        importances[response] = pd.Series(imp, index=sorted(names))
    importances = pd.DataFrame(importances)
    order = importances.max(axis=1).order(ascending=False)
    importances.loc[order.index.values].plot(kind='bar', figsize=(20,3))

