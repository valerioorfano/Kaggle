from sklearn import decomposition
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn import decomposition
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cross_validation import cross_val_score
import json
import cPickle as pickle
from sklearn.metrics import mean_squared_error
from pandas.tools.plotting import scatter_matrix

def redux_features(dataset, features, prefix, n_components=.99, show_exp_variance=False, scale=False, pca_transform=None):

    data = dataset.data.copy()
    
    # Print the features being decomposed
    print 'Running PCA for features decomposition {}'.format(features[:3] + ["...."] + features[-2:])
    
    # Run decomposition
    # Data are already scaled
    if scale:
        X = scale(dataset.X()[features])
    else:
        X = dataset.X()[features]
    n_cols = X.shape[1]
    if pca_transform:
        pca = pca_transform
    else:
        pca = decomposition.PCA(n_components=n_components).fit(X)

    X = pca.transform(X)
    reduced_n_cols = X.shape[1]
    reduced_features = [prefix + str(i) for i in range(reduced_n_cols)]
    
    # Print a summary of the decomposition explaining to what extent the given dimensions were reduced
    print '{} features reduced to {} primary components (with feature names prefixed by "{}")'.format(n_cols, reduced_n_cols, prefix)
    if show_exp_variance:
        vals = pca.explained_variance_ratio_[:reduced_n_cols]
        idx = reduced_features
        pd.Series(vals, index=idx).plot(kind='bar', figsize=(12,4), title='Explained Variance')
    
    # Replace the original features with the transformed features of lower dimensional rank
    ix = dataset.data.index
    X = pd.DataFrame(X, columns = reduced_features, index = ix )
    data = pd.concat([
        dataset.data.drop(features, axis=1).set_index(ix), 
        X
    ], axis=1, ignore_index=False)
    
    dataset.data = data
    dataset.cols = data.columns.tolist()
    return dataset, pca



def get_feature_importances(dataset):
    """Return a data frame containing the relative importances of each feature for each dependent variable"""
    res = {}
    for response in dataset.Y():
        X, y = dataset.X(), dataset.data[response]
        clf = ExtraTreesRegressor(n_estimators=100).fit(X, y)   
        res[response] = pd.Series(clf.feature_importances_, index=X.columns.tolist())
    feature_importances = pd.DataFrame(res)
    order = feature_importances.max(axis=1).order(ascending=False)
    feature_importances.loc[order.index.values].plot(kind='bar', figsize=(20,3))


def save_params(saved_params):
    with open('saved_params.p', 'wb') as fp:
        pickle.dump(saved_params, fp)


def load_params(filename='saved_params.p'):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)
        

def metrica(predictions, test_dataset):
    final_scores = {}
    for response in predictions.keys():
        final_scores[response] = mean_squared_error(test_dataset.Y()[response], predictions[response])

    print '\nMSE scores by response:'
    print pd.Series(final_scores)

    print '\nRMSE scores by response:'
    print pd.Series(final_scores).apply(np.sqrt)

    print '\nMCRMSE (RMSE average over all response variables) score:'
    print pd.Series(final_scores).apply(np.sqrt).mean()




def save_predictions(predictions, prediction_index):
    """Save final predictions, after setting the index (i.e. PIDN per record), as a csv file for submission"""
    
    final_results = pd.DataFrame(predictions)
    final_results.index = prediction_index
    final_results.index.name = 'PIDN'
    
    # Save the "final answer" in a file that can be sent directly to Kaggle, with response predictions in the correct order
    final_results[['Ca', 'P', 'pH', 'SOC', 'Sand']].to_csv('predictions.csv', index=True)
    
    # Plot a scatter matrix of the predictions for comparison to earlier data exploration
    scatter_matrix(final_results, diagonal='kde', figsize=(8,8))


