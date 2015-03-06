from numpy import array, hstack, vstack
from sklearn import metrics, cross_validation, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import naive_bayes
from sklearn import preprocessing, grid_search
from scipy import sparse
from itertools import combinations
	
from sets import Set
import numpy as np
import pandas as pd
import sys
import os.path

def group_data(data,degree=3,hash=hash):
    new_data = []
    m,n = data.shape
    for indicies in combinations(range(n), degree):
        new_data.append([hash(tuple(row)) for row in data[:,indicies]])
    return np.array(new_data).T


def OneHotEncoder(data, keymap=None):
	"""
	OneHotEncoder takes data matrix with categorical columns and
	converts it to a sparse binary matrix.
	
	Returns sparse binary matrix and keymap mapping categories to indicies.
	If a keymap is supplied on input it will be used instead of creating one
	and any categories appearing in the data that are not in the keymap are
	ignored
	"""
	if keymap is None:
         keymap = []
         for col in data.T:
             uniques = set(list(col))
             keymap.append(dict((key, i) for i, key in enumerate(uniques)))
	total_pts = data.shape[0]
	outdat = []
	for i, col in enumerate(data.T):
         km = keymap[i]
         num_labels = len(km)
         spmat = sparse.lil_matrix((total_pts, num_labels))
         for j, val in enumerate(col):
             if val in km:
                 spmat[j, km[val]] = 1
         outdat.append(spmat)
	outdat = sparse.hstack(outdat).tocsr()
	return outdat, keymap

def optimize_hyperparameters(X_tr, y_tr, model, param_grid, cv_tries):
    print "Performing hyperparameter optimization..."
    optimal_model = grid_search.GridSearchCV(model, param_grid, cv=cv_tries, n_jobs=-1)
    optimal_model.fit(X_tr, y_tr)
    print "Selected C: ", optimal_model.best_estimator_.C
    return optimal_model


def save_results(features, filename):
    print "Save results in txt format"
    f = open(filename, 'w')
    f.write("Features List \n")
    for i in range(len(features)):    
        f.write("%i\n" % features[i])

def save_model(model, filename):
    print "Save model in txt format"
    f = open(filename, 'w')
    f.write("Model \n")
    f.write("%f\n" % model)


def create_test_submission(filename, prediction):
    content = ['id,ACTION']
    for i, p in enumerate(prediction):
        content.append('%i,%f' %(i+1,p))
    f = open(filename, 'w')
    f.write('\n'.join(content))
    f.close()
    print 'Saved'


def cross_validate(X, y, model, n_tries):
    mean_auc = 0.0
    for i in range(n_tries):
        X_tr, X_te, y_tr, y_te = cross_validation.train_test_split(X, y, test_size=1.0/float(n_tries),random_state = i*1234)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        roc_auc = metrics.roc_auc_score(y_te, preds)
        mean_auc += roc_auc
    return (mean_auc/n_tries)

    
def greedy_forward_feature_selection(Xts, y, model, cv_tries):
    print "Performing greedy forward feature selection..."
    selected_features = set()
    features_not_selected = range(len(Xts))
    selection_scores = [-1.0,0.0]
    while ((selection_scores[-2] < selection_scores[-1]) and (len(features_not_selected)>1)): 
        # print"len(features_not_selected)", len(features_not_selected)        
        feature_scores = []
        for f in features_not_selected:
            proposed_new_feature_set = list(selected_features) + [f]
            X_projected = sparse.hstack([Xts[j] for j in proposed_new_feature_set]).tocsr()
            score = cross_validate(X_projected, y, model, cv_tries)
            #print "  Adding feature ", f, " gives score ", score, " (it was ", selection_scores[-1] , ")"
            feature_scores.append((score, f))
        best_score, best_feature = sorted(feature_scores)[-1]
        selected_features.add(best_feature)
        print "Current feature selection: ", list(selected_features), " with score ", best_score
        features_not_selected.remove(best_feature)
        selection_scores.append(best_score)
    if (len(features_not_selected) == 0):    
        selected_features.remove(sorted(a)[-1])
    return list(selected_features)
    
    
    
def learn(X, y, model, hyperparameter_grid, cv_tries):
    if os.path.exists("selected_features.txt"):
        print "File selected_features exist"
        selected_features = np.loadtxt("selected_features.txt", skiprows=1,dtype=int)
    else:    
        selected_features = greedy_forward_feature_selection(X, y, model, cv_tries)
        save_results(selected_features, "selected_features.txt")
    #    print "selected_features", selected_features    
    projected_X = sparse.hstack([X[j] for j in selected_features]).tocsr()
    optimal_model = optimize_hyperparameters(projected_X, y, model, hyperparameter_grid, cv_tries)
    optimal_model_score = cross_validate(projected_X, y, model,cv_tries)
    print "Optimal model score on training data: ", optimal_model_score
    return optimal_model, selected_features
    
def load_data():    
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    alldata = np.vstack((train_data.ix[:,1:-1],test_data.ix[:,1:-1]))
    num_train = train_data.shape[0]
    alldata = alldata[:,[0,1,4,5,6,7]]   #removing Role_Code1 & 2
    print "Transforming all the categorical data into numeric"
    relabler = preprocessing.LabelEncoder()
    for col in range(len(alldata[0,:])):
        alldata[:,col] = relabler.fit_transform(alldata[:,col])
    dp = group_data(alldata,degree = 2)
    for col in range(len(dp[0,:])):
        dp[:,col] = relabler.fit_transform(dp[:,col])
    dt = group_data(alldata,degree = 3)
    for col in range(len(dt[0,:])):
        dt[:,col] = relabler.fit_transform(dt[:,col])
    #Train data
    y = np.array(train_data.ACTION)
    X = alldata[0:num_train]
    X_2 = dp[0:num_train]
    X_3 = dt[0:num_train]
    X_train_all = np.hstack((X,X_2,X_3))
    #Test data
    X_test = alldata[num_train:]
    X_test_2 = dp[num_train:]
    X_test_3 = dt[num_train:]
    X_test_all = np.hstack((X_test,X_test_2,X_test_3))
    #    num_features = X_train_all.shape[1]
    return X_train_all,X_test_all,y



#Main
X,X_test,y = load_data()
num_features = X.shape[1]
model = linear_model.LogisticRegression()



X_enc = [OneHotEncoder(X[:,[i]])[0] for i in range(num_features)]
hyper_parameters = {'C': [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]}
n_tries = 10
optimal_model, good_features = learn(X_enc, y, model,hyper_parameters,n_tries)
        
#encoder.fit(vstack((X[:,good_features],X_test[:,good_features])))
#X_train_final = encoder.transform(X[:,good_features])
#X_test_final = encoder.transform(X_test_all[:,good_features])

print "Performing One Hot Encoding on entire dataset..."
Xt = np.vstack((X[:,good_features], X_test[:,good_features]))
Xt, keymap = OneHotEncoder(Xt)
X_train = Xt[:num_train]
X_test = Xt[num_train:]

print "Training the full best model"         
print "Making predictions and saving results"
optimal_model.fit(X_train,y)
preds = optimal_model.predict(X_test)
create_test_submission('Predictions.txt', preds)
