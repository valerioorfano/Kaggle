{\rtf1\ansi\ansicpg1252\cocoartf1265\cocoasubrtf200
{\fonttbl\f0\fnil\fcharset0 Calibri;\f1\froman\fcharset0 Times-Roman;}
{\colortbl;\red255\green255\blue255;\red52\green52\blue52;\red83\green83\blue83;}
{\*\listtable{\list\listtemplateid1\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{none\}}{\leveltext\leveltemplateid1\'00;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid1}
{\list\listtemplateid2\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{none\}}{\leveltext\leveltemplateid101\'00;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid2}}
{\*\listoverridetable{\listoverride\listid1\listoverridecount0\ls1}{\listoverride\listid2\listoverridecount0\ls2}}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\deftab720
\pard\pardeftab720

\f0\fs30 \cf2 from numpy import array, hstack, vstack\
from sklearn import metrics, cross_validation, linear_model\
from sklearn.ensemble import RandomForestClassifier\
from sklearn import naive_bayes\
from sklearn import preprocessing, grid_search\
from scipy import sparse\
from itertools import combinations\
\
from sets import Set\
import numpy as np\
import pandas as pd\
import sys\
import os.path\
\
def group_data(data,degree=3,hash=hash):\
\'a0\'a0\'a0\'a0new_data = []\
\'a0\'a0\'a0\'a0m,n = data.shape\
\'a0\'a0\'a0\'a0for indicies in combinations(range(n), degree):\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0new_data.append([hash(tuple(row)) for row in data[:,indicies]])\
\'a0\'a0\'a0\'a0return np.array(new_data).T\
\
\
def OneHotEncoder(data, keymap=None):\
"""\
OneHotEncoder takes data matrix with categorical columns and\
converts it to a sparse binary matrix.\
\
Returns sparse binary matrix and keymap mapping categories to indicies.\
If a keymap is supplied on input it will be used instead of creating one\
and any categories appearing in the data that are not in the keymap are\
ignored\
"""\
if keymap is None:\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0 keymap = []\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0 for col in data.T:\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0 uniques = set(list(col))\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0 keymap.append(dict((key, i) for i, key in enumerate(uniques)))\
total_pts = data.shape[0]\
outdat = []\
for i, col in enumerate(data.T):\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0 km = keymap[i]\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0 num_labels = len(km)\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0 spmat = sparse.lil_matrix((total_pts, num_labels))\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0 for j, val in enumerate(col):\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0 if val in km:\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0 spmat[j, km[val]] = 1\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0 outdat.append(spmat)\
outdat = sparse.hstack(outdat).tocsr()\
return outdat, keymap\
\
def optimize_hyperparameters(X_tr, y_tr, model, param_grid, cv_tries):\
\'a0\'a0\'a0\'a0print "Performing hyperparameter optimization..."\
\'a0\'a0\'a0\'a0optimal_model = grid_search.GridSearchCV(model, param_grid, cv=cv_tries, n_jobs=-1)\
\'a0\'a0\'a0\'a0optimal_model.fit(X_tr, y_tr)\
\'a0\'a0\'a0\'a0print "Selected C: ", optimal_model.best_estimator_.C\
\'a0\'a0\'a0\'a0return optimal_model\
\
\
def save_results(features, filename):\
\'a0\'a0\'a0\'a0print "Save results in txt format"\
\'a0\'a0\'a0\'a0f = open(filename, 'w')\
\'a0\'a0\'a0\'a0f.write("Features List \\n")\
\'a0\'a0\'a0\'a0for i in range(len(features)):\'a0\'a0\'a0\'a0\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0f.write("%i\\n" % features[i])\
\
def save_model(model, filename):\
\'a0\'a0\'a0\'a0print "Save model in txt format"\
\'a0\'a0\'a0\'a0f = open(filename, 'w')\
\'a0\'a0\'a0\'a0f.write("Model \\n")\
\'a0\'a0\'a0\'a0f.write("%f\\n" % model)\
\
\
def create_test_submission(filename, prediction):\
\'a0\'a0\'a0\'a0content = ['id,ACTION']\
\'a0\'a0\'a0\'a0for i, p in enumerate(prediction):\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0content.append('%i,%f' %(i+1,p))\
\'a0\'a0\'a0\'a0f = open(filename, 'w')\
\'a0\'a0\'a0\'a0f.write('\\n'.join(content))\
\'a0\'a0\'a0\'a0f.close()\
\'a0\'a0\'a0\'a0print 'Saved'\
\
\
def cross_validate(X, y, model, n_tries):\
\'a0\'a0\'a0\'a0mean_auc = 0.0\
\'a0\'a0\'a0\'a0for i in range(n_tries):\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0X_tr, X_te, y_tr, y_te = cross_validation.train_test_split(X, y, test_size=1.0/float(n_tries),random_state = i*1234)\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0model.fit(X_tr, y_tr)\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0preds = model.predict(X_te)\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0roc_auc = metrics.roc_auc_score(y_te, preds)\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0mean_auc += roc_auc\
\'a0\'a0\'a0\'a0return (mean_auc/n_tries)\
\
\'a0\'a0\'a0\'a0\
def greedy_forward_feature_selection(Xts, y, model, cv_tries):\
\'a0\'a0\'a0\'a0print "Performing greedy forward feature selection..."\
\'a0\'a0\'a0\'a0selected_features = set()\
\'a0\'a0\'a0\'a0features_not_selected = range(len(Xts))\
\'a0\'a0\'a0\'a0selection_scores = [-1.0,0.0]\
\'a0\'a0\'a0\'a0while ((selection_scores[-2] < selection_scores[-1]) and (len(features_not_selected)>1)): \
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0# print"len(features_not_selected)", len(features_not_selected)\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0feature_scores = []\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0for f in features_not_selected:\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0proposed_new_feature_set = list(selected_features) + [f]\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0X_projected = sparse.hstack([Xts[j] for j in proposed_new_feature_set]).tocsr()\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0score = cross_validate(X_projected, y, model, cv_tries)\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0#print "\'a0\'a0Adding feature ", f, " gives score ", score, " (it was ", selection_scores[-1] , ")"\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0feature_scores.append((score, f))\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0best_score, best_feature = sorted(feature_scores)[-1]\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0selected_features.add(best_feature)\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0print "Current feature selection: ", list(selected_features), " with score ", best_score\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0features_not_selected.remove(best_feature)\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0selection_scores.append(best_score)\
\'a0\'a0\'a0\'a0if (len(features_not_selected) == 0):\'a0\'a0\'a0\'a0\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0selected_features.remove(sorted(a)[-1])\
\'a0\'a0\'a0\'a0return list(selected_features)\
\'a0\'a0\'a0\'a0\
\'a0\'a0\'a0\'a0\
\'a0\'a0\'a0\'a0\
def learn(X, y, model, hyperparameter_grid, cv_tries):\
\'a0\'a0\'a0\'a0if os.path.exists("selected_features.txt"):\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0print "File selected_features exist"\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0selected_features = np.loadtxt("selected_features.txt", skiprows=1,dtype=int)\
\'a0\'a0\'a0\'a0else:\'a0\'a0\'a0\'a0\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0selected_features = greedy_forward_feature_selection(X, y, model, cv_tries)\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0save_results(selected_features, "selected_features.txt")\
\'a0\'a0\'a0\'a0#\'a0\'a0\'a0\'a0print "selected_features", selected_features\'a0\'a0\'a0\'a0\
\'a0\'a0\'a0\'a0projected_X = sparse.hstack([X[j] for j in selected_features]).tocsr()\
\'a0\'a0\'a0\'a0optimal_model = optimize_hyperparameters(projected_X, y, model, hyperparameter_grid, cv_tries)\
\'a0\'a0\'a0\'a0optimal_model_score = cross_validate(projected_X, y, model,cv_tries)\
\'a0\'a0\'a0\'a0print "Optimal model score on training data: ", optimal_model_score\
\'a0\'a0\'a0\'a0return optimal_model, selected_features\
\'a0\'a0\'a0\'a0\
def load_data():\'a0\'a0\'a0\'a0\
\'a0\'a0\'a0\'a0train_data = pd.read_csv('train.csv')\
\'a0\'a0\'a0\'a0test_data = pd.read_csv('test.csv')\
\'a0\'a0\'a0\'a0alldata = np.vstack((train_data.ix[:,1:-1],test_data.ix[:,1:-1]))\
\'a0\'a0\'a0\'a0num_train = train_data.shape[0]\
\'a0\'a0\'a0\'a0#alldata = alldata[:,[0,1,4,5,6,7,8]]\'a0\'a0 #removing Role_Code1 & 2\
\'a0\'a0\'a0\'a0print "Transforming all the categorical data into numeric"\
\'a0\'a0\'a0\'a0relabler = preprocessing.LabelEncoder()\
\'a0\'a0\'a0\'a0for col in range(len(alldata[0,:])):\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0alldata[:,col] = relabler.fit_transform(alldata[:,col])\
\'a0\'a0\'a0\'a0dp = group_data(alldata,degree = 2)\
\'a0\'a0\'a0\'a0for col in range(len(dp[0,:])):\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0dp[:,col] = relabler.fit_transform(dp[:,col])\
\'a0\'a0\'a0\'a0dt = group_data(alldata,degree = 3)\
\'a0\'a0\'a0\'a0for col in range(len(dt[0,:])):\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0dt[:,col] = relabler.fit_transform(dt[:,col])\
\'a0\'a0\'a0\'a0#Train data\
\'a0\'a0\'a0\'a0y = np.array(train_data.ACTION)\
\'a0\'a0\'a0\'a0X = alldata[0:num_train]\
\'a0\'a0\'a0\'a0X_2 = dp[0:num_train]\
\'a0\'a0\'a0\'a0X_3 = dt[0:num_train]\
\'a0\'a0\'a0\'a0X_train_all = np.hstack((X,X_2,X_3))\
\'a0\'a0\'a0\'a0#Test data\
\'a0\'a0\'a0\'a0X_test = alldata[num_train:]\
\'a0\'a0\'a0\'a0X_test_2 = dp[num_train:]\
\'a0\'a0\'a0\'a0X_test_3 = dt[num_train:]\
\'a0\'a0\'a0\'a0X_test_all = np.hstack((X_test,X_test_2,X_test_3))\
\'a0\'a0\'a0\'a0#\'a0\'a0\'a0\'a0num_features = X_train_all.shape[1]\
\'a0\'a0\'a0\'a0return X_train_all,X_test_all,y,num_train\
\
X,X_test,y,num_train = load_data()\
num_features = X.shape[1]\
model = linear_model.LogisticRegression()\
\
\
\
X_enc = [OneHotEncoder(X[:,[i]])[0] for i in range(num_features)]\
hyper_parameters = \{'C': [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]\}\
n_tries = 10\
optimal_model, good_features = learn(X_enc, y, model,hyper_parameters,n_tries)\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\
#encoder.fit(vstack((X[:,good_features],X_test[:,good_features])))\
#X_train_final = encoder.transform(X[:,good_features])\
#X_test_final = encoder.transform(X_test_all[:,good_features])\
\
print "Performing One Hot Encoding on entire dataset..."\
Xt = np.vstack((X[:,good_features], X_test[:,good_features]))\
Xt, keymap = OneHotEncoder(Xt)\
X_train = Xt[:num_train]\
X_test = Xt[num_train:]\
\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\
print "Training the full best model"\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0 \
print "Making predictions and saving results"\
optimal_model.fit(X_train,y)\
preds = optimal_model.predict(X_test)\
create_test_submission('Predictions.txt', preds)\
\pard\tx220\tx720\pardeftab720\li720\fi-720
\ls1\ilvl0
\f1\fs28 \cf2 		\

\itap1\trowd \taflags0 \trgaph108\trleft-108 \trcbpat1 \trbrdrt\brdrnil \trbrdrl\brdrnil \trbrdrt\brdrnil \trbrdrr\brdrnil 
\clvertalt \clshdrawnil \clwWidth21500\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadl0 \clpadr0 \gaph\cellx8640
\pard\intbl\itap1\tx220\tx720\pardeftab720\li720\fi-720\sa60
\ls2\ilvl0
\fs24 \cf3 		\cf0 \cell \lastrow\row
\pard\pardeftab720

\fs28 \cf0 \
\'a0\
\'a0 \'a0}