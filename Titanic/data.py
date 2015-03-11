import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import cPickle as pickle
import numpy as np
from itertools import combinations

def load_data():
    train = pd.read_csv('train.csv')
    train = train.set_index('PassengerId')
    #Management of missing values
    #Cabin contains too may null values
    #We can remove these 2 columns
    [train[x] for x in train.columns.tolist()]
    [train[x].isnull().sum() for x in train.columns.tolist()]
    train = clean_and_munge_data(train)
    train = filling_Mean_Age(train)
    #We remove ticket as we believe an important predictor for the survivorship
    train = train.drop(['Ticket','Cabin','Name','Title'], axis=1)
    """new_features = add_features(train)
    new_features.set_index = train.index
    train = train.join(new_features)"""
    #train.drop(['Ticket','Cabin'], axis=1, inplace=True)
    #Age contains also NANs but it s an important predictor
    #WE remove sample having Age = NAN
    return train
    

def add_features(train):
    cols = []
    for (x,y) in combinations(train.columns.tolist(),2):
        cols.append(str(x)+'_'+str(y))
    train = np.array(train)
    new_data = []
    for (i,j) in combinations(range(train.shape[1]),2):
        #new_data.append([str(x)+str(y) for (x,y) in train[:,[i,j]]])
        new_data.append([hash(tuple(x,y)) for (x,y) in train[:,[i,j]]])
    train = pd.DataFrame(np.array(new_data).T)
    train.columns = cols
    return train



def filling_Mean_Age(df):
    means = {}
    for title in np.unique(df.Title):
        mean = pd.DataFrame(df.groupby('Title')['Age'].mean())
        means[title] = int(mean[mean.index == title]['Age'])
    for i in (np.array(df.Age[df.Age.isnull()].index)):
        df.ix[i,'Age'] = means.get(df.loc[i]['Title'])
    return df


    
def exploratory(train):
    # Transform categorical into Numeric
    #for i in ['Embarked','Sex']:
    for i in train.columns.tolist():
        le = LabelEncoder()
        le.fit(train[i])
        train[i] = le.transform(train[i])
    # Fill out missing Age values with interpolation
    return (train)    

def save_models(models):
    with open('saved_params.p', 'wb') as fp:
        pickle.dump(models, fp)

def load_models(filename='saved_params.p'):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)


def substrings_in_string(stringa, title):
    for i in title:
        if i in stringa:
            return i



#replacing all titles with mr, mrs, miss, master
def replace_titles(x):
    title=x['Title']
    if title in ['Mr','Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Master']:
        return 'Master'
    elif title in ['Countess', 'Mme','Mrs']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms','Miss']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    elif title =='':
        if x['Sex']=='Male':
            return 'Master'
        else:
            return 'Miss'
    else:
        return title



def clean_and_munge_data(df):
    #creating a title column from name
    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                'Don', 'Jonkheer']
    df['Title']=df['Name'].map(lambda x: substrings_in_string(x, title_list))
    df['Title']=df.apply(replace_titles, axis=1)
    return df






