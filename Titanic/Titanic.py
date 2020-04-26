# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:20:29 2019

@author: Herniz
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
import math
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier,RandomForestClassifier,VotingClassifier
from sklearn.svm import SVC


from sklearn.model_selection import train_test_split,ShuffleSplit,cross_validate,GridSearchCV
from sklearn.feature_selection import RFE,RFECV
from sklearn.metrics import confusion_matrix
#%%
#define helper functions

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    
    return x, y

def plot_ecdf(x,y,xname):
    plt.plot(x,y,marker='.',linestyle='none')
    plt.xlabel(xname)
    plt.ylabel('Culmanative Probability')
    plt.show()
    







def compare_normality(var):
    np.random.seed(seed=42)
    x_norm,y_norm=ecdf(np.random.normal(loc=np.mean(var),scale=np.std(var),size=len(var)))
    x_var,y_var=ecdf(var)
    plt.plot(x_norm,y_norm,marker='.',linestyle='none',color='black')
    plt.plot(x_var,y_var,marker='.',linestyle='none',color='red')
    plt.xlabel('random distribution vs.'+ str(var.name) )
    plt.ylabel('Culmanative Probability')
    plt.legend({'normal':x_norm, str(var.name):x_var })
    plt.show()
    
def plot_correlation_map( df ):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )

def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()

#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


def cv_rmsle(model):
    rmsle = np.sqrt(np.log(-cross_val_score(model, X, y,
                                           scoring = 'neg_mean_squared_error',
                                           cv=kfolds)))
    return(rmsle)
    
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
    
#%%
# =============================================================================
# data loading and merging
# =============================================================================
test=pd.read_csv('C:\\Python\\python projects\\Titanic\\test.csv',index_col='PassengerId')
train=pd.read_csv('C:\\Python\\python projects\\Titanic\\train.csv',index_col='PassengerId')
target=train.Survived
df=pd.concat([train,test])

print(df.describe())
df.head()

#%%
# =============================================================================
# NA's 
# =============================================================================
is_null=df.isna().sum().sort_values(ascending=False)

#cabin has a ton of NA's but maybe those cabins' that are available are valueable info
# maybe only vips got to reserve a cabin

df.Cabin=df.Cabin.fillna('Other')
df.Cabin.describe()

cabin=df.Cabin.map(lambda x:x.strip()[0])
df['cabin']=cabin
sns.swarmplot('cabin','Pclass',data=df,order=['A','B','C','D','E','F','G','T','O'])
#it seems that cabin T is a mistake, so i'll inpute the information for that passenger
#%%
df[df['cabin']=='T']
df.groupby('cabin')['Pclass','Fare'].agg(pd.Series.mode)
df.groupby('cabin')['Pclass','Fare'].mean()
df['Embarked'][df['cabin']!='O'].value_counts()

#df.groupby('cabin')['Fare','Pclass','Embarked'].agg(pd.Series.mode)
#the closest cabin that consist passenger fron 1st and second class and is either D or E
df['cabin'][df['cabin']=='T']='D'
df=df.drop('Cabin',axis=1)

#Age
df.groupby(['Pclass','Sex'])['Age'].median()
df['Age'][(df['Age'].isna()) & (df['Pclass']==1)& (df['Sex']=='female')]=36
df['Age'][(df['Age'].isna()) & (df['Pclass']==1)& (df['Sex']=='male')]=42

df['Age'][(df['Age'].isna()) & (df['Pclass']==2)& (df['Sex']=='female')]=28
df['Age'][(df['Age'].isna()) & (df['Pclass']==2)& (df['Sex']=='male')]=29.5

df['Age'][(df['Age'].isna()) & (df['Pclass']==3)& (df['Sex']=='female')]=22
df['Age'][(df['Age'].isna()) & (df['Pclass']==3)& (df['Sex']=='male')]=25

#Embarked
a=df.groupby(['Pclass','Fare'])['Embarked'].agg(pd.Series.mode)

df['Embarked'][df.Embarked.isna()]=df.Embarked.fillna('S')

#fare
df[['cabin','Pclass','Embarked']][df['Fare'].isna()]
df.groupby(['cabin','Pclass','Embarked'])['Fare'].mean()
df['Fare'][df['Fare'].isna()]=14.5

#%%
#outliers
df.Fare.describe(percentiles =[.95,.97,.99])
df['Fare'][df['Fare']>df['Fare'].quantile(.99)]=263

df.Age.describe(percentiles =[.95,.97,.99])
df['Age'][df['Age']>df['Age'].quantile(.99)]

#%%
#normality

compare_normality(df['Age'])

compare_normality(df['Fare'])
compare_normality(np.log1p(df['Fare']))

#%%
# =============================================================================
# feature engineering
# =============================================================================
df['Total_family']=df['Parch']+df['SibSp']

df['Is_alone']=df['Total_family'].map(lambda x: 1 if x==0 else 0)


title=df.Name.map(lambda x:x.split(',')[1].split('.')[0].strip())
title.value_counts()
Title_Dictionary = {
                    'Mr':'Mr',
                    'Miss':'Miss',
                    'Mrs':'Mrs',
                    'Master':'Master',
                    'Rev':'Misc',
                    'Dr':'Misc',
                    'Col':'Misc',
                    'Major':'Misc',
                    'Mlle':'Miss',
                    'Ms':'Mrs',
                    'Lady':'Misc',
                    'Sir':'Misc',
                    'Jonkheer':'Misc',
                    'Don':'Misc',
                    'Dona':'Misc',
                    'Mme':'Mrs',
                    'the Countess':'Misc',
                    'Capt':'Misc',
                    }
df['Title']=title
df['Title']=df['Title'].map(Title_Dictionary)
df['Title'].value_counts()
df=df.drop('Name',axis=1)
df=df.drop('Ticket',axis=1)



#%%

train=df[df.Survived.notna()]

test=df[df.Survived.isna()]

#%%
# =============================================================================
# EDA
# =============================================================================
plot_categories( train, cat='Sex',target='Survived',col='Sex')

train[train['Sex']=='female']['Survived'].sum()/train[train['Sex']=='female']['Survived'].count()
train[train['Sex']=='male']['Survived'].sum()/train[train['Sex']=='male']['Survived'].count()

sns.barplot('Sex',y='Survived',data=train)
plt.title('Survival rate by sex' )

# =============================================================================
#  right off the bat it is visible that women have a much greater survival rate
# =============================================================================

#%%
plt.figure(figsize=(2,3))
sns.FacetGrid( train , hue='Survived', aspect=4).map( sns.kdeplot , 'Age' , shade= True )
plt.xlim(0,train['Age'].max())
plot_distribution(train,'Age','Survived')
plt.show()

plt.figure(figsize=(10,8))
grid = plt.GridSpec(3, 2, wspace=0.5, hspace=0.4)
ax0=plt.subplot(grid[0,0:])
ax0=plt.hist(x='Age',bins=10,data=train[train['Survived']==1],density=True,alpha=0.5,color='blue')
ax0=plt.hist(x='Age',bins=10,data=train[train['Survived']==0],density=True,alpha=0.5,color='red')
ax0=plt.title('Age comparison of survivors vs.casualties')
ax0=plt.legend(('Survivors','Casualties'))
ax1=plt.subplot(grid[1,0])
ax1=plt.hist(x='Age',bins=10,data=train[train['Survived']==1],density=True,alpha=0.5,color='blue')
ax1=plt.title('Age Histogram of Survivers')
ax2=plt.subplot(grid[1,1])
ax2=plt.hist(x='Age',bins=10,data=train[train['Survived']==0],density=False,alpha=0.5,color='red')
ax2=plt.title('Age Histogram of casualties')
ax3=plt.subplot(grid[2,0:])
ax3=sns.swarmplot('Sex','Age',data=train,hue='Survived')
ax3=plt.axhline(y=14,xmin=-1)
ax3=plt.annotate('Age 14',[.5,14])
ax3=plt.title('Survivors vs casualties by Age & Sex')
plt.show()
# =============================================================================
# conclusion= children were prioritized on the board of the titanic
# young boys show the most survival enhancement comparing to male adults.
# =============================================================================

#%%
plot_categories( train , 'Pclass' , 'Survived')

plot_distribution(train,'Fare','Survived')
plt.xlim(0,train['Fare'].quantile(.95))

plot_distribution(train,'Fare','Survived',row='Sex')
plt.xlim(0,train['Fare'].quantile(.95))

#shockingly, the passenegers who paid lower fare are more likele to day, especially if they are men.


#%%
plt.figure(figsize=(10,6))
grid = plt.GridSpec(4, 2, wspace=0.2, hspace=0.5)
plt.subplot(grid[0,0:])
sns.barplot('Embarked','Survived',data=train,order=['C','S','Q'])
plt.title('Total Survival rate by place of embarkment')
plt.subplot(grid[1,0])
sns.barplot('Embarked','Survived',data=train[train['Sex']=='male'],order=['C','S','Q'])
plt.title('Total Survival for men rate by place of embarkment')
plt.subplot(grid[1,1])
sns.barplot('Embarked','Survived',data=train[train['Sex']=='female'],order=['C','S','Q'])
plt.title('Total Survival rate for women by place of embarkment')
plt.subplot(grid[2,:])
ax=sns.swarmplot('Embarked','Fare',data=train,hue='Survived',order=['C','S','Q'])
plt.title('Fare rate distribution by place of embarkment')
plt.show()

#%%
sns.violinplot(x='Title',y='Survived',data=train)
sns.barplot(x='Title',y='Survived',data=train,hue='Sex')

#%%
sns.barplot('Total_family','Survived',data=train)
sns.lmplot('Total_family','Survived',data=train,logistic=True,hue='Sex')
sns.swarmplot('Survived','Total_family',data=train,hue='Sex')

#%%
numerical=df[['Age','Fare','Parch','SibSp','Total_family']]
numerical['Age']=pd.cut(numerical['Age'],bins=10)
numerical['Fare']=pd.cut(numerical['Fare'],bins=5)
numerical[['Parch','SibSp','Total_family']]=numerical[['Parch','SibSp','Total_family']].astype('category',copy=False)

categorical=df.drop(['Age','Fare','Parch','SibSp','Total_family','Survived'],axis=1)
target=df.Survived

numerical.dtypes
df.Title.dtype

for var in list(categorical.columns):
    df[var]=df[var].astype('category',copy=False)
    categorical[var]=categorical[var].astype('category',copy=False)
df[categorical.columns].dtypes
categorical.dtypes

df1=pd.concat([numerical,categorical,target],axis=1)
df1=pd.get_dummies(df1,drop_first=True)


train=df1[df['Survived'].notna()]
test=df1[df['Survived'].isna()]
target=target[target.notna()]

#%%
X=train.drop('Survived',axis=1)
y=target

X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=42)
cv_split = ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0)


#%%
logreg=LogisticRegression()
tree=DecisionTreeClassifier()
knn=KNeighborsClassifier()
svc=SVC()
gaus= GaussianNB()
ada=AdaBoostClassifier()
bag=BaggingClassifier()
forrest=RandomForestClassifier() 
gbc=GradientBoostingClassifier()


MLA=[logreg,tree,knn,gaus,svc,ada,bag,forrest,gbc]

param_grid={logreg.__class__.__name__:{'penalty' : ['l1', 'l2'],
                      'C' : np.logspace(-4, 4, 20),
                      'solver' : ['liblinear']},
            tree.__class__.__name__:{'criterion': ['gini', 'entropy'],  
                    'max_depth': [2,4,6,8,10,None], 
                    'min_samples_split': [2,5,10,.03,.05], #minimum subset size BEFORE new split (fraction is % of total); default is 2
                    'min_samples_leaf': [1,5,10,.03,.05], #minimum subset size AFTER new split split (fraction is % of total); default is 1
                    'random_state':[0]},
             knn.__class__.__name__:{'n_neighbors': [1,2,3,4,5,6,7], 
                   'weights': ['uniform', 'distance'], #default = ‘uniform’
                   'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
             gaus.__class__.__name__:{},
             svc.__class__.__name__:{'C': [1,2,3,4,5], 
                   'gamma':  [.1, .25, .5, .75, 1.0], 
                   'decision_function_shape': ['ovo', 'ovr'], 
                   'probability': [True],
                   'random_state': [0]},
             ada.__class__.__name__:{'n_estimators': [10, 50, 100, 300],
                   'learning_rate': [.01, .03, .05, .1, .25], 
                   'random_state': [0]},
             bag.__class__.__name__:{'n_estimators': [10, 50, 100, 300],
                   'max_samples': [.1, .25, .5, .75, 1.0], 
                   'random_state': [0]},
             forrest.__class__.__name__:{'n_estimators': [10, 50, 100, 300],
                       'max_depth': [2, 4, 6, 8, 10, None],
                       'criterion':['gini', 'entropy'],
                       'random_state': [0]},
             gbc.__class__.__name__:{'learning_rate': [.01, .03, .05, .1, .25],
                   'max_depth': [2, 4, 6, 8, 10, None],
                   'n_estimators':[10, 50, 100, 300],
                   'random_state': [0]}}

cv_split = ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0)
MLA_columns= ['MLA Name', 'MLA Parameters','MLA Base Score mean','MLA Base Score 3*std','MLA Base Score min','MLA Base Time','MLA_CV score mean', 'MLA_CV score 3*std' ,'MLA_CV score min' 'MLA_CV Time']         
MLA_compare = pd.DataFrame(columns = MLA_columns)
MLA_predict = pd.DataFrame()
MLA_predict['True_value']=y
MLA_predict = pd.DataFrame(y_test)

#%%
row_index = 0
for alg in MLA:

    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    cv_results = cross_validate(alg, X_train, y_train, cv  = cv_split)
    MLA_compare.loc[row_index, 'MLA Base Score mean'] = cv_results['test_score'].mean()*100
    MLA_compare.loc[row_index, 'MLA Base Score 3*std'] = cv_results['test_score'].std()*3
    MLA_compare.loc[row_index, 'MLA Base Score min'] = cv_results['test_score'].min()
    MLA_compare.loc[row_index, 'MLA Base Time'] = cv_results['fit_time'].mean()
    GS=GridSearchCV(alg,param_grid=param_grid[MLA_name],scoring = 'roc_auc', cv = cv_split)
    GS.fit(X_train,y_train)
    MLA_compare.loc[row_index, 'MLA_CV score mean'] = GS.cv_results_['mean_test_score'].mean()*100
    MLA_compare.loc[row_index, 'MLA_CV score 3*std'] = GS.cv_results_['mean_test_score'].std()*3
    MLA_compare.loc[row_index, 'MLA_CV Score min'] = GS.cv_results_['mean_test_score'].min()
    MLA_compare.loc[row_index, 'MLA_CV Time'] = GS.cv_results_['mean_fit_time'].mean()
    alg.set_params(**GS.best_params_)
    alg.fit(X_train, y_train)
    MLA_predict[MLA_name] = GS.predict(X_test)   
    row_index+=1
    
plot_correlation_map(MLA_predict)
#%%
voters=[
        ('logreg',logreg),
        ('tree',tree),
        ('knn',knn),
        ('gaus',gaus),
        ('svc',svc),
        ('ada',ada),
        ('bag',bag),
        ('forrest',forrest),
        ('gbc',gbc)]


vote_hard = VotingClassifier(estimators = voters, voting = 'hard')
vote_hard_cv = cross_validate(vote_hard, X_train, y_train, cv  = cv_split)
vote_hard.fit(X_train, y_train)

vote_soft = VotingClassifier(estimators = voters, voting = 'soft')
vote_soft_cv = cross_validate(vote_soft, X_train, y_train, cv  = cv_split)
vote_soft.fit(X_train, y_train)


print("Hard Voting Test w/bin score mean: {:.2f}". format(vote_hard_cv['test_score'].mean()*100))
print("Soft Voting Test w/bin score mean: {:.2f}". format(vote_soft_cv['test_score'].mean()*100))

#%%
#1st round of prediction
test=test.drop('Survived',axis=1)
hard_pred=vote_hard.predict(test)
soft_pred=vote_soft.predict(test)
forrest_pred=forrest.predict(test)
bag_pred=bag.predict(test)

hard_pred=hard_pred.astype('int',copy=False)
soft_pred=soft_pred.astype('int',copy=False)
forrest_pred=forrest_pred.astype('int',copy=False)
bag_pred=bag_pred.astype('int',copy=False)


#submission 1 - hard vote
Submission = pd.DataFrame({ 'PassengerId': test.index,
                            'Survived': hard_pred})
Submission.to_csv("titanic_hardvote.csv", index=False)

#submission 2 - soft vote
Submission = pd.DataFrame({ 'PassengerId': test.index,
                            'Survived': soft_pred})
Submission.to_csv("titanic_softvote.csv", index=False)

#submission 3 - random forrest
Submission = pd.DataFrame({ 'PassengerId': test.index,
                            'Survived': forrest_pred})
Submission.to_csv("titanic_forrest.csv", index=False)

#submission 4 - bagging classifier
Submission = pd.DataFrame({ 'PassengerId': test.index,
                            'Survived': bag_pred})
Submission.to_csv("titanic_bag.csv", index=False)

#%%
#investigate feature importence and remove
# XGBOOST
