# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:26:25 2019

@author: Herniz
"""

#import moduls
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
import math
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold 
from sklearn.pipeline import make_pipeline


from sklearn import linear_model
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

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
    plt.xlabel('random distribution vs.'+var)
    plt.ylabel('Culmanative Probability')
    plt.legend({'norm':x_norm, 'price':var})
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

#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))
    return (rmse)

# data loading
test=pd.read_csv('D:\\Python\\python projects\\home prices\\test.csv',index_col='Id')
houses=pd.read_csv('D:\\Python\\python projects\\home prices\\train.csv',index_col='Id')
target=houses.SalePrice

#EDA
houses.head()
houses.info()

houses.columns=houses.columns.str.lower()
test.columns=test.columns.str.lower()
cols=sorted(list(houses.columns.str.lower()))
target.describe()

#get rid of vars with over 1000 NA's
to_drop= [col for col in cols if houses[col].isna().sum()>1000]
houses[to_drop].describe()
houses=houses.drop(to_drop,axis=1)
test=test.drop(to_drop,axis=1)

#seperate numeric and categorical features
nums= houses.select_dtypes(np.number)
categorical=houses.select_dtypes(exclude=np.number)
categorical.info()

#create dictionary for hirarchies
grading={np.nan:0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
graded=[column for column in categorical.columns if categorical[column].isin(['Po','Fa','Av','Gd','Ex']).any()]  

nums1=houses[graded]
for column in nums1:
    print(nums1[column].value_counts())
for column in nums1:
    nums1[column].replace(grading, inplace=True)
nums1['bsmtexposure'].replace({np.nan:0,'No':1,'Mn':2,'Av':3,'Gd':4},inplace=True)

#remove the dictionaried features from categorical DF
categorical=categorical.drop(list(nums1.columns),axis=1)

#numeric values - pick the top15 variables according to corralation with sale price
nums=pd.concat([nums,nums1],axis=1)
price_corr=nums.corr().loc['saleprice'].drop('saleprice',axis=0).sort_values(ascending=False)
high_corr= (price_corr[price_corr>abs(0.5)])
high_corr_vars=list(high_corr.index)

#identify variables w\ high colinearity
iner=nums.corr().loc[high_corr.index,high_corr.index]
high_colinear=iner[iner>.7]
high_colinear.replace(to_replace=1,value=np.nan,inplace=True)
high_colinear=high_colinear.dropna(how='all',axis=1).dropna(how='all',axis=0)

#set a new DataFrame ith the high correlation vars

df=nums[high_corr_vars]
inner=df.corr()
high_colinear=iner[iner>.75]
high_colinear.replace(to_replace=1,value=np.nan,inplace=True)
high_colinear=high_colinear.dropna(how='all',axis=1).dropna(how='all',axis=0)

fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(inner, annot=True, xticklabels=inner.columns, 
            yticklabels=inner.columns, ax=ax, linewidths=.5, 
            vmin = -1, vmax=1, center=0)

fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(high_colinear, annot=True, xticklabels=high_colinear.columns, 
            yticklabels=high_colinear.columns, ax=ax, linewidths=.5, 
            vmin = -1, vmax=1, center=0)


colin_pairs=[]
for column in high_colinear.columns:
    row=high_colinear[column].idxmax()
    colin_pairs.append([row,column])
    
    
df=df.drop(['garagearea','1stflrsf','totrmsabvgrd'],axis=1)

###inspect variables normality 
#target var
target=houses['saleprice']
plt.subplot(2,1,1)
sns.distplot(target,fit=stats.norm)
plt.subplot(2,1,2)
compare_normality(target)


target=np.log(target)
plt.subplot(2,1,1)
sns.distplot(target,fit=stats.norm)
plt.subplot(2,1,2)
compare_normality(target)


#numeric vars
columns=[col for col in list(df.columns)]
df['price']=target

for var in columns:
    plt.subplot(1,3,1)
    sns.distplot(df[var],fit=stats.norm)
    plt.subplot(1,3,2)
    np.random.seed(seed=42)
    x_norm,y_norm=ecdf(np.random.normal(loc=np.mean(df[var]),scale=np.std(df[var]),size=len(df[var])))
    x_var,y_var=ecdf(df[var])
    plt.plot(x_norm,y_norm,marker='.',linestyle='none',color='black')
    plt.plot(x_var,y_var,marker='.',linestyle='none',color='red')
    plt.subplot(1,3,3)
    plt.scatter(df[var],df['price'],marker='.')
    plt.show()
    
#patch the year related vars
plt.subplot(2,3,1)
sns.distplot(df['yearbuilt'],fit=stats.norm)
plt.subplot(2,3,2)
np.random.seed(seed=42)
x_norm,y_norm=ecdf(np.random.normal(loc=np.mean(df['yearbuilt']),scale=np.std(df['yearbuilt']),size=len(df['yearbuilt'])))
x_var,y_var=ecdf(df['yearbuilt'])
plt.plot(x_norm,y_norm,marker='.',linestyle='none',color='black')
plt.plot(x_var,y_var,marker='.',linestyle='none',color='red')
plt.subplot(2,3,3)
plt.scatter(df['yearbuilt'],df['price'],marker='.')
plt.subplot(2,3,4)
sns.distplot(np.log(df['yearbuilt']),fit=stats.norm)
plt.subplot(2,3,5)
np.random.seed(seed=42)
x_norm,y_norm=ecdf(np.random.normal(loc=np.mean(np.log(df['yearbuilt'])),scale=np.std(np.log(df['yearbuilt'])),size=len(np.log(df['yearremodadd']))))
x_var,y_var=ecdf(np.log(df['yearbuilt']))
plt.plot(x_norm,y_norm,marker='.',linestyle='none',color='black')
plt.plot(x_var,y_var,marker='.',linestyle='none',color='red')
plt.subplot(2,3,6)
plt.scatter(np.log(df['yearbuilt']),df['price'],marker='.')
plt.show()

plt.subplot(2,3,1)
sns.distplot(df['yearremodadd'],fit=stats.norm)
plt.subplot(2,3,2)
np.random.seed(seed=42)
x_norm,y_norm=ecdf(np.random.normal(loc=np.mean(df['yearremodadd']),scale=np.std(df['yearremodadd']),size=len(df['yearremodadd'])))
x_var,y_var=ecdf(df['yearremodadd'])
plt.plot(x_norm,y_norm,marker='.',linestyle='none',color='black')
plt.plot(x_var,y_var,marker='.',linestyle='none',color='red')
plt.subplot(2,3,3)
plt.scatter(df['yearremodadd'],df['price'],marker='.')
plt.subplot(2,3,4)
sns.distplot(np.log(df['yearremodadd']),fit=stats.norm)
plt.subplot(2,3,5)
np.random.seed(seed=42)
x_norm,y_norm=ecdf(np.random.normal(loc=np.mean(np.log(df['yearremodadd'])),scale=np.std(np.log(df['yearremodadd'])),size=len(np.log(df['yearremodadd']))))
x_var,y_var=ecdf(np.log(df['yearremodadd']))
plt.plot(x_norm,y_norm,marker='.',linestyle='none',color='black')
plt.plot(x_var,y_var,marker='.',linestyle='none',color='red')
plt.subplot(2,3,6)
plt.scatter(np.log(df['yearremodadd']),df['price'],marker='.')
plt.show()

    
#normalize vars!    


#create regression models
X=df.drop('price',axis=1)
y=df['price']

X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=42)

reg = linear_model.Ridge(alpha=.5)
reg.fit(X_train,y_train)
reg_pred=reg.predict(X_test)
reg.score(X_test,y_test)
rmsle(reg_pred,y_test)

clf = tree.DecisionTreeRegressor()
clf.fit(X_train,y_train)
clf_pred=clf.predict(X_test)
clf.score(X_test,y_test)
rmsle(clf_pred,y_test)

forrest=RandomForestRegressor()
forrest.fit(X_train,y_train)
forrest_pred=forrest.predict(X_test)
forrest.score(X_test,y_test)
rmsle(forrest_pred,y_test)

results={'Ridge':{'RMSLE':rmsle(clf_pred,y_test),'Root Square':mean_squared_error(y_test,reg_pred)},'Tree Regressor':{'RMSLE':rmsle(clf_pred,y_test),'Root Square':mean_squared_error(y_test,clf_pred)},'Forrest':{'RMSLE':rmsle(forrest_pred,y_test),'Root Square':mean_squared_error(y_test,forrest_pred)}}

### adding categorical features###
categorical.info()
cat_describe=categorical.describe()
categorical.isna().sum()
cat_describe
cat_feat_list=list(categorical.columns)


#drop features with no varience

cat_describe=cat_describe.T
no_varience=cat_describe[cat_describe['freq']/cat_describe['count']>0.95]
no_varience_list=list(no_varience.index)
categorical=categorical.drop(no_varience_list,axis=1)


importent_cat_vars=[]


#some graphs at least
### assumption no.1 - the nieghberhood matters!###
nieghb_fig=sns.swarmplot(x='saleprice',y='neighborhood',data=houses,orient='v')
importent_cat_vars.append('neighborhood')


#assumption no.2 - having central ac is nice!
houses.hist(column='saleprice',by='centralair',bins=30,normed=True)
plt.subplot(1,2,1)
plt.ylim(0,0.000015)
plt.xlim(0,600000)
plt.subplot(1,2,2)
plt.ylim(0,0.000015)
plt.xlim(0,600000)

houses.hist(column='saleprice',by='centralair',bins=30,normed=True,cumulative=True)
plt.subplot(1,2,1)
plt.ylim(0,1.01)
plt.xlim(0,850000)
plt.subplot(1,2,2)
plt.ylim(0,1.01)
plt.xlim(0,850000)
importent_cat_vars.append('centralair')


#building type is important
sns.swarmplot(x='bldgtype',y='saleprice',data=houses)
sns.swarmplot(x='mszoning',y='saleprice',data=houses,hue='bldgtype')
importent_cat_vars.append('bldgtype')
importent_cat_vars.append('mszoning')

#fishy sales cause fishy prices
meancondition=houses.saleprice.groupby(houses['salecondition']).mean().sort_values()
sns.swarmplot(x='salecondition',y='saleprice',data=houses)
plt.bar(meancondition.index,meancondition)
importent_cat_vars.append('salecondition')

to_dumdum=houses[cat_feat_list]
to_dumdum.info()
to_dumdum=pd.get_dummies(to_dumdum)

#regression models 
df=pd.concat((df,to_dumdum),axis=1)
X=df.drop('price',axis=1)
y=df['price']

X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=42)

reg = linear_model.Ridge(alpha=.7)
reg.fit(X_train,y_train)
reg_pred=reg.predict(X_test)
reg.score(X_test,y_test)
rmsle(y_test,reg_pred)


lasso=linear_model.Lasso(alpha=0.00001, max_iter=10e5)
lasso.fit(X_train,y_train)
lasso_pred=lasso.predict(X_test)
lasso.score(X_test,y_test)
rmsle(lasso_pred,y_test)


clf = tree.DecisionTreeRegressor()
clf.fit(X_train,y_train)
clf_pred=clf.predict(X_test)
clf.score(X_test,y_test)
rmsle(clf_pred,y_test)

forrest=RandomForestRegressor()
forrest.fit(X_train,y_train)
forrest_pred=forrest.predict(X_test)
forrest.score(X_test,y_test)
rmsle(forrest_pred,y_test)

results2={'Ridge':{'RMSLE':rmsle(clf_pred,y_test),'Root Square':mean_squared_error(y_test,reg_pred)},'Tree Regressor':{'RMSLE':rmsle(clf_pred,y_test),'Root Square':mean_squared_error(y_test,clf_pred)},'Forrest':{'RMSLE':rmsle(forrest_pred,y_test),'Root Square':mean_squared_error(y_test,forrest_pred)}}

# Tweaking models
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
ridge = make_pipeline(preprocessing.RobustScaler(), linear_model.RidgeCV(alphas=alphas_alt, cv=kfolds))
lasso = make_pipeline(preprocessing.RobustScaler(), linear_model.LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))

ridge.fit(X_train,y_train)
ridge_pred=ridge.predict(X_test)
ridge.score(X_test,y_test)
rmsle(ridge_pred,y_test)

lasso.fit(X_train,y_train)
lasso_pred=lasso.predict(X_test)
lasso.score(X_test,y_test)
rmsle(lasso_pred,y_test)
