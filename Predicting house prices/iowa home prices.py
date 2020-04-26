# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
from sklearn.model_selection import train_test_split 

from sklearn import linear_model
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

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

def rmsle(y_pred, y_test) : 
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))    

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

#categorical features
categorical.info()
cat_describe=categorical.describe()
categorical.isna().sum()
cat_feat_list=list(categorical.columns)

#create dictionary for hirarchies
grading={np.nan:0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
graded=[column for column in categorical.columns if categorical[column].isin(['Po','Fa','Av','Gd','Ex']).any()]  

nums1=houses[graded]
for column in nums1:
    print(nums1[column].value_counts())
nums1=houses[graded].drop('bsmtexposure',axis=1)
for column in nums1:
    nums1[column].replace(grading, inplace=True)
nums1['bsmtexposure'].replace({np.nan:0,'No':1,'Mn':2,'Av':3,'Gd':4},inplace=True)

#remove the dictionaried features from categorical DF
categorical=categorical.drop(list(nums1.columns),axis=1)

#let us restart
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
importent_cat_vars.append(['bldgtype','mszoning'])

comparison={}

for feat in cat_feat_list:
    comparison[feat]=houses['saleprice'].groupby(houses[feat]).mean()



#numeric values - reduce var list according to correlation
nums=pd.concat([nums,nums1],axis=1)
price_corr=nums.corr().loc['saleprice'].drop('saleprice',axis=0).sort_values(ascending=False)
high_corr= (price_corr[price_corr>abs(0.45)])
high_corr_vars=list(high_corr.index)

#identify variables w\ high colinearity
iner=houses.corr().loc[high_corr.index]
high_colinear=iner[iner>.7]
high_colinear.replace(to_replace=1,value=np.nan,inplace=True)
high_colinear=high_colinear.dropna(how='all',axis=1).dropna(how='all',axis=0).drop('saleprice',axis=1).drop('overallqual',axis=0)

colin_pairs=[]
for column in high_colinear.columns:
    row=high_colinear[column].idxmax()
    colin_pairs.append([row,column])
    

#extracting important features
houses.boxplot('saleprice',by='overallqual')


plt.scatter(df.grlivarea,df.price,marker='.',c=df.overallqual,alpha=0.5,)
plt.ylabel('price')
plt.legend(df.overallqual)
plt.show()

#setting the working frame of vars
df=nums[high_corr_vars[0:14]]
iner=df.corr().loc[high_corr.index[0:14]]
high_colinear=iner[iner>.75]
high_colinear.replace(to_replace=1,value=np.nan,inplace=True)
high_colinear=high_colinear.dropna(how='all',axis=1).dropna(how='all',axis=0)

colin_pairs=[]
for column in high_colinear.columns:
    row=high_colinear[column].idxmax()
    colin_pairs.append([row,column])
    
corr=df.corr(method='pearson')
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(corr, annot=True, xticklabels=corr.columns, 
            yticklabels=corr.columns, ax=ax, linewidths=.5, 
            vmin = -1, vmax=1, center=0)
    
df=df.drop(['garagearea','1stflsf','totroomsabvfrd'],axis=1)

###inspect variables: 1. normality, 2.homoscadasety
columns=[col for col in list(df.columns)]

k=1
for var in columns:
    plt.subplot(3,len(columns),k)
    sns.distplot(df[var],fit=stats.norm)
    k+=1
    plt.subplot(3,len(columns),k)
    np.random.seed(seed=42)
    x_norm,y_norm=ecdf(np.random.normal(loc=np.mean(df[var]),scale=np.std(df[var]),size=len(df[var])))
    x_var,y_var=ecdf(df[var])
    plt.plot(x_norm,y_norm,marker='.',linestyle='none',color='black')
    plt.plot(x_var,y_var,marker='.',linestyle='none',color='red')
    k+=1
    plt.subplot(3,len(columns),k)
    plt.scatter(df[var],df['price'],marker='.')
    k+=1
    
target=np.log(target)
df['price']=target 

compare_normality(df['yearremodadd'])
df.price=np.log(df.price)

sns.pairplot(df.iloc[:,0:4], size = 2.5)
plt.show()

df.info()
transformer=preprocessing.Normalizer()
to_norm=df
for var in list(df.columns):
    to_norm[var]=df[var]
    to_norm[var]=transformer.transform(to_norm)
    df[var]=to_norm


df=transformer.transform(df)
X=df.drop('price',axis=1)
y=df['price']

X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=42)

reclfg = linear_model.Ridge(alpha=.5)
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

# Tweaking models
