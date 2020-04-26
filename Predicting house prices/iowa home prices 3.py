# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 12:54:59 2019

@author: Herniz
"""

import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
import math
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn import linear_model 
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score,GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC,Ridge
from sklearn import tree
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

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
    plt.legend({'norm':x_norm, 'price':str(var.name) })
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


def cv_rmsle(model):
    rmsle = np.sqrt(np.log(-cross_val_score(model, X, y,
                                           scoring = 'neg_mean_squared_error',
                                           cv=kfolds)))
    return(rmsle)
    
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
    

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1) 
#%%
#Data loading
test=pd.read_csv('D:\\Python\\python projects\\home prices\\test.csv',index_col='Id')
train=pd.read_csv('D:\\Python\\python projects\\home prices\\train.csv',index_col='Id')
target=train.SalePrice

#Understanding the explained variable
plt.scatter(train.GrLivArea,train.SalePrice)
train=train[train['GrLivArea']<4500]
plt.scatter(train.GrLivArea,train.SalePrice)

plt.subplot(2,1,1)
sns.distplot(target,fit=stats.norm)
plt.subplot(2,1,2)
compare_normality(target)

plt.figure()
plt.subplot(2,1,1)
sns.distplot(np.log1p(target),fit=stats.norm)
plt.subplot(2,1,2)
compare_normality(np.log1p(target))
plt.show()

target=np.log1p(target)
#%%
# =============================================================================
# Feature Engeneering
# =============================================================================
#concatanating test and train: df
df=pd.concat([train,test], sort=False)

#Imputing NA
null_sum=df.isna().sum().sort_values(ascending=False)
null_sum=null_sum[null_sum!=0]

#handeling garage variables: 
#first, imputing variables to the 2 variables where there is a garage but rest of the variabls are NA
#second, replacing NA with 0
df[['GarageType','GarageYrBlt','GarageFinish','GarageCond','GarageQual']][df['GarageType'].notna( ) & df['GarageQual'].isna() ]
df[['GarageQual','GarageFinish','GarageCond']].groupby(df['GarageType']).agg(pd.Series.mode)
df['GarageFinish'][df['GarageFinish'].isna() & df['GarageType'].notna()]='Unf' 
df['GarageCond'][df['GarageCond'].isna() & df['GarageType'].notna()]='TA' 
df['GarageQual'][df['GarageQual'].isna() & df['GarageType'].notna()]='TA' 
df['GarageYrBlt'][df['GarageYrBlt'].isna() & df['GarageType'].notna()]=df['YearRemodAdd']
df[['GarageType','GarageYrBlt','GarageFinish','GarageCond','GarageQual']]=df[['GarageType','GarageYrBlt','GarageFinish','GarageCond','GarageQual']].fillna(0)
df[['GarageCond','GarageQual','GarageArea','GarageCars']][df['GarageCars'].isna()]
df[['GarageCond','GarageQual','GarageArea','GarageCars']].groupby(['GarageCond','GarageQual']).mean()
df['GarageCars']=df['GarageCars'].fillna(2)
df['GarageArea']=df['GarageArea'].fillna(508)

df['HasGarage']=df['GarageArea'].apply(lambda x: 0 if x==0 else 1)

#handeling BSMT variables
df[['BsmtExposure','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1']][df['BsmtFinType2'].isna() &df['BsmtFinType1'].notna()]
df[['BsmtExposure','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1']].groupby('BsmtFinType1').agg(pd.Series.mode)
df['BsmtExposure'][df['BsmtFinType2'].isna() &df['BsmtFinType1'].notna()]=df['BsmtExposure'].fillna('No')
df['BsmtQual'][df['BsmtFinType2'].isna() &df['BsmtFinType1'].notna()]=df['BsmtQual'].fillna('Gd')
df['BsmtFinType2'][df['BsmtFinType2'].isna() &df['BsmtFinType1'].notna()]=df['BsmtFinType2'].fillna('Unf')

df[['BsmtExposure','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1']][df['BsmtQual'].isna() &df['BsmtFinType1'].notna()]
df[['BsmtExposure','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1']].groupby(['BsmtFinType1','BsmtFinType2']).agg(pd.Series.mode)
df['BsmtQual'][df['BsmtQual'].isna() &df['BsmtFinType1'].notna()]=df['BsmtQual'].fillna('TA')

df[['BsmtExposure','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1']][df['BsmtCond'].isna() &df['BsmtFinType1'].notna()]
df[['BsmtExposure','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1']].groupby(['BsmtQual','BsmtFinType1']).agg(pd.Series.mode)
df['BsmtCond']=df['BsmtCond'].fillna('TA')

df[['BsmtExposure','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1']][df['BsmtExposure'].isna() &df['BsmtFinType1'].notna()]
df[['BsmtExposure','BsmtQual','BsmtFinType1']].groupby(['BsmtQual','BsmtFinType1']).agg(pd.Series.mode)
df['BsmtExposure'][df['BsmtExposure'].isna() &df['BsmtFinType1'].notna()]=df['BsmtExposure'].fillna('No')

df[['TotalBsmtSF','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1']][df['BsmtUnfSF'].isna()]
df[['TotalBsmtSF','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1']][df['BsmtFullBath'].isna()]
df[['TotalBsmtSF','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1']][df['BsmtHalfBath'].isna()]

bsmt_var=['BsmtHalfBath','TotalBsmtSF','BsmtFullBath','BsmtExposure','BsmtUnfSF','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1','BsmtFinSF1','BsmtFinSF2']
for var in bsmt_var:
    df[var]=df[var].fillna(0)

df['HasBsmt']=df['TotalBsmtSF'].apply(lambda x: 0 if x==0 else 1)

    
#other vars:
df['KitchenQual'].groupby(df['OverallQual']).agg(pd.Series.mode)
df['OverallQual'][df['KitchenQual'].isna()]
df['KitchenQual']=df['KitchenQual'].fillna('TA')

df.Electrical=df.Electrical.fillna(df.Electrical.value_counts()[0])
df.Exterior1st=df.Exterior1st.fillna(df.Exterior1st.value_counts()[0])
df.Exterior2nd=df.Exterior2nd.fillna(df.Exterior2nd.value_counts()[0])
df.SaleType=df.SaleType.fillna(df.SaleType.value_counts()[0])
df.Utilities=df.Utilities.fillna(df.Utilities.value_counts()[0])
df.Functional=df.Functional.fillna(df.Functional.value_counts()[0])

df[['MSZoning','Neighborhood']][df['MSZoning'].isna()]
df[['Neighborhood','MSZoning']].groupby('Neighborhood').agg(pd.Series.mode)
df.loc[2905,'MSZoning']='RL'
df.MSZoning=df.MSZoning.fillna('RM')

df['Fireplaces'][df['FireplaceQu'].isna() & df['Fireplaces']!=0]
df['Fireplaces']=df['Fireplaces'].fillna(0)
df['FireplaceQu']=df['FireplaceQu'].fillna(0)

df['HasFireplaces']=df['Fireplaces'].apply(lambda x: 0 if x==0 else 1)

df['MiscFeature']=df['MiscFeature'].fillna(0) 
df=df.rename(columns = {'MiscFeature':'HasMiscFeature'})
df['HasMiscFeature']=df['HasMiscFeature'].apply(lambda x: 0 if x==0 else 1)

df['PoolQC'][df['PoolQC'].isna() & df['PoolArea']!=0]
df['PoolQC']=df['PoolQC'].fillna(0)

df['HasPool']=df['PoolArea'].apply(lambda x: 0 if x==0 else 1)
        
df['Fence']=df['Fence'].fillna(0)
df['HasFence']=df['Fence'].apply(lambda x: 0 if x==0 else 1)

df['Alley']=df['Alley'].fillna(0)
df['HasAlleyAccsess']=df['Alley'].apply(lambda x: 0 if x==0 else 1)
        
df[['LotFrontage','Neighborhood']][df.LotFrontage.isna()]
lotneigh=df.LotFrontage.groupby(df.Neighborhood).mean()

for id in df[['LotFrontage','Neighborhood']][df.LotFrontage.isna()].index:
    neigh=df.loc[id,'Neighborhood']
    df.loc[id,'LotFrontage']=lotneigh[neigh]
    
df.MasVnrType.value_counts()
df.MasVnrArea.value_counts().sort_values()

df[['MasVnrType','MasVnrArea']][df.MasVnrType.isna() & df.MasVnrArea.notna()]='BrkFace'
df.MasVnrType=df.MasVnrType.fillna('None')
df.MasVnrArea=df.MasVnrArea.fillna(0)

null_sum=df.isna().sum().sort_values(ascending=False)
null_sum=null_sum[null_sum!=0]

#%%
#seperatin Categorical features and nums
nums=df.select_dtypes(np.number)
categories=df.select_dtypes(exclude=np.number)

#Factorization
nums.columns
fake_nums=['HasFireplaces','GarageYrBlt','MSSubClass','HasMiscFeature','MoSold','YrSold', 'HasGarage', 'HasBsmt', 'HasPool', 'HasFence', 'HasAlleyAccsess']
fakes=nums[fake_nums]
fakes=fakes.astype('object',copy=False)
nums=nums.drop(fake_nums,axis=1)
categories=pd.concat([categories,fakes],axis=1)
categories=categories.astype('category',copy=False)
categories.info()

#Getting rid of unvaried columns
mode_freq={}
for var in df.columns:
    freq=df[var].value_counts()[df[var].mode()[0]]/len(df[var])
    mode_freq[var]=freq
freq=pd.Series(mode_freq,index=mode_freq.keys())
unvaried=freq[freq>=0.97].drop(['HasBsmt','HasPool'])

df['HasBsmt'].value_counts()
df=df.drop(unvaried.index,axis=1)

#unifing repetative fetures
inner=nums.corr()
high_colinear=inner[inner>.75]
high_colinear.replace(to_replace=1,value=np.nan,inplace=True)
high_colinear=high_colinear.dropna(how='all',axis=1).dropna(how='all',axis=0)
colin_pairs=[]
for column in high_colinear.columns:
    row=high_colinear[column].idxmax()
    colin_pairs.append([row,column])

fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(inner, annot=True, xticklabels=inner.columns, 
            yticklabels=inner.columns, ax=ax, linewidths=.5, 
            vmin = -1, vmax=1, center=0)

fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(high_colinear, annot=True, xticklabels=high_colinear.columns, 
            yticklabels=high_colinear.columns, ax=ax, linewidths=.5, 
            vmin = -1, vmax=1, center=0)


vars=sorted(list(df.columns))
df['TotalSF']=df['1stFlrSF']+df['2ndFlrSF']+df['GrLivArea']+df['BsmtFinSF1'] + df['BsmtFinSF2']
df['TotalPorchSF']=df['OpenPorchSF']+df['EnclosedPorch']+df['ScreenPorch']
df['TotalBath']=df['BsmtFullBath']+(df['BsmtHalfBath']*.5)+df['FullBath']+(df['HalfBath']*.5)
df=df.drop(['EnclosedPorch','ScreenPorch','OpenPorchSF'],axis=1)
df=df.drop(['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath'],axis=1)
df=df.drop(['GarageArea','1stFlrSF'],axis=1)

#numeric values Skew exploration

skewness=nums.skew(axis=0).sort_values(ascending=False)
skewd_list=skewness[abs(skewness)>(0.5)].index
skewd=nums[skewd_list]
skewd.info()

for var in skewd.columns:
    skewd[var]=np.log1p(skewd[var])
    df[var]=skewd[var]
    print(var+' transformed')        
log_skewness=skewd.skew(axis=0).sort_values(ascending=False)

#Getting rid of unvaried columns
mode_freq={}
for var in df.columns:
    freq=df[var].value_counts()[df[var].mode()[0]]/len(df[var])
    mode_freq[var]=freq
freq=pd.Series(mode_freq,index=mode_freq.keys())
unvaried=freq[freq>=0.97].drop(['HasBsmt','HasPool'])

df=df.drop(unvaried.index,axis=1)

#creating dummies
categories1=df.select_dtypes(exclude=np.number)
nums1=df.select_dtypes(np.number)
for var in list(categories1.columns):
    df[var]=df[var].astype('category',copy=False)
df[categories1.columns].dtypes
df1=pd.get_dummies(df)

train=df1.loc[:len(target),:]
test=df1.loc[len(target)+1:,:]

#%%
# =============================================================================
# #create regression models
# =============================================================================
X=train.drop('SalePrice',axis=1)
y=train.SalePrice

X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=42)


kfolds = KFold(n_splits=10, shuffle=True, random_state=23)
lm=linear_model.LinearRegression()
scaler=preprocessing.RobustScaler()
ridge=linear_model.Ridge(alpha=6.099999999999998)
lasso=linear_model.Lasso(alpha=0.00032, max_iter=10e5)
net=linear_model.ElasticNetCV()


r_alphas ={'alpha': [.0001, .0003, .0005, .0007, .0009, 
          .01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 20, 30, 50, 60, 70, 80]}
r_alphas2={'alpha': np.arange(3, 10, 0.5)}
r_alphas3={'alpha': np.arange(5.5, 6.5, 0.1)}
ridge_CV=GridSearchCV(ridge,r_alphas3 ,cv=10)
ridge_CV.fit(X,y)
ridge_pred=ridge_CV.predict(X_test)
ridge_CV.score(X_test,y_test)
rmsle(y_test,ridge_pred)
print('Ridge CV bestcScore :'+str(ridge_CV.best_score_))
print('Ridge CV bestcParams :'+str(ridge_CV.best_params_))
ridge_CV.cv_results_

l_alphas={'alpha':[0.000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1]}
l_alphas1={'alpha':np.arange(0.0003,0.0004,0.00001)}
lasso_CV=GridSearchCV(lasso,l_alphas1 ,cv=10)
lasso_CV.fit(X,y)
lasso_pred=lasso_CV.predict(X_test)
lasso_CV.score(X_test,y_test)
rmsle(y_test,lasso_pred)
print('Lasso CV bestcScore :'+str(lasso_CV.best_score_))
print('Lasso_CV Best params: '+str(lasso_CV.best_params_))


net_ratio=[0.8, 0.85, 0.9, 0.95, 0.99, 1]
net_alphas=[0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
net=linear_model.ElasticNetCV(l1_ratio=net_ratio,alphas=net_alphas ,cv=10)
net.fit(X,y)
net_pred=net.predict(X_test)
net.score(X_test,y_test)
rmsle(y_test,net_pred)
print('Net CV bestcScore :'+str(net.best_score_))
print('Net CV Best params: '+str(net.best_params_))

a=ridge_pred*.35+lasso_pred*.45+net_pred*.2
rmsle(y_test,a)


agg=lasso_CV.predict(test)*0.4+ridge_CV.predict(test)*0.2+net.predict(test)*0.4


#%%
# =============================================================================
# 
# sumitting File to competition
# selected model - Ridge regressor alpha=6.1
# =============================================================================
test=test.drop('SalePrice',axis=1)
SalePrice=averaged_models.predict(test)
SalePrice=np.expm1(SalePrice)
ID=pd.Series(test.index)
ID
Submission=pd.DataFrame({'Id': test.index, 'SalePrice': SalePrice})
Submission.to_csv('submission.csv',index=False)

#%%
# =============================================================================
# expirements in enseemblemethods
# =============================================================================
scaler=preprocessing.RobustScaler()

#Validation function
n_folds = 5


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.00032, random_state=1))

Ridge = make_pipeline(RobustScaler(), Ridge(alpha=6.099999999999998, random_state=1))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


    

averaged_models = AveragingModels(models = (net, ridge_CV, lasso_CV,GBoost))
averaged_models.fit(X_train,y_train)
ave_pred=averaged_models.predict(X_test)
averaged_models.score(X_test,y_test)
rmsle(y_test,ave_pred)
score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

ave=(lasso.predict(X_test)*+Ridge.predict(X_test)+ENet.predict(X_test)+KRR.predict(X_test)+GBoost.predict(X_test))*.2


#%%
test=test.drop('SalePrice',axis=1)
SalePrice=averaged_models.predict(test)
SalePrice=np.expm1(SalePrice)
ID=pd.Series(test.index)
ID
# =============================================================================
# 
# sumitting File to competition
# selected model - averages models
# =============================================================================
Submission=pd.DataFrame({'Id': test.index, 'SalePrice': SalePrice})
Submission.to_csv('submission2.csv',index=False)
#%%
# =============================================================================
# 10.04 - best score so far (for submission file) - .11896 (846/4461)
# Outliers detection
# =============================================================================
#OLS stands for ordinary least squares. it analyzez the array and returns an outliers detection
ols = sm.OLS(endog = y, exog = X)
fit=ols.fit()

#There are several outlier test that are part o statmodels.OLS.outlier_test(), bonf(p) is the default.
#it anyalizes the probability of each indice in the array to not be an outlier, and returns the p-values
#we gonna stamp out each data point that it's probability of NOT being an outlier is smaller than alpha (0.01)
OE= fit.outlier_test()['bonf(p)']
liers_alpha=0.001
liers=OE[OE<liers_alpha]
drop_them=list(liers.index)
drop_them
train=train.drop(drop_them,axis=0)
y=y.drop(drop_them,axis=0)
#%%
#now we run the models again

X=train.drop('SalePrice',axis=1)
y=train.SalePrice
X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=42)


kfolds = KFold(n_splits=10, shuffle=True, random_state=23)
lm=linear_model.LinearRegression()
scaler=preprocessing.RobustScaler()
ridge=linear_model.Ridge(alpha=6.099999999999998)
lasso=linear_model.Lasso(alpha=0.00032, max_iter=10e5)
net=linear_model.ElasticNetCV()


r_alphas ={'alpha': [.0001, .0003, .0005, .0007, .0009, 
          .01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 20, 30, 50, 60, 70, 80]}
r_alphas2={'alpha': np.arange(3, 10, 0.5)}
r_alphas3={'alpha': np.arange(5.5, 6.5, 0.1)}
ridge_CV=GridSearchCV(ridge,r_alphas3 ,cv=10)
ridge_CV=ridge_CV.fit(X_train,y_train)
ridge_pred=ridge_CV.predict(X_test)
print('Ridge Regression score: '+str(ridge_CV.score(X_test,y_test)))
print('RIgge Regression root mean square error: '+str(rmsle(y_test,ridge_pred)))

l_alphas={'alpha':[0.000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1]}
l_alphas1={'alpha':np.arange(0.0003,0.0004,0.00001)}
lasso_CV=GridSearchCV(lasso,l_alphas1 ,cv=10)
lasso_CV=lasso_CV.fit(X_train,y_train)
lasso_pred=lasso_CV.predict(X_test)
print('Lasso Regression score: '+str(lasso_CV.score(X_test,y_test)))
print('Lasso Regression root mean square error: '+str(rmsle(y_test,lasso_pred)))


net_ratio=[0.8, 0.85, 0.9, 0.95, 0.99, 1]
net_alphas=[0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
net=linear_model.ElasticNetCV(l1_ratio=net_ratio,alphas=net_alphas ,cv=10)
net=net.fit(X_train,y_train)
net_pred=net.predict(X_test)
print('Elastic net score: '+str(net.score(X_test,y_test)))
print('Elastic net root mean square error: '+str(rmsle(y_test,net_pred)))

lasso_pred=pd.Series(lasso_pred)
net_pred=pd.Series(net_pred)
ridge_pred=pd.Series(ridge_pred)
X2=[ridge_pred,lasso_pred,net_pred] 
X2=pd.concat([ridge_pred,lasso_pred,net_pred],axis=1)
X2=X2.rename(columns={0:'lasso_pred',1:'net_pred',2:'ridge_pred'})



KRR.fit(X_train,y_train)
print('KRR score: '+str(KRR.score(X_test,y_test)))
krr_pred=KRR.predict(X_test)
print('KRR root mean square error: '+str(rmsle(y_test,krr_pred)))


GBoost=GBoost.fit(X_train,y_train)
print('GBoost root mean square error: '+str(GBoost.score(X_test,y_test)))
gboost_pred=GBoost.predict(X_test)
print('GBoost root mean square error: '+str(rmsle(y_test,gboost_pred)))

averaged_models = AveragingModels(models = (net, ridge_CV, lasso_CV,GBoost))
averaged_models.fit(X_train,y_train)
ave_pred=averaged_models.predict(X_test)
print('Model averages score: '+str(averaged_models.score(X_test,y_test)))
print('Model averages error: '+str(rmsle(y_test,ave_pred)))
meta_ave=ridge_pred*0.25+lasso_pred*.25+net_pred*.25+gboost_pred*.25
rmsle(y_test,meta_ave)

#%%
# =============================================================================
# 
# sumitting File to competition
# selected model - averages models
# ============================================================================
test=test.drop('SalePrice',axis=1)
SalePrice=averaged_models.predict(test)
SalePrice=np.expm1(SalePrice)
ID=pd.Series(test.index)
Submission=pd.DataFrame({'Id': test.index, 'SalePrice': SalePrice})
Submission.to_csv('submission_no_outliers.csv',index=False)
#sub_score=.11830 (808/4461)

SalePrice=ridge_CV.predict(test)
SalePrice=np.expm1(SalePrice)
Submission=pd.DataFrame({'Id': test.index, 'SalePrice': SalePrice})
Submission.to_csv('submission_no_outliers.csv',index=False)
#sub_score=.11790 (779/4461)

SalePrice=net.predict(test)
SalePrice=np.expm1(SalePrice)
Submission=pd.DataFrame({'Id': test.index, 'SalePrice': SalePrice})
Submission.to_csv('submission_no_outliers.csv',index=False)
#sub_score=.11912 

#averaged models with KRR and GBoost
test=test.drop('SalePrice',axis=1)
SalePrice=averaged_models.predict(test)
SalePrice=np.expm1(SalePrice)
ID=pd.Series(test.index)
Submission=pd.DataFrame({'Id': test.index, 'SalePrice': SalePrice})
Submission.to_csv('submission_no_outliers.csv',index=False)
#sub_score=.11994 WTF?!

#averaged models with GBoost and NO KRR
test=test.drop('SalePrice',axis=1)
SalePrice=averaged_models.predict(test)
SalePrice=np.expm1(SalePrice)
ID=pd.Series(test.index)
Submission=pd.DataFrame({'Id': test.index, 'SalePrice': SalePrice})
Submission.to_csv('submission_no_outliers.csv',index=False)
#sub_score=.11577 (581/4461)


#%%
# =============================================================================
# 11.04 - best score so far (for submission file) - .11577 (581/4461)
# XGBoost
# =============================================================================
xgbreg=xgb.XGBRegressor(learning_rate =0.01, n_estimators=3460, max_depth=3,
                     min_child_weight=0 ,gamma=0, subsample=0.7,
                     colsample_bytree=0.7,objective= 'reg:linear',
                     nthread=4,scale_pos_weight=1,seed=42, reg_alpha=0.00006)

#tuning hyperparmeters
xgbreg=xgb.XGBRegressor(objective= 'reg:linear',seed=42)
grid_params={
        'learning_rate':[0.01,0.05,0.1,0.3],
        'n_estimators':np.arange(1000,6000,1000),
        'max_depth':[2,3,4,5,6],
        'colsample_bytree':np.arange(0.5,1,0.1),
        'min_child_weight':[0,0.1,0.2,0.6],
        'gamma':[0,0.000001,0.00001,0.0001,.001,.01],
        'subsample':np.arange(0.5,1,.1),
        'reg_alpha':[0,0.000001,0.00001,0.0001,.001,.01]}
grid_search=GridSearchCV(cv=4,estimator=xgbreg,param_grid=grid_params,scoring='neg_mean_squared_error')
grid_search.fit(X_train,y_train)
print('best parms: '+grid_search.best_params_)
print('best score: '+grid_search.best_score_)
xgbreg=xgb.XGBRegressor(objective= 'reg:linear',seed=42)
xgbreg=xgbreg.fit(X_train,y_train)
xgbreg_pred=xgbreg.predict(X_test)
print('xgbreg score: '+str(xgbreg.score(X_test,y_test)))
print('xbgreg root mean square error ' +str(rmsle(y_test,xgbreg_pred)))

methods={'Ridge Regression':ridge_CV, 'Lasso Regression':lasso_CV,
         'Elastic net':net,'Gradient boosting':GBoost, 'Extreme gradient boosting':xgbreg,
         'All models average':averaged_models,'stacking regressor':stack}


scoring={}
for name,model in methods.items():
    if name=='stacking regressor':
        scoring[name]={'Score':model.score(np.array(X_test),np.array(y_test)),
                       'Root mean square error':rmsle(np.array(y_test),model.predict(np.array(X_test)))}
    else:
        scoring[name]={'Score':model.score(X_test,y_test),
                       'Root mean square error':rmsle(y_test,model.predict(X_test))}


averaged_models = AveragingModels(models = (ridge_CV,lasso_CV,net,GBoost,xgbreg))
averaged_models.fit(X_train,y_train)
ave_pred=averaged_models.predict(X_test)
print('Model averages score: '+str(averaged_models.score(X_test,y_test)))
print('Model averages error: '+str(rmsle(y_test,ave_pred)))
#%%
# =============================================================================
#     
# 14.04.19
# stacking with SVCREG
# =============================================================================
from mlxtend.regressor import StackingCVRegressor

stack = StackingCVRegressor(regressors=(ridge_CV,lasso_CV,net,GBoost,averaged_models,xgbreg),meta_regressor=xgbreg,
                               use_features_in_secondary=True)
stack = stack.fit(X_train,y_train)
stackX = np.array(X_train)
stacky = np.array(y_train)
stack = stack.fit(stackX, stacky)
stack_pred=stack.predict(np.array(X_test))
print('Stacked Model averages score: '+str(stack.score(np.array(X_test),np.array(y_test))))
print('Stacked Model averages error: '+str(rmsle(y_test,stack_pred)))

test=test.drop('SalePrice',axis=1)
SalePrice=stack.predict(np.array(test))
SalePrice=np.expm1(SalePrice)
ID=pd.Series(test.index)
Submission=pd.DataFrame({'Id': test.index, 'SalePrice': SalePrice})
Submission.to_csv('submission_stack_only.csv',index=False)

#%%
#blending
meta_ave=ridge_pred*0.13+lasso_pred*.13+net_pred*.13+gboost_pred*.13+xgbreg_pred*.13+stack_pred*.35
rmsle(y_test,meta_ave)

meta_ave2=(ridge_pred+lasso_pred+net_pred+gboost_pred+xgbreg_pred+stack_pred)/6
rmsle(y_test,meta_ave2)

#%%
final_pred=(ridge_CV.predict(test)*.13
            +lasso_CV.predict(test)*.13
            +net.predict(test)*.13
            +GBoost.predict(test)*.13
            +xgbreg.predict(test)*.13
            +stack.predict(np.array(test))*.35)

SalePrice=final_pred
SalePrice=np.expm1(SalePrice)
ID=pd.Series(test.index)
Submission=pd.DataFrame({'Id': test.index, 'SalePrice': SalePrice})
Submission.to_csv('submission_stack_all.csv',index=False)
#sub score: 0.11438 (303/4465, 94th percent)

final_pred=(ridge_CV.predict(test)
            +lasso_CV.predict(test)
            +net.predict(test)
            +GBoost.predict(test)
            +xgbreg.predict(test)
            +stack.predict(np.array(test)))/6

SalePrice=final_pred
SalePrice=np.expm1(SalePrice)
ID=pd.Series(test.index)
Submission=pd.DataFrame({'Id': test.index, 'SalePrice': SalePrice})
Submission.to_csv('submission_stack_all.csv',index=False)
#sub score: .11497, 359/4465 , 92nd percent!

#%%
# =============================================================================
# what will happen if i trained all the models on the entire train data?
# =============================================================================
ridge_CV=ridge_CV.fit(X,y)
lasso_CV=lasso_CV.fit(X,y)
net=net.fit(X,y)
GBoost=GBoost.fit(X,y)
xgbreg=xgbreg.fit(X,y)
stack=stack.fit(np.array(X),np.array(y))

scoring2={}
for name,model in methods.items():
    if name=='stacking regressor':
        scoring2[name]={'Score':model.score(np.array(X_test),np.array(y_test)),
                       'Root mean square error':rmsle(np.array(y_test),model.predict(np.array(X_test)))}
    else:
        scoring2[name]={'Score':model.score(X_test,y_test),
                       'Root mean square error':rmsle(y_test,model.predict(X_test))}
final_pred=(ridge_CV.predict(test)*.13
            +lasso_CV.predict(test)*.13
            +net.predict(test)*.13
            +GBoost.predict(test)*.13
            +xgbreg.predict(test)*.13
            +stack.predict(np.array(test))*.35)

SalePrice=final_pred
SalePrice=np.expm1(SalePrice)
ID=pd.Series(test.index)
Submission=pd.DataFrame({'Id': test.index, 'SalePrice': SalePrice})
Submission.to_csv('submission_stack_all.csv',index=False)
#sub score=.11606

final_pred=(ridge_CV.predict(test)
            +lasso_CV.predict(test)
            +net.predict(test)
            +GBoost.predict(test)
            +xgbreg.predict(test)
            +stack.predict(np.array(test)))/6

SalePrice=final_pred
SalePrice=np.expm1(SalePrice)
ID=pd.Series(test.index)
Submission=pd.DataFrame({'Id': test.index, 'SalePrice': SalePrice})
Submission.to_csv('submission_stack_all.csv',index=False)
#sub score=.11606

# =============================================================================
# ok ok just the CVREG
# =============================================================================
ridge_CV=ridge_CV.fit(X_train,y_train)
lasso_CV=lasso_CV.fit(X_train,y_train)
net=net.fit(X_train,y_train)
GBoost=GBoost.fit(X_train,y_train)
xgbreg=xgbreg.fit(X_train,y_train)
stack=stack.fit(np.array(X),np.array(y))

final_pred=(ridge_CV.predict(test)
            +lasso_CV.predict(test)
            +net.predict(test)
            +GBoost.predict(test)
            +xgbreg.predict(test)
            +stack.predict(np.array(test)))/6

SalePrice=final_pred
SalePrice=np.expm1(SalePrice)
ID=pd.Series(test.index)
Submission=pd.DataFrame({'Id': test.index, 'SalePrice': SalePrice})
Submission.to_csv('submission_stack_all.csv',index=False)
#sub score:.11483

final_pred=(ridge_CV.predict(test)*.13
            +lasso_CV.predict(test)*.13
            +net.predict(test)*.13
            +GBoost.predict(test)*.13
            +xgbreg.predict(test)*.13
            +stack.predict(np.array(test))*.35)

SalePrice=final_pred
SalePrice=np.expm1(SalePrice)
ID=pd.Series(test.index)
Submission=pd.DataFrame({'Id': test.index, 'SalePrice': SalePrice})
Submission.to_csv('submission_stack_all.csv',index=False)

