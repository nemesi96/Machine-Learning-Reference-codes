
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#%%
import random
import os
import tarfile
import urllib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import autocomplete
import matplotlib.pyplot as plt

#%%
os.chdir('D:\\self learn\\books\\ML and DL\\handson-ml2-master\\datasets\\housing_1')
os.getcwd()

#%%
housing=pd.read_csv('housing.csv')
housing.describe()

#%%

housing.hist(bins=50, figsize=(20,15))
plt.show()

#%%
#Creating test case
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
from zlib import crc32
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32
def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

#%%
housing_with_id = housing.reset_index()
train_set, test_set = split_train_test(housing, 0.2)
#%%
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
#%%
housing["income_cat"] = pd.cut(housing["median_income"],
       bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
       labels=[1, 2, 3, 4, 5])
housing["income_cat"].hist()
#%%
#Stratified sampling
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index] #training index allocated
    strat_test_set = housing.loc[test_index]
#%%
strat_test_set["income_cat"].value_counts()/len(strat_test_set) #Counts the values
strat_train_set["income_cat"].value_counts()/len(strat_train_set)

#%% data is back to original state with stratified sampling
strat_test_set.drop("income_cat",axis=1,inplace=True)
strat_train_set.drop("income_cat",axis=1,inplace=True)

#%%



#%%
housing=strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1,figsize=(30,40))
#%%
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
s=housing["population"]/100, label="population", figsize=(12,7),
c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()

#%% #Correlation
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=True)
#%%
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
#%%
housing.plot(kind="scatter", x="median_income", y="median_house_value",figsize=(8,25))

#%%

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
#%%
housing=strat_train_set.drop('median_house_value',axis=1)
housing_labels=strat_train_set['median_house_value'].copy()

#%%
#housing_tr['total_bedrooms'].isna().sum()
#%% 
#DATA CLEANING
#housing.dropna(subset=["total_bedrooms"]) # Get rid of corresponding districts CCA
#housing.drop("total_bedrooms", axis=1) # get rid of entire attribute
#median = housing["total_bedrooms"].median() # option 3
#housing["total_bedrooms"].fillna(median, inplace=True)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
imputer.statistics_
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
index=housing_num.index)
#%%
housing_cat=housing[['ocean_proximity']] #2D Array
housing_cat.head(10)

'''encode string categoies in to numerical since ML recognices Numerical
'''
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder= OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)


#to view 
ordinal_encoder.categories_
housing_cat_encoded[:10]
#%%
''' Dummy variable representation'''
from sklearn.preprocessing import OneHotEncoder
cat_encoder=OneHotEncoder()
housing_cat_1hot=cat_encoder.fit_transform(housing_cat)
housing_cat_1hot #Scipy sparce matrix'''
print(cat_encoder.categories_)
housing_cat_1hot.toarray()
#%%
''' PIPLINE  for only Numerical Attriburtes'''
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)
#%% 
''' handles all the columns NUM and CAT'''
from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
    ])
housing_prepared = full_pipeline.fit_transform(housing)


#%% SVR trails
from sklearn.svm import SVR
trail=housing_prepared[:,[2,3,4,5,6,7]]
svm_linr=SVR(kernel='linear',epsilon=1000,gamma=200,C=1000)
#svm_rbf=SVR(kernel='rbf',gamma=0.1,C=10,epsilon=1.5)
#svm_poly=SVR(kernel='poly',degree=2,gamma=0.1,C=10,epsilon=1.5)
svm_linr.fit(housing_prepared,housing_labels)
#svm_rbf.fit(trail,housing_labels)
#svm_poly.fit(trail,housing_labels)

house_pred_linr=svm_linr.predict(housing_prepared)
#house_pred_rbf=svm_rbf.predict(trail)
#house_pred_poly=svm_poly.predict(trail)

from sklearn.metrics import mean_squared_error
lin_mse=np.sqrt(mean_squared_error(housing_labels,house_pred_linr))
#rbf_mse=np.sqrt(mean_squared_error(housing_labels,house_pred_rbf))
#poly_mse=np.sqrt(mean_squared_error(housing_labels,house_pred_poly))
lin_mse
#%%
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


#%%
some_data=housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared=full_pipeline.transform(some_data)
print("Predictions:",lin_reg.predict(some_data_prepared)) #gives predictions
print("Labels", list(some_labels))

#%%

#%%
from sklearn.metrics import mean_squared_error
housing_predictions=lin_reg.predict(housing_prepared)
lin_mse=mean_squared_error(housing_labels,housing_predictions)
lin_rmse=np.sqrt(lin_mse)
lin_rmse
#%%
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
#WHAT?????? 0.0 = overfitting


#%%
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores) 
scores_r = cross_val_score(tree_reg, housing_prepared, housing_labels,scoring="r2", cv=10)
tree_rmse_scores = np.sqrt(-scores)
'''Scikit-Learn’s cross-validation features expect a utility function (greater is better) rather than a cost function (lower is
better), so the scoring function is actually the opposite of the MSE (i.e., a negative value), which is why the preceding code
computes -scores before calculating the square root.'''
tree_rmse_scores

#%%
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
display_scores(tree_rmse_scores)
display_scores(scores_r)
#rmse is a cost function lower the better R2 is a utility function higher the better.
#%%
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
scoring="neg_mean_squared_error", cv=10)

lin_scores_r = cross_val_score(lin_reg, housing_prepared, housing_labels,
scoring="r2", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
display_scores(lin_scores_r)

#%%
#training score
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions=forest_reg.predict(housing_prepared)
forest_rmse=np.sqrt(mean_squared_error(housing_labels,housing_predictions))
print (forest_rmse)
#crossvalidation score
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
scoring="neg_mean_squared_error", cv=10)
forest_scores_r = cross_val_score(forest_reg, housing_prepared, housing_labels,
scoring="r2", cv=10)
forest_rmse_scores=np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
display_scores(forest_scores_r)
#%%
from sklearn.model_selection import GridSearchCV
param_grid = [
{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
# n_estimators= number of trees 
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
scoring='neg_mean_squared_error',
return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
# the grid search will exlplore 3*4 + 2*3 combinations  and 5 CV scores for each combination we gt 18 models 18*5=90 scores

#%%
print(grid_search.best_params_)
grid_search.best_estimator_
cvres = grid_search.cv_results_
model_scores={}
scores=[]
param_1=[]
#%%
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]): #combines 2 lists
    print(np.sqrt(-mean_score), params)

#%%
feature_importances = grid_search.best_estimator_.feature_importances_
print (feature_importances)
#%%
#importance of attributes
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

#%%
#Evaluting TEST SET
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test) #we use Transform notfit_transform
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse

#%%
#compute a 95% confidence interval for the generalization error using
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                        loc=squared_errors.mean(),
                        scale=stats.sem(squared_errors)))


