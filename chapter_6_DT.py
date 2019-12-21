# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#%%
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris=load_iris()
X=iris['data'][:,2:]
y=iris.target
tree_clf = DecisionTreeClassifier(max_depth=3) #remove vcriterion for gini
tree_clf.fit(X, y)
pred=tree_clf.predict(X)
#%%

from IPython.display import Image  
from sklearn.externals.six import StringIO  

from sklearn.tree import export_graphviz
import pydot

#%%

dot_data = StringIO()  
export_graphviz(tree_clf, out_file=dot_data,class_names=iris.target_names,feature_names=iris.feature_names[2:],filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())

#%%

tree_clf.predict_proba([[5,1.5]]) #1/3,2/3 probability of training instance in the particular leaf node of that class

#%% regression
import numpy as np
import autocomplete
import matplotlib.pyplot as plt
np.random.seed(42)
m = 200
X = np.random.rand(m, 1)
y = 4 * (X - 0.5) ** 2
y = y + np.random.randn(m, 1) / 10
#%%

#plt.plot(X,y,'o')
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(max_depth=4)
tree_reg.fit(X, y)


#%%
dot_data = StringIO()  
export_graphviz(tree_reg, out_file=dot_data,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())

#%%
pred=tree_reg.predict(X)

#%%
from sklearn.metrics import mean_squared_error
mean_squared_error(pred,y)

#%%




#%%


#%%




#%%


#%%




#%%


#%%




#%%


#%%




#%%


#%%




#%%


#%%




#%%


#%%




#%%


#%%


