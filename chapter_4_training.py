# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 18:37:31 2019

@author: vipra
"""

#%%
import numpy as np
import autocomplete
import matplotlib.pyplot as plt
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)



#%%


#%%
X_b=np.c_[np.ones((100, 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
 

#%%
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
#%%
plt.plot(X,y,"b.")
plt.plot(X_new,y_predict,'r-')
plt.axis([0, 2, 0, 15])
plt.show()

#%%
from sklearn.linear_model import LinearRegression
liner_Reg=LinearRegression()
liner_Reg.fit(X,y)
liner_Reg.coef_,liner_Reg.intercept_
liner_Reg.predict(X_new)
#%%
eta = 0.1 # learning rate
n_iterations = 1000
m = 100
theta = np.random.randn(2,1)
print (theta)
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
print (theta)
#%%
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
#%%
lin_reg=LinearRegression()
lin_reg.fit(X_poly,y)
X_new=np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new) #transforming a new X in trained model
y_new = lin_reg.predict(X_new_poly) #predicting new variable
plt.plot(X,y,"b.")
plt.plot(X_new,y_new,"r-")


#%% 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-+", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)   # not shown in the book
    plt.xlabel("Training set size", fontsize=14) # not shown
    plt.ylabel("RMSE", fontsize=14) 
    
#lin=LinearRegression()
#plot_learning_curves(lin,X,y)



#%% using more complex model csince the previous model is underfitting
from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=8, include_bias=False)),
        ("lin_reg", LinearRegression()),
    ])
plot_learning_curves(polynomial_regression,X,y)

''' more degree leads to overfitting '''
#%% EARLY STOPPAGE IMPLIMENTATION
from sklearn.base import clone 
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
poly_scaler = Pipeline([
        ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
        ("std_scaler", StandardScaler())
    ])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)

sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,
                       penalty=None, learning_rate="constant", eta0=0.0005, random_state=42)
# =============================================================================
# =============================================================================
#  Note that with
#  warm_state=True, when the fit()
#  method is called it continues training where it left
#  warm_start
#  fit()
#  off, instead of restarting from scratch.
# =============================================================================
# =============================================================================
min_val,best_epch,value_er=[],[],[]
minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, y_train)  # continues where it left off
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val, y_val_predict)
    value_er.append(val_error)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        min_val.append(minimum_val_error)
        best_epch.append(best_epoch)
        best_epoch = epoch
        best_model = clone(sgd_reg)




#%%

from sklearn import datasets 

iris=datasets.load_iris()
X=iris['data'][:,3:]
y=(iris['target']==2).astype(np.int)

#%%


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver="lbfgs", random_state=42)
log_reg.fit(X, y)
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
#%%
y_proba = log_reg.predict_proba(X_new)
#%%
plt.plot(X_new, y_proba[:, 1], "r-", linewidth=2, label="Iris virginica")
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris virginica")
plt.xlabel("Petal width (cm)", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="center left", fontsize=14)
#%% predict predicts the class predict_proba predicts the probability of the classes
log_reg.predict_proba([[1.66],[2.4]]) 
log_reg.predict([[1.66],[2.4]])
#%%
#x[y probability[:,1] >0.5] give the 1 class prob >0.2 and [0] gives the first instance 
decision_boundary=X_new[y_proba[:, 1] >= 0.5][0]


#%%
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.int)

log_reg = LogisticRegression(solver="lbfgs", C=10**10, random_state=42)
log_reg.fit(X, y)

x0, x1 = np.meshgrid(
        np.linspace(2.9, 7, 500).reshape(-1, 1),
        np.linspace(0.8, 2.7, 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()] # ravel converts all instances in one column from row 1 to last
y_proba = log_reg.predict_proba(X_new)
plt.figure(figsize=(10, 4))
plt.plot(X[y==0, 0], X[y==0, 1], "bs") #x[y=0]petal length, x[y=1]petal width
plt.plot(X[y==1, 0], X[y==1, 1], "g^")
#%% SOFT MAX REGRESSION multiclass classification
X = iris["data"][:, (2, 3)] # petal length, petal width
y = iris["target"]

softmax_reg =LogisticRegression(multi_class='multinomial', solver= "lbfgs",C=10)
softmax_reg.fit(X,y)
softmax_reg.predict([[2,3]]) #2 predictors petal_length and petal width instances
softmax_reg.predict_proba([[2,3]]) #probabilites of 3 classes
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


