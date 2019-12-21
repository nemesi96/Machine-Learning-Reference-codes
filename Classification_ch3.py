# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:53:42 2019

@author: vipra
"""
#%%
def plot_digit(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")
#%%
import random
import os
import tarfile
import urllib
import autocomplete
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
#%%
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

X,y=mnist['data'],mnist['target']
y=y.astype(np.uint8)

some_digit=X[0]
some_digit_image=some_digit.reshape(28,28)
plt.imshow(some_digit_image, cmap='binary')
plt.axis("off")
plt.show()

X_train,X_test,y_train,y_test=X[:60000],X[60000:],y[:60000],y[60000:]

y_train_5=(y_train==5)
y_test_5=(y_test==5)
#%%
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
#%%
pred=sgd_clf.predict(X_test)
sum(pred==y_test)*100/len(y_test)

#%% cross validation stratified sampling
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
scores=[]
skfolds = StratifiedKFold(n_splits=3, random_state=42)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    scores.append(n_correct / len(y_pred))

#%%
print(sum(scores)/len(scores),'/n')
#%%
from sklearn.model_selection import cross_val_score
scores=cross_val_score(sgd_clf,X_train,y_train_5,cv=3,scoring="accuracy")
scores.mean()
#%% cv_pred predicts the values of validationset
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
confusion_matrix(y_train_5,y_train_pred)
#%%
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
precision_score(y_train_5,y_train_pred)
recall_score(y_train_5,y_train_pred)
f1_score(y_train_5,y_train_pred)
#%% treshold scores
#y_scores=sgd_clf.decision_function(some_digit)

y_scores=cross_val_predict(sgd_clf,X_train,y_train_5,method='decision_function')
#%%
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5,y_scores)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16) # Not shown in the book
    plt.xlabel("Threshold", fontsize=16)        # Not shown
    plt.grid(True)                              # Not shown
    plt.axis([-50000, 50000, 0, 1])             # Not shown

plt.figure(figsize=(8, 4))                      # Not shown
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.plot([7813, 7813], [0., 0.9], "r:")         # Not shown
plt.plot([-50000, 7813], [0.9, 0.9], "r:")      # Not shown
plt.plot([-50000, 7813], [0.4368, 0.4368], "r:")# Not shown
plt.plot([7813], [0.9], "ro")                   # Not shown
plt.plot([7813], [0.4368], "ro")                # Not shown
  # Not shown
plt.show()
#%% y_scores-dcision scores
threshold_90_prec=thresholds[np.argmax(precisions>=0.90)]
y_train_pred_90=(y_scores>=threshold_90_prec)
#%%
precision_score(y_train_5,y_train_pred_90)
recall_score(y_train_5,y_train_pred_90)
#%%
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])                                    # Not shown in the book
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Not shown
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    # Not shown
    plt.grid(True)                                            # Not shown

plt.figure(figsize=(8, 6))                         # Not shown
plot_roc_curve(fpr, tpr)
plt.plot([4.837e-3, 4.837e-3], [0., 0.4368], "r:") # Not shown
plt.plot([0.0, 4.837e-3], [0.4368, 0.4368], "r:")  # Not shown
plt.plot([4.837e-3], [0.4368], "ro")               # Not shown
                        # Not shown
plt.show()

#%%
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)

#%%
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")
y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)
#%%
plt.plot(fpr,tpr,label="SGD")
plt.plot(fpr_forest,tpr_forest,label="Random forest")
plt.legend(loc='lower right')


#%%
y_pred_for=cross_val_predict(forest_clf,X_train,y_train_5,cv=3)
#%%
roc_auc_score(y_train_5,y_scores_forest)
precision_score(y_train_5,y_pred_for)
recall_score(y_train_5,y_pred_for)


#%% Multi class Classification
# from sklearn.svm import SVC
# svm_clf = SVC()
# #%%
# svm_clf.fit(X_train, y_train) # y_train, not y_train_5
# #%%
# svm_clf.predict([X[0:10]])

# #%% decisions scores for 5
# svm_clf.decision_function([some_digit])


# #%%
# from sklearn.multiclass import OneVsRestClassifier
# ovr_clf = OneVsRestClassifier(SVC())
# ovr_clf.fit(X_train, y_train)
# ovr_clf.predict([some_digit])
#%%
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
sgd_clf = SGDClassifier(random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cv_scores=cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")


#%%
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)



#%%

plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()


#%% errors
norm_cf=conf_mx/conf_mx.sum(axis=1, keepdims=True)
np.fill_diagonal(norm_cf,0)
norm_cf #errors proportion
plt.matshow(norm_cf, cmap=plt.cm.gray)
#%%
from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel=np.c_[y_train_large,y_train_odd] #column stack np.r_ row stack
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
#knn_clf.predict(X_test[1:10])

#%%
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")


#%%
from sklearn.neighbors import KNeighborsClassifier
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)
#%%

#%%



#%%




#%%




#%%




#%%








