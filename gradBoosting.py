# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 11:46:48 2016

@author: 2
"""

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

def sigm(y_pred):
    y_new = np.zeros((y_pred.size,), dtype=np.float64)
    #print y_new
    for i,y in enumerate(y_pred):
        y_new[i] = 1/(1 + math.exp(-y))
        #print y_new
    return y_new


data = pd.read_csv('C:/Users/2/Downloads/gbm-data.csv')
y,X = np.array(data.Activity.values),np.array(data[data.columns[1:]].values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

original_params = {'n_estimators': 250, 'verbose': True, 'random_state': 241}

#plt.figure()
labels=list()
ansTest={'learning_rate=0.1': '',
 'learning_rate=0.2': '',
 'learning_rate=0.3': '',
 'learning_rate=0.5': '',
 'learning_rate=1': ''}
ansTrain={'learning_rate=0.1': '',
 'learning_rate=0.2': '',
 'learning_rate=0.3': '',
 'learning_rate=0.5': '',
 'learning_rate=1': ''}

for label, color, setting in [('learning_rate=1', 'orange',
                               {'learning_rate': 1.0}) ,
                              ('learning_rate=0.5', 'turquoise',
                               {'learning_rate': 0.5}),
                              ('learning_rate=0.3', 'gray',
                               {'learning_rate': 0.3}),
                              ('learning_rate=0.2', 'blue',
                               {'learning_rate': 0.2}),
                              ('learning_rate=0.1', 'magenta',
                               {'learning_rate': 0.1})]:
    params = dict(original_params)
    params.update(setting)
    clf = ensemble.GradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)
    
    test_dev = np.zeros((params['n_estimators'],), dtype=object)
    train_dev = np.zeros((params['n_estimators'],), dtype=object)
    
    for i,y_pred in enumerate(clf.staged_decision_function(X_test)):
        #print i, y_pred.size
        test_dev[i] = log_loss(y_test,sigm(y_pred))
    for i,y_pred in enumerate(clf.staged_decision_function(X_train)):
        train_dev[i] = log_loss(y_train,sigm(y_pred))
    
    #test_loss = log_loss(y_test,test_dev[0])
    #train_loss = log_loss(y_train,train_dev[0])
    ansTest[label] = test_dev
    ansTrain[label] = train_dev    
    
    labels.append(label)

norm1=pd.DataFrame(ansTest)
norm2=pd.DataFrame(ansTest)

plt.figure(1)
plt.plot(norm1);plt.legend(norm1.columns)
plt.figure(2)
plt.plot(norm2);plt.legend(norm2.columns)

rf = ensemble.RandomForestClassifier(n_estimators=50,random_state=241)
rf.fit(X_train,y_train)
log_loss(y_test,rf.predict_proba(X_test))
