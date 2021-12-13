#!/usr/bin/env python3

"""
Created on Mon Dec 13 10:37:14 2021

@author: Ryky Nelson

multiclass Preceptron class:
trains the model, i.e. obtains the weight vectors for 10 (digit) classes 
that form the hyperplanes that separate the data
into 10 classes, i.e. the multiclass classifications
"""

import numpy as np

def delta(x):
    yv = [-1]*10
    yv[x] = 1
    return np.array(yv)

class perceptron:
    def __init__(self):
        self.w = np.array([])

    def fit(self, x, y):
        nrow, col = x.shape
        self.xtrain = np.concatenate( (  np.array( x ), np.ones((nrow,1)) ), axis=1 )
        self.ytrain = np.array( y )

        self.w = np.zeros( (10, (col + 1)) )

        for index, iy in enumerate(self.ytrain):
            ypred = np.argmax( np.dot( self.w, self.xtrain[index] ) )
            if iy != ypred:
                self.w[iy]    += self.xtrain[index]
                self.w[ypred] -= self.xtrain[index]           

    def predict(self, x):
        nrow, col = x.shape
        self.xtest = np.concatenate( (  np.array( x ), np.ones((nrow,1)) ), axis=1 )
        return np.argmax( np.dot( self.xtest, self.w.T ), axis=1 )
