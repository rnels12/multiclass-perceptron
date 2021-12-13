#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 10:37:14 2021

@author: Ryky Nelson

Main function: 
- get & process the data
- separate the data into the training and test sets
- call and feed training data to the perceptron
- measure the training perceptron against (sparred) test data
"""

import pandas as pd
import time 

from perceptron import perceptron
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    with  open("train.csv", 'r') as tr:
        data = pd.read_csv( tr )
        
    # Seprating data to the feature matrix and label vector
    Y = data[['label']].values.ravel()
    X = data.drop(['label'], axis=1)
    
    # divide data to traing and test sets
    Xtr, Xte, Ytr, Yte = train_test_split( X, Y, \
                                test_size=0.20, random_state=0)

    tic = time.perf_counter()
    
    # training the perceptron model
    per = perceptron()
    per.fit(Xtr, Ytr)
    
    toc = time.perf_counter()
    print(f"Train time for perceptron = {toc - tic:0.4f} seconds")

    # predicting from the perceptron model
    Ypred = per.predict(Xte)
    print(f"accuracy from perceptron = {(Yte == Ypred).mean() * 100:.2f}%" )

    tic = time.perf_counter()
    
    # training the random forest (RF) model
    rf = RandomForestClassifier()
    rf.fit(Xtr, Ytr)
    
    toc = time.perf_counter()
    print(f"Train time for RF = {toc - tic:0.4f} seconds")
    
    # predicting from the perceptron model
    Ypred = rf.predict(Xte)
    print(f"accuracy from RF = {(Yte == Ypred).mean() * 100:.2f}%" )

    

    
    
