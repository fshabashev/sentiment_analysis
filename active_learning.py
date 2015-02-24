# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 02:48:37 2015

@author: fedor
"""
import numpy as np
import random 


class ActiveLearning:
    """This class stores information about fitted classification algorithm 
    and provides methods for selecting most important objects"""
    algorithm = None
    Y_train = np.array([])
    X = np.array([])
    fitted = False
    labeled_indexes = []
    
    def __init__(self, algorithm, X):
        self.algorithm = algorithm
        self.X = X 
        self.Y_train = np.array([None]*X.shape[0])
        self.fitted = False
    
    def refit(self, *args, **kwargs):
        self.algorithm.fit(self.X[self.labeled_indexes, :], self.Y_train[self.labeled_indexes], 
                               *args, **kwargs)
        self.fitted = True
        
    def predict(self, X_test):
        return self.algorithm.predict(X_test)
        
    def add_labels(self, indexes, labels):
        self.labeled_indexes = self.labeled_indexes + indexes
        self.Y_train[indexes] = labels
        self.refit()
        
    def request_labels(self, k):
        unlab_num = self.X.shape[0]
        unlab_ind = list(set(range(unlab_num)) - set(self.labeled_indexes))
        if ('predict_proba' in dir(self.algorithm)) and self.fitted:
            probs = self.algorithm.predict_proba(self.X)
            probs_diff = np.abs(probs[:,0] - probs[:,1])
            probs_diff[self.labeled_indexes] = 1
            requested_ind = probs_diff.argsort()[:k].tolist()
        elif self.fitted and ('decision_function' in dir(self.algorithm)):
            decisions = self.algorithm.decision_function(self.X)
            probs_diff = np.abs(decisions)
            probs_diff[self.labeled_indexes] = np.inf
            requested_ind = probs_diff.argsort()[:k].tolist()
        else:
            requested_ind = random.sample(unlab_ind, k)  
        return requested_ind
    
    def dummy_request_labels(self, k):
        unlab_num = self.X.shape[0]
        unlab_ind = list(set(range(unlab_num)) - set(self.labeled_indexes))
        requested_ind = random.sample(unlab_ind, k)  
        return requested_ind 
        
    def get_unlabeled_indexes(self):
        unlab_num = self.X.shape[0]
        return list(set(range(unlab_num)) - set(self.labeled_indexes))
            