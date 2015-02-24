# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 13:11:23 2015

@author: fedor
"""

import pickle
import sklearn
from sklearn.naive_bayes import MultinomialNB 
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier 
import sklearn.feature_extraction
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.decomposition import TruncatedSVD
from getMorph import stem_corpus1
import random


#set random seed
random.seed(21)

#get texts from pickle file
texts = pickle.load( open( "texts_300.p", "rb" ) )
stem_texts, labels = stem_corpus1(texts)
#calculate tf-idf
tf_idf = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range = (1,2)).fit(stem_texts)
tf_idf_matrix = tf_idf.fit_transform(stem_texts)
X_train = tf_idf_matrix
X_train_dense = X_train.toarray()
Y_train = np.array([1 if label==u'pos' else -1 for label in labels])



#test linear classifier
clf = svm.LinearSVC()
scores = cross_validation.cross_val_score(clf, X_train, Y_train, cv = 20)
print("Linear Classifier:: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


clf.fit(X_train, Y_train)
coef =  clf.coef_
f_names = tf_idf.get_feature_names()
names = [y for (x, y) in sorted(zip(coef[0], f_names))]

print('Twenty most positive words: \n')
for name in names[-40:]:
    print name
print('Twenty most negative words: \n')
for name in names[:40]:
    print name


#test KNN classifier in euclid metric
clf = KNeighborsClassifier(n_neighbors=3)
scores = cross_validation.cross_val_score(clf, X_train_dense, Y_train, cv = 4)
print("KNN euclid:: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#test KNN classifier with cosine metric
clf = KNeighborsClassifier(n_neighbors=3, algorithm = 'brute', metric = 'cosine')
scores = cross_validation.cross_val_score(clf, X_train_dense, Y_train, cv = 4)
print("KNN cosine:: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



#test Naive Bayes classifier
clf = MultinomialNB()
scores = cross_validation.cross_val_score(clf, X_train, Y_train, cv = 5)
print("Naive Bayes:: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#calculate PCA projection

svd = TruncatedSVD(n_components=100, random_state=42)
X_train_pca = svd.fit_transform(X_train) 

#test Random forest
clf = RandomForestClassifier(n_estimators = 100)
scores = cross_validation.cross_val_score(clf, X_train_pca, Y_train, cv=5)
print("Random forest Classifier:: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#test GBM
clf = GradientBoostingClassifier()
scores = cross_validation.cross_val_score(clf, X_train_pca, Y_train, cv=5)
print("GBM Classifier:: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

