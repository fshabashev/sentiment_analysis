# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 07:19:20 2015

@author: fedor
"""

import sklearn
import numpy as np
import pickle
from getMorph import stem_corpus
from sklearn import svm
from sklearn import cross_validation
import sklearn.feature_extraction
import random 
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.ensemble import RandomForestClassifier
from active_learning import ActiveLearning
from sklearn.cross_validation import train_test_split
import pylab as plt


#set random seed
random.seed(123)

texts = pickle.load( open( "texts.p", "rb" ) )
stem_texts, labels = stem_corpus(texts)
#calculate tf-idf
tf_idf = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range = (1,2)).fit(stem_texts)
tf_idf_matrix = tf_idf.fit_transform(stem_texts)
X_train = tf_idf_matrix
X_train_dense = X_train.toarray()
Y_train = np.array([1 if label==u'pos' else -1 for label in labels])



#test linear classifier
#clf = svm.LinearSVC(C = 1)
#scores = cross_validation.cross_val_score(clf, X_train, Y_train, cv = 4)
#print("Linear Classifier:: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#svd = TruncatedSVD(n_components=100, random_state=42)
#X_train_pca = svd.fit_transform(X_train) 


a_train, a_test, b_train, b_test = train_test_split(X_train, Y_train, test_size=0.50, random_state=42)



#clf1 = RandomForestClassifier()
clf1 = svm.LinearSVC()
act_learner1 = ActiveLearning(clf1, a_train)


#clf2 = RandomForestClassifier()
clf2 = svm.LinearSVC()

act_learner2 = ActiveLearning(clf2, a_train)

#scores = cross_validation.cross_val_score(clf, X_train_pca, Y_train, cv=20)
#print("GBM Classifier:: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



values = []
#act_learner.fit(X_train, Y_train)

requested_first = random.sample(range(a_train.shape[0]), 20)
act_learner1.add_labels(requested_first, b_train[requested_first])
act_learner2.add_labels(requested_first, b_train[requested_first])


for t in range(200):
    requested_ind = act_learner1.request_labels(1)
    act_learner1.add_labels(requested_ind, b_train[requested_ind])
    cur_unlab1 = act_learner1.get_unlabeled_indexes()
    current_prediction1 = act_learner1.algorithm.predict(a_test)    
    accuracy1 = np.mean(current_prediction1 == b_test)

    requested_ind = act_learner2.dummy_request_labels(1)
    act_learner2.add_labels(requested_ind, b_train[requested_ind])
    cur_unlab2 = act_learner2.get_unlabeled_indexes()
    current_prediction2 = act_learner2.algorithm.predict(a_test)    
    accuracy2 = np.mean(current_prediction2 == b_test)
    values.append((accuracy1, accuracy2))
    print("Number of current iteration: ", t, ' Current accuracy 1: ', accuracy1, 
          'Current accuracy 2:', accuracy2)
plt.plot(np.array(values))