# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 13:11:23 2015

@author: fedor
"""

import re
import Stemmer
#import pymorphy2 
 
 
def string_to_words(string):
    return re.findall(r'(?u)\w+', string)
    
def stem_text(string):
    stemmer = Stemmer.Stemmer('ru')
    words = string_to_words(string)    
    return ' '.join(stemmer.stemWords(words)).lower()
    
def stem_corpus(corpus):
    """Stem each word in each document in given corpus"""
    stem_texts = []
    labels = []
    for text in corpus:
        label = text[2]
        stem_texts.append(text[1])               
        labels.append(label)
    return stem_texts, labels

def stem_corpus1(corpus):
    """Stem each word in each document in given corpus"""
    stem_texts = []
    labels = []
    for text in corpus:
        label = text[2]
        text1 = stem_text(text[1])
        stem_texts.append(text1)               
        labels.append(label)
    return stem_texts, labels


def stem_corpus2(corpus):
    """Stem each word in each document in given corpus"""
    stem_texts = []
    labels = []
    for text in corpus:
        label = text[2]
        text1 = stem_text(text[1]) + text[1]
        stem_texts.append(text1)                
        labels.append(label)
    return stem_texts, labels

