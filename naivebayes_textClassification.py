# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 21:49:36 2018

@author: MANISH PATKAR
"""

import nltk
import random
from nltk.corpus import movie_reviews


Documents = [(list(movie_reviews.words(fileid)),category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]
                  
random.shuffle(Documents)

#print(Documents[1])  

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())
    

all_words = nltk.FreqDist(all_words)
#print(all_words.most_common(15))

word_featureset = list(all_words.keys())[:3000]

def find_feature(document):
    words = set(document)
    features ={}
    for w in word_featureset:
        features[w] = (w in words)
        
    return features

#print(find_feature(movie_reviews.words('neg/cv000_29416.txt')))

featuresets = [(find_feature(rev),category) for (rev , category) in Documents]

training_set = featuresets[:1900]
test_set =featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive bayes algo accuracy is :",(nltk.classify.accuracy(classifier, test_set))*100)

classifier.show_most_informative_features(15)




         
            
                                                       
        
        
        


