"""
Determine the Keywords of a document
Author : Ajay Halthor

"""

from keywords import extract_candidate_keywords
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.metrics.scores import accuracy, precision, recall, f_measure
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from random import shuffle

import collections
import os
import re
import nltk
import pprint
import time
import numpy as np

#Print features with nice indentation
pp = pprint.PrettyPrinter(indent=4)

#Extract features of a given word
def get_features(document,candidate,candidate_list, isKeyword):
    features = {} 
    features['length'] = len(candidate.split(' '))
    features['part_of_speech'] = ' '.join([pos for word,pos in nltk.pos_tag(nltk.word_tokenize(candidate))])

    position_list = [ m.start()/float(len(document)) for m in re.finditer(re.escape(candidate),document,flags=re.IGNORECASE)] 

    if len(position_list):
        for i in range(0,len(position_list)):
            features[str(i) + 'th occrrence'] = position_list[i]

    N = len(document)
    #Line approximation

    if len(position_list):
        y = 0
        for pos in position_list:
            #x = document.index(candidate)
            x = pos*N
            if x >= float(N)/2:
                y += 2/float(N) * x - 1
            else : 
                y += -2/float(N) * x + 1

        y /= len(position_list)
    else:
        x = 0
        y = 0


    features['line_position'] = y

    #Feature: Try the same approximation with parabola
    if len(position_list):
        y = 0
        for pos in position_list:
                x = pos*N
                y += ((x - float(N)/2)**2 )/ ((float(N)**2) / 4)
        y /= len(position_list)
    else:
        x = 0
        y = 0

    features['parabolic_position'] = y


    #Feature: Standard deviation
    if len(position_list):
        avg_position = sum([pos*float(len(document)) for pos in position_list])/float(len(position_list))
        ss = 0.0
        for p in position_list:
            ss += (p*float(len(document))  - avg_position)**2
        ss /= len(position_list)
        ss = ss**0.5
        features['Standard deviation'] = ss/float(len(document))


    #Feature: Text Frequency
    features['frequency'] = len(position_list)/ float(len(set(candidate_list))) #check


    pp.pprint(features)
    print 'for the top : ' + str(isKeyword)
    return features


# Determine Precision, Recall & F-Measure
def accuracy_measure(classifier,cross_valid_set):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(cross_valid_set):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    print 'pos Precision:', precision(refsets[1], testsets[1])
    print 'pos Recall:', recall(refsets[1], testsets[1])
    print 'pos F-measure:', f_measure(refsets[1], testsets[1])
    print 'neg Precision:', precision(refsets[0], testsets[0])
    print 'neg Recall:', recall(refsets[0], testsets[0])
    print 'neg F-measure:', f_measure(refsets[0], testsets[0])



doc_path = 'wiki20/documents/'
team_path = 'wiki20/teams/'

#Initialize data and target lists
target = []
data = []
feature_list = []

start_time = time.time()
for doc in os.listdir(doc_path):
    doc_name = doc.split('.')[0]

    if doc.endswith('txt'):
        document = open(doc_path + doc,"r").read()
        document = unicode(document + doc,errors='replace')

        #Get List of candidates
        candidates = extract_candidate_keywords(document)

        # #Initialize data and target lists
        print 'Now going to ',doc
        keyword_list = []
        for team in os.listdir(team_path):

            if team.startswith('team'):

                keywords = open(team_path + team + '/' + doc_name + '.key',"r").read()
                keywords = unicode(keywords + team + '/' + doc_name + '.key',errors='replace')

                #Take list of Keywords 
                keyword_list.extend([line.split(':')[1].lower().strip() for line in keywords.splitlines() if ':' in line ])             

        feature_list.extend([ (get_features(document,key,candidates,1),1) for key in set(keyword_list)])
        feature_list.extend([ (get_features(document,key,candidates,0),0) for key in candidates if key not in set(keyword_list)])

        print len(feature_list)

end_time = time.time()

shuffle(feature_list)

good_list_train = [(x,y) for x,y in feature_list if y == 1][:500]
good_list_test = [(x,y) for x,y in feature_list if y == 1][500:]
bad_list_train = [(x,y) for x,y in feature_list if y == 0][:500]
bad_list_test = [(x,y) for x,y in feature_list if y == 0][500:]

shuffle(good_list_train)
shuffle(bad_list_train)
shuffle(good_list_test)
shuffle(bad_list_test)

print 'number of keywords', len([x for x,y in feature_list if y == 1])
print 'number of non keywords', len([x for x,y in feature_list if y == 0])

training_set = []
training_set.extend(good_list_train)
training_set.extend(bad_list_train)
print 'all indicies of keywords training', len([x for x,y in training_set if y == 1])

testing_set = []
testing_set.extend(good_list_test)
testing_set.extend(bad_list_test)
print 'all indicies of keywords testing', len([x for x,y in testing_set if y == 1])

#Logistic Regression 
LogisticRegression_classifier =  SklearnClassifier(LogisticRegression()).train(training_set)
print ("LogisticRegression_classifier algorithm accuracy : ",nltk.classify.accuracy(LogisticRegression_classifier,testing_set) * 100)
accuracy_measure(LogisticRegression_classifier,testing_set)

#SVM
LinearSVC_classifier =  SklearnClassifier(LinearSVC()).train(training_set)
print ("LinearSVC_classifier algorithm accuracy : ",nltk.classify.accuracy(LinearSVC_classifier,testing_set) * 100)
accuracy_measure(LinearSVC_classifier,testing_set)

print 'Time of execution is %s seconds!' % (end_time - start_time)
