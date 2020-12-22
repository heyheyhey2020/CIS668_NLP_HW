#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import numpy as np
import nltk
import string
import random
import collections
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentence_polarity
from sklearn.model_selection import train_test_split
from nltk.corpus import sentence_polarity
from nltk.metrics.scores import (precision, recall,f_measure)


# In[3]:


# Import data.
dataset = []
file1 = open('baby.txt')

info = []
for line in file1.readlines():
    if line in ['\n', '\r\n']:
        dataset.append(info)
        info = []
        continue
    l = line.split(':')
    if 'reviewText' in l[0]:
        info.append(l[1].split('\n')[0])
# print(dataset[0])
# print(type(dataset[0]))
# print(dataset[0][0])
# print(type(dataset[0][0]))


# In[4]:


### Pre-processing
def get_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

stop_words = list(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def preprocess(text):
    # Lower text
    t = text.lower()
    # Tokenize text 
    t = [w for w in t.split(' ')]
    # Remove stopwords, numbers and punctuations
    t = [w.strip(string.punctuation) for w in t if w not in stop_words and w.isdigit()!=1]
    t = [w for w in t if w != '']
    # lemmatize text and remove the single character
    pos_tags = pos_tag(t)
    t = [lemmatizer.lemmatize(w[0],get_pos(w[1])) for w in pos_tags if len(w[0])>1]
    # Combine them into a string
    #t = ' '.join(t)
    return t
# print(dataset[0][0])
# print(preprocess(dataset[0][0]))


# In[5]:


# Get the sentence corpus and look at some sentences
sentences = sentence_polarity.sents()
documents = []
for cat in sentence_polarity.categories():
    for sent in sentence_polarity.sents(categories = cat):
        documents.append((preprocess(' '.join(sent)),cat))
# print(documents)
# documents = [(sent,cat) for cat in sentence_polarity.categories() for sent in sentence_polarity.sents(categories=cat)]
random.shuffle(documents)
all_words_list = [word for (sent,cat) in documents for word in sent]
all_words = nltk.FreqDist(all_words_list)
# # get the 200 most frequently appearing keywords in the corpus
word_items = all_words.most_common(200)
# print(word_items)
word_features = [word for (word,count) in word_items]
# print(word_features)


# In[6]:


# Data pre-processing.
for i in range(len(dataset)):
    dataset[i] = preprocess(dataset[i][0])
print(dataset[0])
print(len(dataset))


# In[5]:


# Baseline model:
def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    return features
featuresets = [(document_features(d, word_features), c) for (d, c) in documents]

# Training using the NaiveBayes algorithm
train_set, test_set = train_test_split(featuresets,test_size =0.33, random_state = 42)
base_classifier = nltk.NaiveBayesClassifier.train(train_set)
print('Accuracy:',nltk.classify.accuracy(base_classifier, test_set))

# Evaluation
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
for i, (feats, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = base_classifier.classify(feats)
    testsets[observed].add(i)
print ('Precision:', nltk.precision(refsets['pos'], testsets['pos']))
print ('Recall:', nltk.recall(refsets['pos'], testsets['pos']))
print ('f_measure:', nltk.f_measure(refsets['pos'], testsets['pos']))


# In[6]:


## Using the lexicon file (Subjectivity) and define the feature
def readSubjectivity(path):
    flexicon = open(path, 'r')
    # initialize an empty dictionary
    sldict = { }
    for line in flexicon:
        fields = line.split()   # default is to split on whitespace
        # split each field on the '=' and keep the second part as the value
        strength = fields[0].split("=")[1]
        word = fields[2].split("=")[1]
        posTag = fields[3].split("=")[1]
        stemmed = fields[4].split("=")[1]
        polarity = fields[5].split("=")[1]
        if (stemmed == 'y'):
            isStemmed = True
        else:
            isStemmed = False
        # put a dictionary entry with the word as the keyword
        #     and a list of the other values
        sldict[word] = [strength, posTag, isStemmed, polarity]
    return sldict
SLpath = "subjclueslen1-HLTEMNLP05.tff"
SL = readSubjectivity(SLpath)


# In[7]:


# Adding other features.
negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']
def NOT_SL_features(document, word_features, negationwords, SL):
    features = {}
    document_words = set(document)
    for word in word_features:
        features['V_{}'.format(word)] = False
        features['V_NOT{}'.format(word)] = False
    # go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            features['V_NOT{}'.format(document[i])] = (document[i] in word_features)
        else:
            features['V_{}'.format(word)] = (word in word_features)
    # count variables for the 4 classes of subjectivity
    weakPos = 0
    strongPos = 0
    weakNeg = 0
    strongNeg = 0
    for word in document_words:
        if word in SL:
            strength, posTag, isStemmed, polarity = SL[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            if strength == 'strongsubj' and polarity == 'positive':
                strongPos += 1
            if strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            if strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
            features['positivecount'] = weakPos + (2 * strongPos)
            features['negativecount'] = weakNeg + (2 * strongNeg)
    return features
SL_featureset = [(NOT_SL_features(d, word_features,negationwords, SL), c) for (d, c) in documents]
# train_set, test_set = SL_featureset[100000:], SL_featureset[:100000]
train_set, test_set = train_test_split(SL_featureset, test_size=0.33, random_state=42)
new_classifier = nltk.NaiveBayesClassifier.train(train_set)
print('Accuracy:', nltk.classify.accuracy(new_classifier, test_set))
#classifier.show_most_informative_features(30)
# Evaluation
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
for i, (feats, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = new_classifier.classify(feats)
    testsets[observed].add(i)
print ('precision:', nltk.precision(refsets['pos'], testsets['pos']))
print ('recall:', nltk.recall(refsets['pos'], testsets['pos']))
print ('f_measure:', nltk.f_measure(refsets['pos'], testsets['pos']))


# In[8]:





# In[9]:


testfeatures = [(NOT_SL_features(d, word_features,negationwords, SL)) for d in dataset]


# In[10]:


new_classifier.classify(testfeatures[1])


# In[ ]:




