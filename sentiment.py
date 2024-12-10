#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:59:11 2024

@author: luzi
"""
# Sentiment analysis tutorial Real Python with Keras TensorFlow
# https://realpython.com/python-keras-text-classification/

import pandas as pd # for reading csv files and making list of data frame
from sklearn.feature_extraction.text import CountVectorizer # for making bag of words
from sklearn.model_selection import train_test_split # for spliting data into training and testing data
from sklearn.linear_model import LogisticRegression # for building baseline model with logistic regression

# Load data set with Pandas
# data: sentences + sentiment label (1 positive, 0 negative)

# set up a dictionary of source and filepath with key:value pairs 
filepath_dict = {
    'yelp': '~/Desktop/NLP/SentimentAnalysis-RealPython/data/yelp_labelled.txt',
    'amazon': '~/Desktop/NLP/SentimentAnalysis-RealPython/data/amazon_cells_labelled.txt',
    'imdb': '~/Desktop/NLP/SentimentAnalysis-RealPython/data/imdb_labelled.txt'
    }
# set up an empty list of data frame
df_list = []
# for each source:filepath pair in the dict
for source, filepath in filepath_dict.items():
    # read the file specified by the path, the first part is called sentence, 
    # the second part is called label, they are separated by a tab
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    # add another column filled with the source name
    df['source'] = source 
    # append this data frame to the data frame list
    df_list.append(df)

#print(df)
#print(df_list)
# concatenate pandas objects        
df = pd.concat(df_list)
print(df.iloc[0]) # print the 1st object in the df list just to check


# create bag of words
# create feature vectors out of sentences: words and their frequencies

# small example
sentence = ['John likes ice cream', 'John hates chocolate']
# first create a vocabulary of the example sentences: each word with a index
vectorizer = CountVectorizer(min_df=0.0, lowercase=False)
vectorizer.fit(sentence)
vectorizer.vocabulary_
# transform sentences into arrays of the count of each word
vectorizer.transform(sentence).toarray()


# defining a baseline model to compare to the neural network model

# split the data into training and testing set 
# use the yelp data
df_yelp = df[df['source'] == 'yelp']
# create a Numpy array of sentences
sentences = df_yelp['sentence'].values
# create a Numpy array of labels
y = df_yelp['label'].values
# arguments: arrays = input data, output data (lists/Numpy arrays/pandas Dataframes, need to be of the same length), 
# test_size=25% of the dataset, 
# random_state=the obj that controls randomization during splitting, so that every time the randomized splitting is the same
# shuffle = default True, shuffle the dataset before splitting
# stratify = if not None, determines how to use a stratified split, e.g. stratify=y keeps the proportion of y values through the training and test sets, useful when using an imbalanced dataset
# outputs: training part of input array (x), testing part of the input array,
# training part of the output array (y), testing part of the output array
sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.25, random_state=1000)

# vectorize sentences
verctorizer = CountVectorizer()
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)
# 750 samples, each has 1938 dimensions (size of the vocabulary)
# sparse matrix: a data type that is optimized for matrices with only a few non-zero elements,
# which only keeps track of the non-zero elements reducing the memory load
X_train

# building baseline model with logistic regression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print("Accuracy:", score)

# try on all datasets
# loop through the datasets based on their sources
for source in df['source'].unique():
    df_source = df[df['source'] == source] # a list of the dataframes from the specific source in this loop
    sentences = df_source['sentence'] # a list of the sentences in the current list of dataframe
    y = df_source['label'].values # a list of the labels in the current dataframe
    # split the sentences and y into train and test datasets
    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.25, random_state=1000)
    # vectorize sentences, build bag of words
    vectorizer.fit(sentences_train)
    X_train = vectorizer.transform(sentences_train)
    X_test = vectorizer.transform(sentences_test)
    # fit data to logistic regression
    classifier.fit(X_train, y_train)
    # check accuracy of the regression model
    score = classifier.score(X_test, y_test)
    # print out the accuracy for the dataset
    # {} to insert the 1st variable (source)
    # {:.4f} to insert the 2nd variable (score) until the 4th decimal 
    print("Accuracy for {} data: {:.4f}".format(source, score))
    


# a neural network model
