#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:59:11 2024

@author: Yanru Lu
"""
# Sentiment analysis tutorial Real Python with Keras
# https://realpython.com/python-keras-text-classification/

import pandas as pd # for reading csv files and making list of data frame
from sklearn.feature_extraction.text import CountVectorizer # for making bag of words
from sklearn.model_selection import train_test_split # for spliting data into training and testing data
from sklearn.linear_model import LogisticRegression # for building baseline model with logistic regression

from keras.models import Sequential # for using Keras sequential model API
from keras import layers # for adding layers in Keras

# Load data set with Pandas
# data: sentences + sentiment label (1 positive, 0 negative)


# set up a dictionary of source and filepath with key:value pairs 
path = '~/Desktop/DataAnalysis/NLP/Tutorial_RealPython_SentimentAnalysis/data/'
filepath_dict = {
    'yelp': path+'yelp_labelled.txt',
    'amazon': path+'amazon_cells_labelled.txt',
    'imdb': path+'imdb_labelled.txt'
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

# o = f(sum(wa)+b)
# a: input nodes
# w: weight
# b: bias
# o: output nodes
# f: activation funtion
# "It is generally common to use 
# a rectified linear unit (ReLU) for hidden layers, 
# a sigmoid function for the output layer in a binary classification problem, 
# or a softmax function for the output layer of multi-class classification problems."

# calculate weight -> backpropagation
# use optimizer to reduce error determined by a loss function
# optimizer: most commonly Adam
# loss function: here  binary cross entropy used for binary classification problems

# build a keras model
input_dim = X_train.shape[1] # Number of features

model = Sequential()
# using tensorflow backend
# add a dense layer (hidden layer) to the model,
# which has 10 neurons, 
# num of features equal to the length of X_train, 
# and relu as the activation funtion
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
# add another dense layer (output layer) to the model,
# which has 1 neuron
# and sigmoid as the activation funtion
model.add(layers.Dense(1, activation='sigmoid'))