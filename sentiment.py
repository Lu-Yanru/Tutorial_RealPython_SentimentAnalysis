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
#from keras.backend import clear_session # for removing trained weights before retraining model

import matplotlib.pyplot as plt # for visualizing loss and accuracy for training and testing data

from sklearn.preprocessing import LabelEncoder # encode words into categorical interger values
from sklearn.preprocessing import OneHotEncoder # encode categorical values into one-hot encoded numeric array

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
# Accuracy: 0.776

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
# Accuracy for yelp data: 0.7760
# Accuracy for amazon data: 0.7920
# Accuracy for imdb data: 0.7326




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

# build a keras sequential model
# a linear stack of layers
input_dim = X_train.shape[1] # Number of features 2720

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

# compile the model which
# specifies the loss function as binary cross entropy and
# specifies the optimizer as adam
# add a list of metrics that can be later used for evaluation (do not influence training)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
    )
# prinnt an overview of the model and the number of parameters available for training
model.summary()
# hidden layer: 2720 features * 10 nodes = 27200 weights + 10 biases = 27210 parameters
# output layer: 10 nodes as input -> 10 weights + 1 bias = 11 parameters

# model training
# epochs specifies the number of iterations
# batch size specifies how many samples are used in one forward/backward pass
# -> increases the speed of the computation: needs fewer epochs, but more memory
# -> model may degrade with larger batch size
# verbose sets whether the training process is shown: 0 = silent, 1 = progess bar, 2 = one line per epoch
history = model.fit(X_train, y_train,
                    epochs=100,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10
                    )
# if rerun the training (.fit funtion),
# need to clear session to remove the already computed weights from the previous training
# clear_session()

# measure accuracy of the model
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training accuracy: {:.4f}".format(accuracy))
# Training accuracy: 1.0000 overfitted due to many epochs
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing accuracy: {:.4f}".format(accuracy))
# Testing accuracy: 0.7754 higher than logistic regression model

# visualize loss and accuracy for training and testing data based on the history callback
# the callback records the loss and additional metrics that can be added in the .fit() method
plt.style.use('ggplot')
# define a helper funtion
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) +1)
    
    plt.figure(figsize = (12,5))
    plt.subplot(1,2,1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
plot_history(history)
# sign of overfitting: accuracy reached 100%, the loss of the validation starts rising again
# usu separate testing and validation sets
# take the model with the highest balidation accuracy and then test the model with the testing set.



# represent words as dense vectors

# one-hot encoding
# take a vactor of the length of the vocabulary with an entry for each word in the corpus
# for each word a vector with zeros everywhere except for the spot for the word in the vocabulary, which is set to 1
# vector can be large for each word
# does not give any additional info like the relationship btw words
# usu used for categories/categorical features 
# which you cannot represent as a numeric value but still want to use it in ML
cities = ['London', 'Berlin', 'Berlin', 'New York', 'London']
# encode the list of cities into categorical interger values with LabelEncoder from scikit-learn
encoder = LabelEncoder()
city_labels = encoder.fit_transform(cities)
city_labels
# array([1, 0, 0, 2, 1])

# use OneHotEncoder from scikit-learn to encode the categorical values into a one-hot encoded numeric array
encoder = OneHotEncoder()
# reshape array above so that each categorical values to be in a separate row
# because that's what OneHotEncoder expects
city_labels = city_labels.reshape((5,1))
city_labels
#array([[1],
#       [0],
#       [0],
#       [2],
#       [1]])
city_arrays = encoder.fit_transform(city_labels)
# show the transformed result as an array
city_arrays.toarray()
#array([[0., 1., 0.],
#       [1., 0., 0.],
#       [1., 0., 0.],
#       [0., 0., 1.],
#       [0., 1., 0.]])



# word embeddings/dense word vectors
# map the statistical structure of the language used in the corpus
# aim to map semantic meaning into a geometric space: embedding space
# map semantically similar words close on the embedding space
# vector arithmetic: king-man+woman=queen
# how to obtain?
# option1: train word embedding during the training of the neural network
# option2: use pretrained word embeddings, can train them further

# prepare the text
