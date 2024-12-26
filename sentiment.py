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

from keras.layers import TextVectorization # tokenize the text into a format that can be used by the word embeddings
# https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/

import numpy as np # for creating embedding matrix that only contain words in our vocab

# for random search hyperparameters optimization
# keras.wrappers depricated
# from scikeras.wrappers import KerasClassifier
# Migrating from keras.wrappers.scikit_learn
# https://adriangb.com/scikeras/stable/migration.html
# from sklearn.model_selection import RandomizedSearchCV
# I cannot find how to install scikeras on anaconda, thus use KerasTuner instead
import keras_tuner
# tutorial:
# https://keras.io/keras_tuner/getting_started/
# documentation:
# https://keras.io/keras_tuner/api/





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
vectorizer = CountVectorizer()
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
# Accuracy: 0.796

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
# Accuracy for yelp data: 0.7960
# Accuracy for amazon data: 0.7960
# Accuracy for imdb data: 0.7487



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
input_dim = X_train.shape[1] # Number of features 2505

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
# vectorize text into a list of intergers
# each interger maps to a value in a dictionary that encodes the entire corpus
# keys in the dictionary are the vocabulary terms
max_tokens = 5000 # max vocab size
max_len = 100 # Sequence length to pad the outputs to, adds 4 zero indexes reserved for unknown words
# create the vectorizer
tokenizer = TextVectorization(
    max_tokens=max_tokens,
    standardize="lower_and_strip_punctuation", # lowercase text and strips punctuation
    split="whitespace", # splite on whitespace
    output_mode='int', # outputs interger indices
    output_sequence_length=max_len)
# create the vocab
tokenizer.adapt(sentences_train)
# vectorize sentences_train and sentences_test
X_train = tokenizer(sentences_train)
X_test = tokenizer(sentences_test)
print(sentences_train[2])
print(X_train[2])
# index ordered after the most common words in the text
# 0 index reserved for padding
# unknown words are indexed word_count+1
input_data = ["black white foo", "clever the dee"]
tokenizer(input_data)
# diff btw TextVectorization in keras and CountVectorization in scikitlearn:
# CountVectorizer stack vectors of word count, length of each vector is the size of the total corpus vocab
# With TextVectorization, the resulting vectors equal the length of each text, the numbers correspond to the values from the dict
embedding_dim = 50
voc = tokenizer.get_vocabulary() # the vocab
vocab_size = len(voc) +1 # length of the vocab, +1 for 0 index reserved for unknown words
# get a dict mapping words to their indices
word_index = dict(zip(voc, range(vocab_size)))

# keras embedding layer
# take the previously calculated intergers and maps them to a dense vector of embedding
# option 1: take the output of the embedding layer and plug it into a Dense layer
model = Sequential()
model.add(layers.Embedding(
    input_dim=vocab_size, # the size of the vocab
    output_dim=embedding_dim, # the size of the dense vector
    input_length=max_len # the length of the sequence
    ))
# add a flatten layer that prepared the sequential input for the dense layer
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
    )
# Build the model explicitly with the input shape
# for embedding, input_shape = (batch_size, sequence_length),
# none indicates any batch size
# sequence_length should match the expected input length for your sequences, i.e. the number of tokens in each sequence
model.build(input_shape=(None, max_len))
model.summary()
# fit and evaluate model
history = model.fit(X_train, y_train,
                    epochs=20,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Testing Accuracy: {:.4f}".format(accuracy))
# Training Accuracy: 1.0000
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Training Accuracy {:.4f}".format(accuracy))
# Training Accuracy 0.6417
plot_history(history)
# not a good way to work with sequential data
# for sequential data, focus on methods that look at local and sequential info instead of absolute positional info

# option 2:  using a MaxPooling1D/AveragePooling1D or a GlobalMaxPooling1D/GlobalAveragePooling1D layer after the embedding
# pooling layers are a way to reduce the size of (downsample) the incoming feature
# max pooling: take the max value of all features in the (size defined) pool for each feature dimension (more commonly used as it highlights large values)
# average pooling: take average
# global max/average pooling:  takes the max/avg of all features
model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim,
                           input_length=max_len))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.build(input_shape=(None, max_len))
model.summary()
# train and evaluate model
history = model.fit(X_train, y_train,
                    epochs=50,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
# Training Accuracy: 1.0000
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
# Testing Accuracy:  0.7487
plot_history(history)
# improves the model

# use pretrained word embeddings
# precomputed embeddings trained on a large corpus
# Word2Vec (Google):  neural networks, more accurate
# GloVe (Stanford NLP): co-occurrence mateix and matrix factorization, faster
# both have dimensionality reduction
# here: GloVe, each line represents a word followed by its vector as a stream of floats

# make a function to create embedding matrix that only contain the words in our vocab
def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1 # Add 1 because reserve 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix
# retrieve the embedding matrix
embedding_matrix = create_embedding_matrix(
    '~/Desktop/DataAnalysis/NLP/glove.6B/glove.6B.50d.txt',
    word_index, embedding_dim)
# calculate how many embeddings are non-zero
nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
nonzero_elements/vocab_size
# 0.9507313317936874

# build a model with a global max pooling layer
model = Sequential()
model.add(layers.Embedding(vocab_size,
                           embedding_dim,
                           weights=[embedding_matrix],
                           input_length=max_len,
                           trainable=False))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])
model.build(input_shape=(None, max_len))
model.summary()
# model fit and evaluation
history = model.fit(X_train, y_train,
                    epochs=50,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
# Training Accuracy: 0.7914
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
#Testing Accuracy:  0.7380
plot_history(history)

# if we allow the embedding to be trained additionally
model = Sequential()
model.add(layers.Embedding(vocab_size,
                           embedding_dim,
                           weights=[embedding_matrix],
                           input_length=max_len,
                           trainable=True))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])
model.build(input_shape=(None, max_len))
model.summary()
# model fit and evaluation
history = model.fit(X_train, y_train,
                    epochs=50,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
# Training Accuracy: 1.0000
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
# Testing Accuracy:  0.7861
plot_history(history)
# more effective





# Convolutional Neural Networks (CNN/convnets)
# specialized neural network that can detect specific patterns
# can extract features from images and use them in neural networks
# can be used for both image processing and sequence processing
# has hidden layers = convolutional layers consisting of multiple filters which are slid across one image and can detect specific features (edges, corners, textures etc.)
# math: convolution: take a patch of input features with the size of the filter kernel. With this patch, take the dot product of the multiplied weight of the filter.
# more convolutional layers can detect more complex patterns
# image: 2d matrix of numbers
# sequential data e.g. text: 1d convolution
embedding_dim = 100
model = Sequential()
model.add(layers.Embedding(
    vocab_size,
    embedding_dim,
    input_length=max_len
    ))
# add convolutional layer btw embedding layer and global max pool layer
# with 128 filters, a kernal size of 5 and the activation function of relu
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.build(input_shape=(None, max_len))
model.summary()
history = model.fit(X_train, y_train,
                    epochs=10,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
# Training Accuracy: 1.0000
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
# Testing Accuracy:  0.7540
plot_history(history)
# plateau at 80% might because:
# there are not enough training samples
# the data you have does not generalize well
# missing focus on tweaking the hyperparameters
# CNN work best with large training sets






# hyperparameters optimization
# parameters to adjust the models
# one populer method: grid seearch (most thorough but computationallz heavy)
# take lists of parameters,
# run the model with each parameter combination
# another common method: random search (here)
# take random combination of parameters

# cross-validation: a way to validate the model, take the whole data and separate it intp muiltiple testing and training data sets
# type1: k-fold
# the data set is partitioned into k equal sized sets， 1 for testing, the rest for training
# run k different runs, where each partition is once used as a testing set
# the hight k, the more accurate the model evaluation is, but the smaller each testing set

# a function that creates a keras model
# allow various parameters to be set
#def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
#    model = Sequential()
#    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
#    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
#    model.add(layers.GlobalMaxPooling1D())
#    model.add(layers.Dense(10, activation='relu'))
#    model.add(layers.Dense(1, activation='sigmoid'))
#    model.compile(optimizer='adam',
#                  loss='binary_crossentropy',
#                  metrics=['accuracy'])
#    model.build(input_shape=(None, maxlen))
#    return model

# define the parameter grid that zou want to use in training
# a dictionary with each parameters
#param_grid = dict(num_filters=[32,64,128],
#                  kernel_size=[3,5,7],
#                  vocab_size=[5000],
#                  embedding_dim=[50],
#                  maxlen=[100])



# define a keras model with hyperparameters to be tuned
#def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
#    model = Sequential()
#    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
#    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
#    model.add(layers.GlobalMaxPooling1D())
#    model.add(layers.Dense(10, activation='relu'))
#    model.add(layers.Dense(1, activation='sigmoid'))
#    model.compile(optimizer='adam',
#                  loss='binary_crossentropy',
#                  metrics=['accuracy'])
#    return model

#def build_model(hp):
#    model = create_model(
#        
#        )
#    return model


# run the random search
# 