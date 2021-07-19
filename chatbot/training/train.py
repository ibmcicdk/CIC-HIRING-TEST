#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" A module that trains the chatbot model. """

import os
import json
import shutil
import pickle

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow import keras


def translate_relative_filepath(file_path):
    """ Changes a relative filepath to an absolute filepath. """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), file_path))


INPUT_FILE = translate_relative_filepath("input/intents.json")
DATA_DIR = translate_relative_filepath("data/")


def train():
    """ Trains the Bot based on the input and generates the models in data. """
    cleanup()
    json_input = load_input(INPUT_FILE)
    classes, word_set, documents = clean_input(json_input)
    save_input(classes, word_set)
    train_class, train_words = create_training_data(classes, word_set, documents)
    create_model(train_class, train_words)


def cleanup():
    """ Removes any old data to make sure nothing is accidentally left over from old trainings. """
    shutil.rmtree(DATA_DIR)
    os.mkdir(DATA_DIR)


def load_input(file_name):
    """ Load the json data from the input file. """
    with open(file_name, "r") as file_pointer:
        return json.load(file_pointer)


def save_input(classes, words):
    """ Saves the input for easier reading. """
    with open(os.path.join(DATA_DIR, "classes.pickle"), "wb") as file_pointer:
        pickle.dump(classes, file_pointer)
    with open(os.path.join(DATA_DIR, "words.pickle"), "wb") as file_pointer:
        pickle.dump(words, file_pointer)


def clean_sentence(sentence):
    """ Removes duplicate words, capital letters and special words. """
    ignore_words = ('?', '!', "'s")
    lemmatizer = WordNetLemmatizer()
    # Convert all words to lowercase to remove complexity
    words = nltk.word_tokenize(sentence.lower())
    # Lemmatize to remove complexity and remove duplicates and ignore words.
    words = set(map(lemmatizer.lemmatize, words)).difference(ignore_words)
    return words


def clean_input(json_input):
    """ Classifies, tokenizes and lemmatize the input. """
    # Ensure the required data has been downloaded.
    nltk.download('punkt')
    nltk.download('wordnet')
    classes = set()
    word_set = set()
    documents = list()
    for intent in json_input["intents"]:
        for pattern in intent["patterns"]:
            # Create a list of all words
            words = clean_sentence(pattern)
            word_set.update(words)
            # Create classes
            classes.add(intent["tag"])
            # Create documents
            documents.append((intent["tag"], words))
    # Order words and classes
    word_set = sorted(list(word_set))
    classes = sorted(list(classes))
    return classes, word_set, documents


def create_training_data(classes, word_set, documents):
    """ Creates the model and trains it. """
    # Crate numpy bool arrays of selectors for words and their tags.
    training_class = list()
    training_words = list()
    for tag, words in documents:
        word_bag = np.zeros(len(word_set), dtype=np.bool)
        for word in words:
            word_bag[word_set.index(word)] = True
        class_bag = np.zeros(len(classes), dtype=np.bool)
        class_bag[classes.index(tag)] = True
        training_class.append(class_bag)
        training_words.append(word_bag)
    training_class = np.array(training_class)
    training_words = np.array(training_words)
    return training_class, training_words


def create_model(train_class, train_words):
    """ Creates the neural net model.

    Notes:
        3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons

    """
    model = keras.Sequential([
        keras.layers.Dense(128, input_shape=(len(train_words[0]),), activation="relu", name="layer1"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation="relu", name="layer2"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(len(train_class[0]), activation="softmax", name="layer3"),
        ])
    # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # Fitting and saving the model
    hist = model.fit(train_words, train_class, epochs=200, batch_size=5, verbose=1)
    model.save(os.path.join(DATA_DIR, 'chatbot_model.h5'), hist)
