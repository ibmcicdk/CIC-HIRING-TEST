#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" A WSGI launhcer of the flask app. """

import os
import json
import pickle
import random

import numpy as np
from tensorflow import keras
from flask import Flask, render_template, request

from chatbot.training.train import INPUT_FILE, DATA_DIR, clean_sentence


def load_model():
    """ Loads the chatbot model. """
    return keras.models.load_model(os.path.join(DATA_DIR, "chatbot_model.h5"))


def load_input():
    """ Loads the cleaned input. """
    with open(os.path.join(DATA_DIR, "classes.pickle"), "rb") as file_pointer:
        classes = pickle.load(file_pointer)
    with open(os.path.join(DATA_DIR, "words.pickle"), "rb") as file_pointer:
        words = pickle.load(file_pointer)
    with open(INPUT_FILE, "r") as file_pointer:
        intents = json.load(file_pointer)
    return classes, words, intents


def predict_class(model, classes, words, msg):
    """ Predicts which class the msg is in. """
    error_threshold = 0.75
    msg = clean_sentence(msg)
    word_bag = bag_sentence(words, msg)
    prediction = model.predict(np.array([word_bag]))[0]
    index = np.argmax(prediction)
    if prediction[index] > error_threshold:
        print("Classified:", msg, "as:", classes[index], "prob:", str(prediction[index]))
        return {"intent": classes[index], "probability": str(prediction[index])}
    else:
        print("Classification failed:", msg, "Closest match:", classes[index], "prob:", str(prediction[index]))
        return {}


def bag_sentence(words, msg):
    """ Bags the msg. """
    bag = np.zeros(len(words), dtype=np.bool)
    for word in msg:
        try:
            bag[words.index(word)] = True
        except ValueError:
            pass
    return bag


def get_response(predicted_class, intents):
    """ Finds the appropriate solution based on the predicted class topic. """
    no_answer_class = "noanswer"
    no_answer_response = "Sorry, i did not understand."
    answer = ""
    for intent in intents["intents"]:
        if intent["tag"] == no_answer_class:
            no_answer_response = random.choice(intent["responses"])
            if "intent" not in predicted_class:
                break
        if "intent" in predicted_class and intent["tag"] == predicted_class["intent"]:
            answer = random.choice(intent["responses"])
    if not answer:
        answer = no_answer_response
    return answer


def web_launch(port, debug):
    """ Loads the model and launches the webpage. """
    model = load_model()
    classes, words, intents = load_input()

    app = Flask(__name__)
    app.static_folder = 'static'

    @app.route("/")
    def home():
        """ The main index page. """
        return render_template("index.html")

    @app.route("/askbot")
    def get_bot_response():
        """ The bot question and response. """
        predicted_class = predict_class(model, classes, words, request.args.get('msg'))
        return get_response(predicted_class, intents)

    app.run(port=8000, debug=True)
