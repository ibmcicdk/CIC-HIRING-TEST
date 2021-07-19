#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" A WSGI launhcer of the flask app. """

import argparse
from chatbot.webapp.app import web_launch
from chatbot.training.train import train


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the Chatbot.")
    parser.add_argument("--train", help="Retrain the chatbot.", action='store_true', dest="train")
    parser.add_argument("--no-web", help="Skip launching the flask server.", action='store_true', dest="no_web")
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()
    if args["train"]:
        train()
    if not args["no_web"]:
        web_launch(port=8000, debug=True)
