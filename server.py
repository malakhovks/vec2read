#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# load tempfile for temporary dir creation
import sys, os, tempfile
# load misc utils
import json
# import uuid
from werkzeug.utils import secure_filename
import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

# load libraries for string proccessing
import re, string

# load libraries for API proccessing
from flask import Flask, jsonify, flash, request, Response, redirect, url_for, abort, render_template

# A Flask extension for handling Cross Origin Resource Sharing (CORS), making cross-origin AJAX possible.
from flask_cors import CORS

import gensim

ALLOWED_EXTENSIONS = set(['txt', 'xlsx'])

__author__ = "Kyrylo Malakhov <malakhovks@nas.gov.ua> and Vitalii Velychko <aduisukr@gmail.com>"
__copyright__ = "Copyright (C) 2020 Kyrylo Malakhov <malakhovks@nas.gov.ua> and Vitalii Velychko <aduisukr@gmail.com>"

app = Flask(__name__)
CORS(app)

"""
Limited the maximum allowed payload to 16 megabytes.
If a larger file is transmitted, Flask will raise an RequestEntityTooLarge exception.
"""
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

"""
Set the secret key to some random bytes. Keep this really secret!
How to generate good secret keys.
A secret key should be as random as possible. Your operating system has ways to generate pretty random data based on a cryptographic random generator. Use the following command to quickly generate a value for Flask.secret_key (or SECRET_KEY):
$ python -c 'import os; print(os.urandom(16))'
b'_5#y2L"F4Q8z\n\xec]/'
"""
# app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.secret_key = os.urandom(42)

# Load and init word2vec model
model = gensim.models.KeyedVectors.load_word2vec_format('./models/fiction.lowercased.lemmatized.word2vec.300d')
model.init_sims(replace=True)

def getExistsWordsInModel(words):
    exists = []
    for word in words:
        if word in model.vocab:
            exists.append(word)
    return exists

@app.route('/')
def index():
    return Response(render_template('index.html'), mimetype='text/html')

@app.route('/word2vec/similarity', methods=['POST'])
def similarity():
    if not request.json or not 'word_1' in request.json or not 'word_2' in request.json:
        abort(400)
    try:
        cosine_similarity = model.similarity(request.json['word_1'], request.json['word_2'])
        return jsonify({"similarity": cosine_similarity.item()})
    except KeyError:
        return jsonify({"Error": {"KeyError": "One of the words is missing in the word2vec model"}})

@app.route('/word2vec/similar', methods=['POST'])
def find_similar():
    if not request.json or not 'word' in request.json:
        abort(400)
    n = 100
    try:
        return jsonify({"similar": model.most_similar(request.json['word'], topn=n)})
    except KeyError:
        return jsonify({"Error": {"KeyError": "Word " + request.json['word'] + " does not exist in the word2vec model" , "Word": request.json['word']}})

@app.route('/word2vec/center', methods=['POST'])
def find_lexical_cluster_center():
    if not request.json or not 'words' in request.json:
        abort(400)
    n = 100
    try:
        return jsonify({"center" : model.most_similar(positive=getExistsWordsInModel(request.json['words']), topn=n)})
    except KeyError:
        return jsonify({"Error": {"KeyError": "Some words does not exist in the word2vec model" , "Words": request.json['words']}})

if __name__ == '__main__':
    # default port = 5000
    # app.run(host = '0.0.0.0')
    app.run(host = '0.0.0.0', port=3000)