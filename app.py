from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob

#visual recognition
import json

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import urllib.request
from getword import *


with open('tokenizer.pkl', 'rb') as handle:
	tokenizer = pkl.load(handle)

# Define a flask app
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('base.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        feature=extract_features(file_path)
        model_sq = tf.keras.models.load_model('model_10.h5')
        caption=generate_desc(model_sq, tokenizer, feature, 34)
        caption=caption.split(' ')
        caption=' '.join(caption[1:-1])
        return caption
    return None


if __name__ == '__main__':
    app.run(debug=True)

