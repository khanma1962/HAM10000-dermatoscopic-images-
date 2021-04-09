from __future__ import division, print_function
# conding=utf-8
# tensorflow and keras
# import tensorflow as tf 
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# flask util
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import numpy as np 
import sys
import os
import glob
import re 

#https://stackoverflow.com/questions/65907365/tensorflow-not-creating-xla-devices-tf-xla-enable-xla-devices-not-set

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


# define flask app
app = Flask(__name__)

#model path
model_path = '../HAM10000_4_7.h5'

# load the trained model
model = load_model(model_path)
# model._make_predict_function() # this is for old TF

print('Model loaded. Check http://127.0.0.1:5000')

def model_predict(img_path, model):

    img = image.load_img(img_path, target_size = (254, 254))

    #preprocess the image
    x = image.img_to_array(img)
    
    #expand the dimension
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x, mode= 'caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])

def upload():
    if request.method == 'POST':
        # get the file from post request
        f = request.file['file']

        #save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'upload', secure_filename(f.filename))

        f.save(file_path)

        # predict the pict
        preds = model_predict(file_path, model)

        # process for human
        result = preds.argmax(axis=1)
        return result


if __name__ == '__main__':
    app.run(debug=True)










