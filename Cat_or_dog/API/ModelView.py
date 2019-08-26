# import the necessary packages

from flask import Blueprint
from keras.preprocessing.image import img_to_array

from PIL import Image
import numpy as np
import flask
import io
import tensorflow as tf
from .model.prediction import make_prediction
from keras.applications import imagenet_utils

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

from keras import backend as K


def check_input(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image


# create route for prediction
prediction_app = Blueprint('/Prediction', __name__)


@prediction_app.route('/', methods=['GET'])
def hello():
    return "Hello"


@prediction_app.route('/Prediction', methods=['POST'])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = check_input(image, target=(150, 150))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = make_prediction(image)
            K.clear_session()
            data["predictions"] = []
            if preds[0][0] == 1:
                prediction = 'dog'

            else:
                prediction = 'cat'
            data["predictions"].append(prediction)
            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


tf.keras.backend.clear_session


# create route for get request
@prediction_app.route('/health', methods=['GET'])
def check_health():
    return "Server running"
