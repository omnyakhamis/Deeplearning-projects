import os
import tensorflow as tf

Model_path = os.path.dirname(os.path.realpath(__file__)) + '/saved_model/cat_dog_blog.h5'


def make_prediction(input_data):
    model = tf.keras.models.load_model(Model_path)
    result = model.predict(input_data)
    return result
