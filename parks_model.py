
import pdb
import dla_cnn
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from keras.layers import Dropout, BatchNormalization
from keras.models import model_from_json


# Define custom loss
def mse_mask():
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        return K.mean(mask * K.square(y_pred - y_true), axis=-1)
    # Return a function
    return loss


# load the Parks et al. (2018) model
def load_model(verbose=1):
    parks_dir = '/'.join(dla_cnn.__file__.split('/')[:-1]) + '/models/'
    #parks_wght = r'model_gensample_v7.1.ckpt.data-00000-of-00001'
    #parks_meta = r'model_gensample_v7.1.ckpt.index'
    #parks_json = r'model_gensample_v7.1_hyperparams.json'
    parks_wght = r'model_gensample_v4.3.ckpt'
    parks_meta = parks_wght+'.meta'
    parks_json = r'model_gensample_v4.3_hyperparams.json'

    # load json and create model
    # json_file = open(parks_dir+parks_json, 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)

    # start tensorflow session
    with tf.Session() as sess:

        # import graph
        saver = tf.train.import_meta_graph(parks_dir+parks_meta)

        # load weights for graph
        saver.restore(sess, parks_dir+parks_wght)

        # get all global variables (including model variables)
        vars_global = tf.global_variables()

        # get their name and value and put them into dictionary
        sess.as_default()
        model_vars = {}
        for var in vars_global:
            try:
                model_vars[var.name] = var.eval()
            except:
                print("For var={}, an exception occurred".format(var.name))
    pdb.set_trace()

    # Build the Parks et al. (2018) model


    inputs = []
    concat_arr = []
    for ll in range(nHIwav):
        inputs.append(Input(shape=(spec_len, nHIwav), name='Ly{0:d}'.format(ll+1)))
        conv11 = Conv1D(filters=128, kernel_size=3, activation='relu')(inputs[-1])
        pool11 = MaxPooling1D(pool_size=3)(conv11)
        norm11 = BatchNormalization()(pool11)
        conv12 = Conv1D(filters=128, kernel_size=5, activation='relu')(norm11)
        pool12 = MaxPooling1D(pool_size=3)(conv12)
        norm12 = BatchNormalization()(pool12)
#        conv13 = Conv1D(filters=128, kernel_size=16, activation='relu')(norm12)
#        pool13 = MaxPooling1D(pool_size=2)(conv13)
#        norm13 = BatchNormalization()(pool13)
        concat_arr.append(Flatten()(norm12))
    # merge input models
    merge = concatenate(concat_arr)
    # interpretation model
    #hidden2 = Dense(100, activation='relu')(hidden1)
    fullcon = Dense(300, activation='relu')(merge)
    ID_output = Dense(1+nHIwav, activation='softmax', name='ID_output')(fullcon)
    N_output = Dense(1, activation='linear', name='N_output')(fullcon)
    z_output = Dense(1, activation='linear', name='z_output')(fullcon)
    model = Model(inputs=inputs, outputs=[ID_output, N_output, z_output, b_output])
    # Summarize layers
    print(model.summary())
    # Plot graph
    plot_model(model, to_file='cnn_find_model.png')
    # Compile
    loss = {'ID_output': 'categorical_crossentropy',
            'N_output': mse_mask(),
            'z_output': mse_mask()}
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

    # Now set the Parks et al. (2018) weights

    return model


# Load the parks model
if __name__ == '__main__':
    model = load_model()
