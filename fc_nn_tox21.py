""" Fully connected Neural Network fingerprints classifier for tox21 """
import os

import tensorflow as tf
import keras
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import keras.backend as K
import numpy as np

from utils.build_dataset import get_data
from utils.misc import set_up_logging, f1_score,plot_data, save_history

from sklearn.metrics import fbeta_score

mask_value = -1
MODEL_NAME = 'three_1layer_Dropout'
ml = False

# Set up logging
LOGGER = set_up_logging(__name__)

#architecture for a general multi-task classification
def fcnn_classifier_tox21(n_x, n_y):
    """
    This function returns a Fully Connected NN keras classifier

    :param n_x:     size of the input
    :param n_y:     size of the output
    :return:        keras untrained Fully Connected NN multi-class classifier
    """
    classifier = keras.Sequential([
        keras.layers.InputLayer(input_shape=(n_x,)),
        keras.layers.Dense(n_x, activation=tf.nn.relu),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(n_y, activation=tf.nn.sigmoid)
    ])

    classifier.compile(optimizer=tf.train.AdamOptimizer(),
                       loss='binary_crossentropy',
                       metrics=[f1_score])

    return classifier

#loss function for multi-task with missing label
def masked_loss_function(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    return K.binary_crossentropy(y_true * mask, y_pred * mask)

#architecture for multi-task with missing label
def fcnn_classifier_tox21_ml(n_x, n_y):
    """
    This function returns a Fully Connected NN keras classifier

    :param n_x:     size of the input
    :param n_y:     size of the output
    :return:        keras untrained Fully Connected NN multi-class classifier
    """
    classifier = keras.Sequential([
        keras.layers.InputLayer(input_shape=(n_x,)),
        keras.layers.Dense(n_x, activation=tf.nn.relu),
        keras.layers.Dense(int(n_x / 2), activation=tf.nn.relu),
        keras.layers.Dense(n_y, activation=tf.nn.sigmoid)

    ])

    classifier.compile(optimizer=tf.train.AdamOptimizer(),
                       loss=masked_loss_function,
                       metrics=[f1_score])

    return classifier

#print out the f1 score for each toxicological properties
def each_metric(input_data, y_test, model):

    classes = np.array(['NR-AR', 'NR-ER-LBD', 'SR-ATAD5'])
    y_pred = model.predict(input_data)

    # Performing masking
    y_pred = (y_pred > 0.5) * 1.0
    total = y_pred.shape[0]

    for i in range(y_pred.shape[1]):
        y_p = y_pred[:,i]
        y_t = y_test[:,i]
        f1 = fbeta_score(y_t, y_p, beta=1)
        print(classes[i], "f1 score:", f1)

    return


def main(train=False):
    """ Main function """
    # Get train and test dataset
    (x_train, y_train), (x_test, y_test) = get_data('data/tox21_10k_data_all_fingerprints.npz')

    #set different data path if run with missing labels
    if ml==True:
        (x_train, y_train), (x_test, y_test) = get_data('data/tox21_10k_data_all_fingerprints_multi.npz')

    n_x = x_train[0].shape[0]
    n_y = y_train[0].shape[0]

    #get the classifier
    fcnn_clf = fcnn_classifier_tox21(n_x, n_y)
    if ml == True:
        fcnn_clf = fcnn_classifier_tox21_ml(n_x, n_y)


    epochs = 50

    if train:
        # Train classifier
        print('\ntrain the classifier')

        # Define checkpoints
        # Create save weights checkpoint callback function
        weights_ckpt = keras.callbacks.ModelCheckpoint(
            'weights/%s_{epoch:d}.h5' % MODEL_NAME,
            save_weights_only=True,
            period=5
        )

        # Create save best weights checkpoint callback function
        best_ckpt = keras.callbacks.ModelCheckpoint(
            'weights/%s_best_f1_score.h5' % MODEL_NAME,
            monitor='val_f1_score',
            verbose=1,
            save_best_only=True,
            mode='max'
        )

        history = fcnn_clf.fit(x_train,
                               y_train,
                               epochs=epochs,
                               validation_split=0.1,
                               callbacks=[weights_ckpt, best_ckpt]  # Save weights
                               )

        # Get data from history
        metrics = ['f1_score', 'loss']
        save_history(history, "output/%s_%s_history.json" % (MODEL_NAME, epochs))
        plot_data(history, MODEL_NAME, epochs, metrics=metrics)


    else:
        # Load the model weights
        weights_file_path = os.path.abspath(os.path.join(os.curdir, 'weights/fcnn_tox21_%s.h5' % epochs))
        if not os.path.exists(weights_file_path):
            raise Exception(
                "The weights file path specified does not exists: %s"
                % os.path.exists(weights_file_path)
            )
        fcnn_clf.load_weights(weights_file_path)

    print('\ntest the classifier')

    print('\n#######################################')
    test_loss, test_f1_score = fcnn_clf.evaluate(x_test, y_test)

    print('\n#######################################')
    print('Test loss:', test_loss)
    print('Test F1 score:', test_f1_score)
    print('\n#######################################')
    each_metric(x_test, y_test, fcnn_clf)



if __name__ == '__main__':
    main(train=True)

