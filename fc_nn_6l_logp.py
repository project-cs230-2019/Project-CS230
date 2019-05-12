""" Fully connected Neural Network fingerprints classifier for logP """
import os
from sys import platform
import json

from utils.build_dataset import get_data
from utils.misc import r_squared
import tensorflow as tf
import keras
import matplotlib
if platform == "darwin":  # OS X
    matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


MODEL_NAME = 'fcnn_logp_6l'


def fcnn_classifier_logp(n_x, n_y):
    """
    This function returns a Fully Connected NN keras classifier

    :param n_x:     size of the input
    :param n_y:     size of the output
    :return:        keras untrained Fully Connected NN linear regression classifier
    """

    classifier = keras.Sequential([
        keras.layers.InputLayer(input_shape=(n_x,)),
        keras.layers.Dense(n_x, activation=tf.nn.relu),
        keras.layers.Dense(int(n_x/2), activation=tf.nn.relu),
        keras.layers.Dense(int(n_x/8), activation=tf.nn.relu),
        keras.layers.Dense(int(n_x/4), activation=tf.nn.relu),
        keras.layers.Dense(int(n_x/16), activation=tf.nn.relu),
        keras.layers.Dense(n_y)
    ])

    classifier.compile(optimizer=tf.train.AdamOptimizer(),
                       loss='mse',
                       metrics=['mae', r_squared]
                       )

    return classifier


def save_history(history, filepath):
    with open(filepath, 'w') as fp:
        json.dump(history.history, fp)


def plot_data(history, model_name, epochs, metrics):
    # Get data from history
    print(history.history.keys())
    # Plot the mean_absolute_error
    for metric in metrics:
        plt.plot(history.history[metric])
        plt.plot(history.history['val_%s' % metric])
        plt.title("model %s" % metric)
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        # Save the plot
        plt.savefig("output/%s_%s_%s.png" % (model_name, metric, epochs))
        plt.show()


def main(train=False):
    """ Main function """
    # Get train and test dataset
    (x_train, y_train), (x_test, y_test) = get_data('data/ncidb_fingerprints.npz')

    n_x = x_train[0].shape[0]
    n_y = y_train[0].shape[0]

    # Build classifier
    fcnn_clf = fcnn_classifier_logp(n_x, n_y)

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
            'weights/%s_best_val_r2.h5' % MODEL_NAME,
            monitor='val_r_squared',
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

        #Get data from history
        metrics = ['mean_absolute_error', 'r_squared', 'loss']
        save_history(history, "output/%s_%s_history.json" % (MODEL_NAME, epochs))
        plot_data(history, MODEL_NAME, epochs, metrics=metrics)

    else:
        # Load the model weights
        weights_file_path = os.path.abspath(os.path.join(os.curdir, 'weights/%s_%s.h5' % (MODEL_NAME, epochs)))
        if not os.path.exists(weights_file_path):
            raise Exception(
                "The weights file path specified does not exists: %s"
                % os.path.exists(weights_file_path)
            )
        fcnn_clf.load_weights(weights_file_path)

    print('\ntest the classifier')
    test_loss, test_mae, test_r_squared = fcnn_clf.evaluate(x_test, y_test)

    print('\n#######################################')
    print('Test loss:', test_loss)
    print('Test mae:', test_mae)
    print('Test R2:', test_r_squared)


if __name__ == '__main__':
    main(train=False)

