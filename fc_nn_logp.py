""" Fully connected Neural Network fingerprints classifier for logP """
import os

from utils.build_dataset import get_data
from utils.misc import r_squared
import tensorflow as tf
import keras
import matplotlib.pyplot as plt


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
        keras.layers.Dense(n_y)
    ])

    classifier.compile(optimizer=tf.train.AdamOptimizer(),
                       loss='mse',
                       metrics=['mae', r_squared]
                       )

    return classifier


def main(train=False):
    """ Main function """
    # Get train and test dataset
    (x_train, y_train), (x_test, y_test) = get_data('data/ncidb_fingerprints.npz')

    n_x = x_train[0].shape[0]
    n_y = y_train[0].shape[0]

    # Build classifier
    fcnn_clf = fcnn_classifier_logp(n_x, n_y)

    epochs = 5

    if train:
        # Train classifier
        print('\ntrain the classifier')

        history = fcnn_clf.fit(x_train, y_train, epochs=epochs, validation_split=0.1)

        # Save weights
        fcnn_clf.save_weights('weights/fcnn_logp_%s.h5' % epochs)


        #Get data from history
        # Plot the mean_absolute_error
        print(history.history.keys())
        plt.plot(history.history['mean_absolute_error'])
        plt.plot(history.history['val_mean_absolute_error'])
        plt.title("model mean_absolute_error")
        plt.ylabel('mae')
        plt.xlabel('epoch')
        plt.xticks(range(0, epochs, 1))
        plt.legend(['train', 'val'], loc='upper left')
        #Save the plot
        plt.savefig("output/fcnn_logp_mae_%s.png" % epochs)
        plt.show()

        #Plot the R2
        plt.plot(history.history['r_squared'])
        plt.plot(history.history['val_r_squared'])
        plt.title("model R2")
        plt.ylabel('R2')
        plt.xlabel('epoch')
        plt.xticks(range(0, epochs, 1))
        plt.legend(['train', 'val'], loc='upper left')
        #Save the plot
        plt.savefig("output/fcnn_logp_r2_%s.png" % epochs)
        plt.show()

        #Plot the loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.xticks(range(0, epochs, 1))
        plt.legend(['train', 'val'], loc='upper left')
        #Save the plot
        plt.savefig("output/fcnn_logp_loss_%s.png" % epochs)
        plt.show()
    else:
        # Load the model weights
        weights_file_path = os.path.abspath(os.path.join(os.curdir, 'weights/fcnn_logp_%s.h5' % epochs))
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
    main(train=True)

