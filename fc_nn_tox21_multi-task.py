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
from utils.misc import set_up_logging, f1_score

num_classes = 3
mask_value = -1

# Set up logging
LOGGER = set_up_logging(__name__)

def multitask_loss(y_true, y_pred):
    # Avoid divide by 0
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # Multi-task loss
    return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))

def masked_loss_function(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    return K.binary_crossentropy(y_true * mask, y_pred * mask)

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
        keras.layers.Dense(int(n_x/2), activation=tf.nn.relu),
        keras.layers.Dense(int(n_x / 4), activation=tf.nn.relu),
        keras.layers.Dense(int(n_x / 8), activation=tf.nn.relu),
        keras.layers.Dense(n_y, activation=tf.nn.sigmoid)

    ])

    classifier.compile(optimizer=tf.train.AdamOptimizer(),
                       loss=masked_loss_function,
                       metrics=['binary_accuracy', f1_score])

    return classifier


def infer(input_data, y_test, model):

    classes = np.array(['NR-AR', 'NR-ER-LBD', 'SR-ATAD5'])

    y_pred = model.predict(input_data)

    # Performing masking
    y_pred = (y_pred > 0.5) * 1.0

    total = y_pred.shape[0]
    for i in range(y_pred.shape[1]):
        right = 0
        # select the indices
        # indices = np.where(y_pred[i] == y_test[i])[0]
        #
        # # Adding the results
        # inference.append(classes[indices].tolist())
        for j in range (total):
            if y_pred[j][i] == y_test[j][i]:
                right+=1
        accuracy = right/total
        print(classes[i], "accuracy:", accuracy)
    return accuracy

def main(train=False):
    """ Main function """
    # Get train and test dataset
    (x_train, y_train), (x_test, y_test) = get_data('data/tox21_10k_data_all_fingerprints_multi.npz')

    n_x = x_train[0].shape[0]
    n_y = y_train[0].shape[0]


    # Build classifier
    fcnn_clf = fcnn_classifier_tox21(n_x, n_y)

    epochs = 5

    if train:
        # Train classifier
        print('\ntrain the classifier')

        history = fcnn_clf.fit(x_train, y_train, epochs=epochs, validation_split=0.1)

        # Save weights
        fcnn_clf.save_weights('weights/fcnn_tox21_%s-one.h5' % epochs)


        # #Get data from history
        # print(history.history.keys())
        # plt.plot(history.history['acc'])
        # plt.plot(history.history['val_acc'])
        # plt.title("model accuracy")
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.xticks(range(0, epochs, 1))
        # plt.legend(['train', 'val'], loc='upper left')
        # plt.savefig("output/fcnn_tox21_acc_%s-one.png" % epochs)
        # plt.show()
        # #Save the plot

        #Get data from history
        print(history.history.keys())
        plt.plot(history.history['f1_score'])
        plt.plot(history.history['val_f1_score'])
        plt.title("model F1 score")
        plt.ylabel('F1 score')
        plt.xlabel('epoch')
        plt.xticks(range(0, epochs, 1))
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("output/fcnn_tox21_f1_%s-one.png" % epochs)
        plt.show()
        #Save the plot

        #Plot the loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.xticks(range(0, epochs, 1))
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("output/fcnn_tox21_loss_%s-one.png" % epochs)
        plt.show()
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
    test_loss, test_acc, test_f1_score = fcnn_clf.evaluate(x_test, y_test)

    print('\n#######################################')
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)
    print('Test F1 score:', test_f1_score)
    print('\n#######################################')
    infer(x_test, y_test, fcnn_clf)



if __name__ == '__main__':
    main(train=True)

