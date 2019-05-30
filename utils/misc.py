from sys import platform
import json
import logging
from rdkit import RDLogger
from keras import backend as K
import matplotlib
if platform == "darwin":  # OS X
    matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


def set_up_logging(logger_name):
    # Set up logging
    FORMAT = '%(asctime)s - %(levelname)s: %(message)s'
    logging.basicConfig(format=FORMAT)
    LOGGER = logging.getLogger(logger_name)
    LOGGER.setLevel(logging.DEBUG)

    # Set rdkit logger to critical
    rdlg = RDLogger.logger()
    rdlg.setLevel(RDLogger.CRITICAL)

    return LOGGER


def save_history(history, filepath):
    with open(filepath, 'w') as fp:
        json.dump(history.history, fp)


def plot_data(history, model_name, epochs, metrics, show=False):
    # Get data from history
    print(history.history.keys())
    # Plot the mean_absolute_error
    for i, metric in enumerate(metrics):
        fig = plt.figure(i)
        plt.plot(history.history[metric])
        plt.plot(history.history['val_%s' % metric])
        plt.title("model %s" % metric)
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        # Save the plot
        plt.savefig("output/%s_%s_%s.png" % (model_name, metric, epochs))
        if show:
            plt.show()
        plt.close(fig)


def r_squared(y_true, y_pred):
    """ Keras R2 metric function """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def precision(y_true, y_pred):
    """"Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """Calculates the F score, the weighted harmonic mean of precision and recall.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def f1_score(y_true, y_pred):
    """ Calculates the f1 score, the harmonic mean of precision and recall. """
    return fbeta_score(y_true, y_pred, beta=1)
