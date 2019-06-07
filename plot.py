from sys import platform
import os
import json
import matplotlib
if platform == "darwin":  # OS X
    matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

from incep_resnet_compact_exp_logp import incep_model_logp
from utils.build_dataset import get_data


def l2_plot(metric='loss', loc='upper left'):
    for i, history_file in enumerate([
        'fcnn_logp_6l_50_history.json',
        'fcnn_logp_6l_l2reg00001_50_history.json',
        'fcnn_logp_6l_l2reg0001_50_history.json',
        'fcnn_logp_6l_l2reg001_50_history.json',
    ]):
        with open(os.path.join('..','output', history_file), 'r') as hf:
            history = json.load(hf)

        plt.plot(history[metric], color='C%s' % i)
        plt.plot(history['val_%s' % metric], '--', color='C%s' % i)

    plt.title("model (L2 regularization) %s" % metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(
        ['train ($\lambda = 0$)',
         'val ($\lambda = 0$)',
         'train ($\lambda = 10^{-5}$)',
         'val ($\lambda = 10^{-5}$)',
         'train ($\lambda = 10^{-4}$)',
         'val ($\lambda = 10^{-4}$)',
         'train ($\lambda = 10^{-3}$)',
         'val ($\lambda = 10^{-3}$)',
        ],
        loc=loc,
        prop={'size': 8}
        )
    # Save the plot
    # plt.savefig("output/%s_%s_%s.png" % (model_name, metric, epochs))
    plt.show()


def layers_plot(metric='loss', loc='upper left'):
    for i, history_file in enumerate([
        'fcnn_logp_50_history.json',
        'fcnn_logp_4l_50_history.json',
        'fcnn_logp_6l_50_history.json',
    ]):
        with open(os.path.join('..','output', history_file), 'r') as hf:
            history = json.load(hf)

        plt.plot(history[metric], color='C%s' % i)
        plt.plot(history['val_%s' % metric], '--', color='C%s' % i)

    plt.title("model %s" % metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(
        ['train 2L',
         'val 2L',
         'train 4L',
         'val 4L',
         'train 6L',
         'val 6L'
        ],
        loc=loc
        )
    # Save the plot
    # plt.savefig("output/%s_%s_%s.png" % (model_name, metric, epochs))
    plt.show()


def transfer_learning_plot(metric='loss', loc='upper left'):
    for i, history_file in enumerate([
        'fcnn_exp_logp_6l_50_history.json',
        'fcnn_exp_logp_6l_trsf_lrng_50_history.json',
    ]):
        with open(os.path.join('..','output', history_file), 'r') as hf:
            history = json.load(hf)

        plt.plot(history[metric], color='C%s' % i)
        plt.plot(history['val_%s' % metric], '--', color='C%s' % i)

    plt.title("model %s" % metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(
        ['train',
         'val',
         'train (transf learn)',
         'val (transf learn)',
        ],
        loc=loc,
        )
    # Save the plot
    # plt.savefig("output/%s_%s_%s.png" % (model_name, metric, epochs))
    plt.show()


def plot_2Dimg_logp_predictions():
    # Load LogP data
    print('Loading data')
    (_, _), (x_test, y_test) = get_data('../data/ncidb_2Dimg.npz')

    # Load Experimental LogP data
    print('Loading data')
    (_, _), (exp_x_test, exp_y_test) = get_data('../data/ncidb_experim_data_2Dimg.npz', split=0.2)

    print('Normalize input dividing it by 255')
    # Input data normalization
    # Transform all input matrix elements in values belonging to [0,1] interval
    x_test /= 255
    exp_x_test /= 255

    # Get input and output dims
    n_h, n_w, n_c = x_test[0].shape
    n_y = y_test[0].shape[0]

    # Declare weights file path
    weights_file_path = '../weights/incep_resnet_compact_v4_logp_best_val_r2.h5'
    exp_weights_file_path = '../weights/incep_resnet_compact_v4_exp_logp_trsf_lrng_small_lr_best_val_r2.h5'


    # Load best LogP data predictor
    incep_mdl = incep_model_logp(n_h, n_w, n_c, n_y, lmbda=0)
    incep_mdl.load_weights(weights_file_path)


    # Load best Experimental LogP data predictor
    exp_incep_mdl = incep_model_logp(n_h, n_w, n_c, n_y, lmbda=0, frozen_index=7)
    exp_incep_mdl.load_weights(exp_weights_file_path)

    # Predict both LogP and experimental LogP
    y_pred = incep_mdl.predict(x_test)
    exp_y_pred = exp_incep_mdl.predict(exp_x_test)

    # Plot scatter
    fig = plt.figure()
    plt.scatter(y_test, y_pred, label="Test LogP")
    plt.scatter(exp_y_test, exp_y_pred, label="Test Experimental LogP")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    # plt.plot([-10, 6], [-10, 6])
    plt.plot()
    plt.legend()
    # Save the plot
    plt.savefig("output/2Dimg_logp_predictions.png")
    plt.close(fig)


def main():
    # l2_plot()
    # layers_plot()
    # layers_plot('r_squared', 'lower right')
    # transfer_learning_plot()
    # transfer_learning_plot(metric='r_squared', loc='lower right')
    plot_2Dimg_logp_predictions()


if __name__ == '__main__':
    main()