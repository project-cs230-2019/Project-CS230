from sys import platform
import os
import json
import matplotlib
if platform == "darwin":  # OS X
    matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from utils.misc import plot_data, save_history


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


def main():
    l2_plot()
    layers_plot()
    layers_plot('r_squared', 'lower right')
    transfer_learning_plot()
    transfer_learning_plot(metric='r_squared', loc='lower right')


if __name__ == '__main__':
    main()