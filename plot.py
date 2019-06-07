from sys import platform
import os
import json

from keras.models import Model
from keras.preprocessing.image import img_to_array, array_to_img, load_img
import matplotlib
if platform == "darwin":  # OS X
    matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np

from incep_resnet_compact_exp_logp import incep_model_logp
from fc_nn_6l_exp_logp import fcnn_model_logp
from utils.build_dataset import get_data


def l2_plot(metric='loss', loc='upper left'):
    for i, history_file in enumerate([
        'fcnn_logp_6l_50_history.json',
        'fcnn_logp_6l_l2reg00001_50_history.json',
        'fcnn_logp_6l_l2reg0001_50_history.json',
        'fcnn_logp_6l_l2reg001_50_history.json',
    ]):
        with open(os.path.join('.','output', history_file), 'r') as hf:
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
        with open(os.path.join('.','output', history_file), 'r') as hf:
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
        with open(os.path.join('.','output', history_file), 'r') as hf:
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


def plot_fingerprints_logp_predictions():
    # Load LogP data
    print('Loading data')
    (_, _), (x_test, y_test) = get_data('data/ncidb_fingerprints.npz')

    # Load Experimental LogP data
    print('Loading data')
    (_, _), (exp_x_test, exp_y_test) = get_data('data/ncidb_experim_data_fingerprints.npz', split=0.2)

    # Get input and output dims
    n_x = x_test[0].shape[0]
    n_y = y_test[0].shape[0]

    # Declare weights file path
    weights_file_path = 'weights/fcnn_logp_6l_best_val_r2.h5'
    exp_weights_file_path = 'weights/fcnn_exp_logp_6l_trsf_lrng_best_val_r2.h5'


    # Load best LogP data predictor
    fcnn_mdl = fcnn_model_logp(n_x, n_y, lmbda=0)
    fcnn_mdl.load_weights(weights_file_path)


    # Load best Experimental LogP data predictor
    exp_fcnn_mdl = fcnn_model_logp(n_x, n_y, lmbda=0, frozen_layers=4)
    exp_fcnn_mdl.load_weights(exp_weights_file_path)

    # Predict both LogP and experimental LogP
    y_pred = fcnn_mdl.predict(x_test)
    exp_y_pred = exp_fcnn_mdl.predict(exp_x_test)

    # Plot scatter
    fig = plt.figure()
    plt.scatter(y_test, y_pred, label="Test Set (LogP model). $R^2$: 0.839")
    plt.scatter(exp_y_test, exp_y_pred, label="Test Set (Experimental LogP model). $R^2$: 0.923")
    plt.title("Predicted LogP values from fingerprints")
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.plot([-20, 40], [-20, 40], '--', color='red')
    plt.legend()
    # Save the plot
    plt.savefig("output/1Dfingerprints_logp_predictions.png")
    plt.close(fig)


def plot_2Dimg_logp_predictions():
    # Load LogP data
    print('Loading data')
    (_, _), (x_test, y_test) = get_data('data/ncidb_2Dimg.npz')

    # Load Experimental LogP data
    print('Loading data')
    (_, _), (exp_x_test, exp_y_test) = get_data('data/ncidb_experim_data_2Dimg.npz', split=0.2)

    print('Normalize input dividing it by 255')
    # Input data normalization
    # Transform all input matrix elements in values belonging to [0,1] interval
    x_test /= 255
    exp_x_test /= 255

    # Get input and output dims
    n_h, n_w, n_c = x_test[0].shape
    n_y = y_test[0].shape[0]

    # Declare weights file path
    weights_file_path = 'weights/incep_resnet_compact_v4_logp_best_val_r2.h5'
    exp_weights_file_path = 'weights/incep_resnet_compact_v4_exp_logp_trsf_lrng_small_lr_best_val_r2.h5'


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
    plt.scatter(y_test, y_pred, label="Test Set (LogP model). $R^2$: 0.852")
    plt.scatter(exp_y_test, exp_y_pred, label="Test Set (Experimental LogP model). $R^2$: 0.964")
    plt.title("Predicted LogP values from 2D molecule images")
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.plot([-20, 40], [-20, 40], '--', color='red')
    plt.legend()
    # Save the plot
    plt.savefig("output/2Dimg_logp_predictions.png")
    plt.close(fig)


def _plot_kernels(kernels):
    fig, axes = plt.subplots(2, 3, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.matshow(kernels[:,:,i])
        ax.set_title("Kernel %s"%i, y=1.1)


def visualize_layers_attention(model, layers, image):
    x = np.array([img_to_array(image)])
    x /= 255
    for layer in layers:
        print("Layer %i"%layer)
        _plot_kernels(
            Model(
                inputs=model.input,
                outputs=model.layers[layer].output
            ).predict(x)[0]
        )
        plt.show()


def visualize_exp_logp_mdl_attention():
    # Load model
    exp_weights_file_path = 'weights/incep_resnet_compact_v4_exp_logp_trsf_lrng_small_lr_best_val_r2_nomultigpu.h5'

    exp_incep_mdl = incep_model_logp(150, 150, 1, 1, lmbda=0, frozen_index=7)
    exp_incep_mdl.load_weights(exp_weights_file_path)

    # Load image
    image = load_img('data/exp_logP_test_mol3.bmp', color_mode="grayscale")

    visualize_layers_attention(exp_incep_mdl, [16, 17, 18, 19, 29, 30, 31], image)


def visualize_tox21_mdl_attention():
    #TODO apply similar logic for tox21
    # Load model
    weights_file_path = 'weights/...'

    tox21_mdl = tox21_mdl(150, 150, 1, 1, lmbda=0)
    tox21_mdl.load_weights(weights_file_path)

    # Load image
    image = load_img('data/...', color_mode="grayscale")

    visualize_layers_attention(tox21_mdl, [16, 17, 18, 19, 29, 30, 31], image)


def main():
    # TODO Uncomment those lines as necessary
    # l2_plot()
    # layers_plot()
    # layers_plot('r_squared', 'lower right')
    # transfer_learning_plot()
    # transfer_learning_plot(metric='r_squared', loc='lower right')
    # plot_fingerprints_logp_predictions()
    # plot_2Dimg_logp_predictions()
    visualize_exp_logp_mdl_attention()


if __name__ == '__main__':
    main()