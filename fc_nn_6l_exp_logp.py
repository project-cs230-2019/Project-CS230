""" Fully connected Neural Network fingerprints model for logP """
import os

from utils.build_dataset import get_data
from utils.misc import r_squared, plot_data, save_history
import tensorflow as tf
import keras


MODEL_NAME = 'fcnn_exp_logp_6l_trsf_lrng'
# MODEL_NAME = 'fcnn_exp_logp_6l'


def fcnn_model_logp(n_x, n_y, lmbda, transf_learn_weights_path=None, frozen_layers=0):
    """
    This function returns a Fully Connected NN keras model

    :param n_x:     size of the input
    :param n_y:     size of the output
    :return:        keras untrained Fully Connected NN linear regression model
    """

    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(n_x,)),
        keras.layers.Dense(n_x, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(lmbda)),
        keras.layers.Dense(int(n_x/2), activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(lmbda)),
        keras.layers.Dense(int(n_x/8), activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(lmbda)),
        keras.layers.Dense(int(n_x/4), activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(lmbda)),
        keras.layers.Dense(int(n_x/16), activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(lmbda)),
        keras.layers.Dense(n_y)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                       loss='mse',
                       metrics=['mae', r_squared]
                       )

    # Transfer learning
    if transf_learn_weights_path:
        model.load_weights(transf_learn_weights_path)
        for layer in model.layers[:frozen_layers]:
            # Freeze the layers except the last 4 layers
            layer.trainable = False

    return model


def main(train=False):
    """ Main function """
    # Get train and test dataset
    (x_train, y_train), (x_test, y_test) = get_data('data/ncidb_experim_data_fingerprints.npz', split=0.2)

    n_x = x_train[0].shape[0]
    n_y = y_train[0].shape[0]

    # Build model
    fcnn_mdl = fcnn_model_logp(n_x, n_y, lmbda=0, transf_learn_weights_path='weights/fcnn_logp_6l_50.h5', frozen_layers=4)
    # fcnn_mdl = fcnn_model_logp(n_x, n_y, lmbda=0)


    epochs = 50

    if train:
        # Train model
        print('\ntrain the model')

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

        history = fcnn_mdl.fit(x_train,
                               y_train,
                               epochs=epochs,
                               validation_split=0.2,
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
        fcnn_mdl.load_weights(weights_file_path)

    print('\ntest the model')
    test_loss, test_mae, test_r_squared = fcnn_mdl.evaluate(x_test, y_test)

    print('\n#######################################')
    print('Test loss:', test_loss)
    print('Test mae:', test_mae)
    print('Test R2:', test_r_squared)


if __name__ == '__main__':
    main(train=True)

