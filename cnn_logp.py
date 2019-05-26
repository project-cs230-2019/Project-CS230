""" Convolutional Neural Network @D image of skeletal formula model for logP """
import os

from utils.build_dataset import get_data
from utils.misc import r_squared, plot_data, save_history
import tensorflow as tf
import keras


MODEL_NAME = 'CNN_logp'


def fcnn_model_logp(n_h, n_w, n_c, n_y, lmbda):
    """
    This function returns a Fully Connected NN keras model

    :param n_h:     height of the 2D image input
    :param n_w:     width of the 2D image input
    :param n_c:     number of channels of the 2D image input
    :param n_y:     size of the output
    :return:        keras untrained Fully Connected NN linear regression model
    """

    model = keras.Sequential([
        keras.layers.Conv2D(16, kernel_size=7,
                            strides=4,
                            activation='relu',
                            input_shape=(n_h, n_w, n_c)),
        keras.layers.Conv2D(32, kernel_size=5,
                            strides=2,
                            activation='relu'),
        keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(lmbda)),
        keras.layers.Dense(n_y, kernel_regularizer=keras.regularizers.l2(lmbda))
    ])

    model.compile(optimizer=keras.optimizers.Adam(),
                       loss='mse',
                       metrics=['mae', r_squared]
                       )

    model.summary()

    return model


def main(train=False):
    """ Main function """
    # Get train and test dataset
    (x_train, y_train), (x_test, y_test) = get_data('data/ncidb_2Dimg.npz')

    # Change element type
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Input data normalization
    # Transform all input matrix elements in values belonging to [0,1] interval
    x_train /= 255
    x_test /= 255

    n_h, n_w, n_c = x_train[0].shape
    n_y = y_train[0].shape[0]


    # Build model
    fcnn_mdl = fcnn_model_logp(n_h, n_w, n_c, n_y, lmbda=0)

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
        fcnn_mdl.load_weights(weights_file_path)

    print('\ntest the model')
    test_loss, test_mae, test_r_squared = fcnn_mdl.evaluate(x_test, y_test)

    print('\n#######################################')
    print('Test loss:', test_loss)
    print('Test mae:', test_mae)
    print('Test R2:', test_r_squared)


if __name__ == '__main__':
    main(train=True)

