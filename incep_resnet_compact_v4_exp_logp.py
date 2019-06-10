""" Convolutional Neural Network 2D image of skeletal formula model for logP """
import os

from utils.build_dataset import get_data
from utils.misc import r_squared, plot_data, save_history
import tensorflow as tf
import keras
from keras.utils import multi_gpu_model

MODEL_NAME = 'incep_resnet_compact_v4_exp_logp_trsf_lrng_small_lr'
# MODEL_NAME = 'incep_resnet_compact_v4_exp_logp'
# Set the number of available GPUs for the training
GPUs=0


def incept_res_block_a(input, n_c):
    # first inception block
    layer_1 = keras.layers.Conv2D(n_c, (3, 3), padding='same')(input)
    layer_2 = keras.layers.Conv2D(n_c, (5, 5), padding='same')(input)
    layer_3 = keras.layers.Conv2D(n_c, (7, 7), padding='same')(input)
    # Concatenate the different filters for the inception block
    concat_1 = keras.layers.concatenate([layer_1, layer_2, layer_3], axis=-1)
    # second inception block
    layer_1 = keras.layers.Conv2D(n_c*2, (3, 3), padding='same')(concat_1)
    layer_2 = keras.layers.Conv2D(n_c*2, (5, 5), padding='same')(concat_1)
    layer_3 = keras.layers.Conv2D(n_c*2, (7, 7), padding='same')(concat_1)
    # Concatenate the different filters for the inception block
    concat_2 = keras.layers.concatenate([layer_1, layer_2, layer_3], axis=-1)
    # Add shortcut for residual block
    shortcut = keras.layers.Conv2D(3 * 2 * n_c, (1,1))(input)
    z = keras.layers.add([shortcut, concat_2])
    output = keras.layers.LeakyReLU(alpha=0.03)(z)
    return output


def incept_res_block_b(input, n_c):
    # first inception block
    layer_1 = keras.layers.Conv2D(n_c, (1, 1))(input)
    layer_2 = keras.layers.Conv2D(n_c, (3, 3), padding='same')(input)
    layer_3 = keras.layers.Conv2D(n_c, (5, 5), padding='same')(input)
    # Concatenate the different filters for the inception block
    concat_1 = keras.layers.concatenate([layer_1, layer_2, layer_3], axis=-1)
    # second inception block
    layer_1 = keras.layers.Conv2D(n_c*2, (1, 1))(concat_1)
    layer_2 = keras.layers.Conv2D(n_c*2, (3, 3), padding='same')(concat_1)
    layer_3 = keras.layers.Conv2D(n_c*2, (5, 5), padding='same')(concat_1)
    # Concatenate the different filters for the inception block
    concat_2 = keras.layers.concatenate([layer_1, layer_2, layer_3], axis=-1)
    # Add shortcut for residual block
    shortcut = keras.layers.Conv2D(3 * 2 * n_c, (1,1))(input)
    z = keras.layers.add([shortcut, concat_2])
    output = keras.layers.LeakyReLU(alpha=0.03)(z)
    return output


def incep_model_logp(n_h, n_w, n_c, n_y, lmbda, transf_learn_weights_path=None, frozen_index=0):
    """
    This function returns a Fully Connected NN keras model

    :param n_h:     height of the 2D image input
    :param n_w:     width of the 2D image input
    :param n_c:     number of channels of the 2D image input
    :param n_y:     size of the output
    :return:        keras untrained Fully Connected NN linear regression model
    """

    input_img = keras.layers.Input(shape=(n_h, n_w, n_c))
    x = incept_res_block_a(input_img, n_c=2)
    x = keras.layers.MaxPool2D(pool_size=2)(x)
    x = incept_res_block_a(x, n_c=4)
    x = keras.layers.MaxPool2D(pool_size=2)(x)
    x = incept_res_block_a(x, n_c=8)
    x = keras.layers.MaxPool2D(pool_size=2)(x)
    x = incept_res_block_a(x, n_c=16)
    x = keras.layers.MaxPool2D(pool_size=2)(x)
    x = incept_res_block_b(x, n_c=32)
    x = keras.layers.MaxPool2D(pool_size=2)(x)
    x = keras.layers.Conv2D(256, (1, 1))(x)
    x = keras.layers.AveragePooling2D(pool_size=2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1024)(x)
    x = keras.layers.LeakyReLU(alpha=0.03)(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Dense(512)(x)
    x = keras.layers.LeakyReLU(alpha=0.03)(x)
    x = keras.layers.Dropout(0.25)(x)
    output = keras.layers.Dense(n_y)(x)

    if GPUs > 1:
        with tf.device('/cpu:0'):
            model = keras.models.Model(inputs=input_img, outputs=output)
            model.summary()
            # Replicates the model on n GPUs
            model = multi_gpu_model(model, gpus=GPUs)
    else:
        model = keras.models.Model(inputs=input_img, outputs=output)
        model.summary()

    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001, clipnorm=1.0, clipvalue=0.5),
                    loss='mse',
                    metrics=['mae', r_squared]
                    )

    # Transfer learning
    if transf_learn_weights_path:
        model.load_weights(transf_learn_weights_path)

    if frozen_index:
        if GPUs > 1:
            base_model = model.layers[-2]
        else:
            base_model = model
        for layer in base_model.layers[:-frozen_index]:
            # Freeze the layers except the last 4 layers
            layer.trainable = False
        base_model.summary()

    return model


def main(train=False, weights_file_path=None):
    """ Main function """
    # get train and test dataset
    print('Loading data')
    (x_train, y_train), (x_test, y_test) = get_data('data/ncidb_experim_data_2Dimg.npz', split=0.2)

    print('Normalize input dividing it by 255')
    # Input data normalization
    # Transform all input matrix elements in values belonging to [0,1] interval
    x_train /= 255
    x_test /= 255

    n_h, n_w, n_c = x_train[0].shape
    n_y = y_train[0].shape[0]


    epochs = 100

    if train:
        # Build model
        incep_mdl = incep_model_logp(
            n_h, n_w, n_c, n_y, lmbda=0,
            transf_learn_weights_path='weights/incep_resnet_compact_v4_logp_best_val_r2.h5',
            frozen_index=7
        )
        # incep_mdl = incep_model_logp(n_h, n_w, n_c, n_y, lmbda=0)

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

        csv_logger = keras.callbacks.CSVLogger('output/%s_%s_history.csv' % (MODEL_NAME, epochs))


        # Adjust Batch size based on GPUs number
        batch_size = 64 * GPUs or 32

        if weights_file_path:
            incep_mdl.load_weights(weights_file_path)

        history = incep_mdl.fit(x_train,
                               y_train,
                               epochs=epochs,
                               validation_split=0.2,
                               batch_size=batch_size,
                               callbacks=[weights_ckpt, best_ckpt, csv_logger]  # Save weights
                               )

        #Get data from history
        metrics = ['mean_absolute_error', 'r_squared', 'loss']
        save_history(history, "output/%s_%s_history.json" % (MODEL_NAME, epochs))
        plot_data(history, MODEL_NAME, epochs, metrics=metrics)
    else:
        # Build model
        # In order to reload the saved weights you need to refroze exactly the same number of layers
        # Change frozen index if using different weights from the one provided.
        incep_mdl = incep_model_logp(n_h, n_w, n_c, n_y, lmbda=0, frozen_index=7)
        incep_mdl.load_weights(weights_file_path)

        # Load the model weights
        if not weights_file_path:
            weights_file_path = os.path.abspath(os.path.join(os.curdir, 'weights/%s_%s.h5' % (MODEL_NAME, epochs)))
        if not os.path.exists(weights_file_path):
            raise Exception(
                "The weights file path specified does not exists: %s"
                % os.path.exists(weights_file_path)
            )

    print('\ntest the model')
    test_loss, test_mae, test_r_squared = incep_mdl.evaluate(x_test, y_test)

    print('\n#######################################')
    print('Test loss:', test_loss)
    print('Test mae:', test_mae)
    print('Test R2:', test_r_squared)


if __name__ == '__main__':
    main(train=True)
    # Comment the previous line and Uncomment the following for running in test mode
    # main(train=False, weights_file_path='weights/incep_resnet_compact_v4_exp_logp_trsf_lrng_small_lr_best_val_r2.h5')

    # N.B: If testing with less than 2 GPUs use the following weights instead
    # main(train=False, weights_file_path='weights/incep_resnet_compact_v4_exp_logp_trsf_lrng_small_lr_best_val_r2_nomultigpu.h5')
