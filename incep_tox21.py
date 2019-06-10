""" Convolutional Neural Network @D image of skeletal formula model for logP """
""" Reference: https://towardsdatascience.com/hitchhikers-guide-to-residual-networks-resnet-in-keras-385ec01ec8ff"""
import os

from utils.build_dataset import get_data
from utils.misc import set_up_logging, f1_score, plot_data, save_history
import tensorflow as tf
import keras
from keras.utils import multi_gpu_model

MODEL_NAME = 'incep_tox21'
# Number of available GPUs for the training
GPUs = 1

# Set up logging
LOGGER = set_up_logging(__name__)


def inception_block(input, n_c):
    layer_1 = keras.layers.Conv2D(n_c, (3, 3), padding='same', activation='relu')(input)
    layer_2 = keras.layers.Conv2D(n_c, (5, 5), padding='same', activation='relu')(input)
    layer_3 = keras.layers.Conv2D(n_c, (7, 7), padding='same', activation='relu')(input)
    # Concatenate the different filters for the inception block
    output = keras.layers.concatenate([layer_1, layer_2, layer_3], axis=-1)
    return output

def incep_model_tox21(n_h, n_w, n_c, n_y, lmbda):
    """
    This function returns a Fully Connected NN keras model

    :param n_h:     height of the 2D image input
    :param n_w:     width of the 2D image input
    :param n_c:     number of channels of the 2D image input
    :param n_y:     size of the output
    :return:        keras untrained Fully Connected NN linear regression model
    """

    input_img = keras.layers.Input(shape=(n_h, n_w, n_c))
    x = inception_block(input_img, n_c=8)
    x = keras.layers.MaxPool2D(pool_size=2)(x)
    x = inception_block(x, n_c=16)
    x = keras.layers.MaxPool2D(pool_size=2)(x)
    x = inception_block(x, n_c=32)
    x = keras.layers.MaxPool2D(pool_size=4)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    output = keras.layers.Dense(n_y, activation = 'sigmoid')(x)

    if GPUs > 1:
        with tf.device('/cpu:0'):
            model = keras.models.Model(inputs=input_img, outputs=output)
            model.summary()
            # Replicates the model on n GPUs
            model = multi_gpu_model(model, gpus=GPUs)
    else:
        model = keras.models.Model(inputs=input_img, outputs=output)
        model.summary()

    model.compile(optimizer=keras.optimizers.Adam(lr=5e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy', f1_score]
                  )
    return model


def main(train=False):
    """ Main function """
    # get train and test dataset
    print('Loading data')
    (x_train, y_train), (x_test, y_test) = get_data('data/tox21_10k_data_all_2Dimg.npz')

    print('Normalize input dividing it by 255')
    # Input data normalization
    # Transform all input matrix elements in values belonging to [0,1] interval
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    n_h, n_w, n_c = x_train[0].shape
    n_y = y_train[0].shape[0]

    # Build model
    incep_tox21 = incep_model_tox21(n_h, n_w, n_c, n_y, lmbda=0)

    epochs = 15

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
            monitor='f1_score',
            verbose=1,
            save_best_only=True,
            mode='max'
        )

        # Adjust Batch size based on GPUs number
        batch_size = 64

        history = incep_tox21.fit(x_train,
                                  y_train,
                                  epochs=epochs,
                                  validation_split=0.1,
                                  batch_size=batch_size,
                                  callbacks=[weights_ckpt, best_ckpt]  # Save weights
                                  )

        # Get data from history
        metrics = ['f1_score', 'loss', 'acc']
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
        incep_mdl.load_weights(weights_file_path)

    print('\ntest the model')
    test_loss, test_acc, test_f1_score = incep_tox21.evaluate(x_test, y_test)

    print('\n#######################################')
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)
    print('Test F1 score:', test_f1_score)


if __name__ == '__main__':
    main(train=True)
