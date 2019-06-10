""" Convolutional Neural Network @D image of skeletal formula model for logP """
import os

from utils.build_dataset import get_data
from utils.misc import f1_score, plot_data, save_history
import tensorflow as tf
import numpy as np

import keras
from keras.utils import multi_gpu_model
from keras.initializers import glorot_uniform
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K


MODEL_NAME = 'incep_tox21_big_ml'
# Number of available GPUs for the training
GPUs = 1
mask_value = -1


def masked_f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))- K.sum(K.cast(K.equal(y_true * y_pred, mask_value), dtype = 'float32'))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def masked_accuracy(y_true, y_pred):
    total = K.sum(K.cast(K.not_equal(y_true, mask_value), dtype = 'float32'))
    correct = K.sum(K.cast(K.equal(y_true, K.round(y_pred)), dtype = 'float32'))
    return correct / total

#loss function for multi-task with missing label
def masked_loss_function(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    return K.binary_crossentropy(y_true * mask, y_pred * mask)


def each_metric(input_data, y_test, model):
    classes = np.array(['NR-AhR', 'NR-AR', 'NR-AR-LBD', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
                  'SR-MMP', 'SR-p53', 'NR-Aromatase'])
    y_pred = model.predict(input_data)

    for i in range(y_pred.shape[1]):
        y_p = y_pred[:,i]
        y_t = y_test[:,i]
        y_p = tf.convert_to_tensor(y_p, np.float32)
        y_t = tf.convert_to_tensor(y_t, np.float32)
        f1 = f1_score(y_t, y_p)
        if ml==True:
            f1 = masked_f1(y_t, y_p)
        with tf.Session() as sess:
            sess.run(f1)
            sess.run(acc)
            print(classes[i])
            print("f1 score:", f1.eval())
            print("accuracy:", acc.eval())
    return


def identity_block(X, f, filters, stage, block):
    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def incep_model_tox21(n_h, n_w, n_c, n_y, lmbda):
    """
    This function returns a Fully Connected NN keras model
    :param n_h:     height of the 2D image input
    :param n_w:     width of the 2D image input
    :param n_c:     number of channels of the 2D image input
    :param n_y:     size of the output
    :return:        keras untrained Fully Connected NN linear regression model
    """
    # create the base pre-trained model
    # Zero-Padding
    input_img = keras.layers.Input(shape=(n_h, n_w, n_c))
    X = ZeroPadding2D((3, 3))(input_img)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    # Output layer
    X = Flatten()(X)

    output = keras.layers.Dense(n_y, activation = 'sigmoid')(X)

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
                  loss = masked_loss_function,
                  metrics = [masked_f1, masked_accuracy]
                  )
    return model


def main(train=False):
    """ Main function """
    # get train and test dataset
    print('Loading data')
    (x_train, y_train), (x_test, y_test) = get_data('data/tox21_10k_data_all_2Dimg_ml.npz')

    print('Normalize input dividing it by 255')
    # Input data normalization
    # Transform all input matrix elements in values belonging to [0,1] interval
    x_train /= 255
    x_test /= 255

    n_h, n_w, n_c = x_train[0].shape
    n_y = y_train[0].shape[0]

    # Build model
    incep_tox21 = incep_model_tox21(n_h, n_w, n_c, n_y, lmbda=0)

    epochs = 100

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
            monitor='val_masked_f1',
            verbose=1,
            save_best_only=True,
            mode='max'
        )

        # Adjust Batch size based on GPUs number
        batch_size = 64 * GPUs or 32

        history = incep_tox21.fit(x_train,
                                  y_train,
                                  epochs=epochs,
                                  validation_split=0.1,
                                  batch_size=batch_size,
                                  callbacks=[weights_ckpt, best_ckpt]  # Save weights
                                  )

        # Get data from history
        metrics = ['masked_f1', 'loss','masked_accuracy']
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
    print('\n#######################################')
    each_metric(x_test, y_test, incep_tox21)


if __name__ == '__main__':
    main(train=True)