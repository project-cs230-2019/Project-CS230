""" Convolutional Neural Network @D image of skeletal formula model for logP """
import os

from utils.build_dataset import get_data
from utils.misc import f1_score, plot_data, save_history
import tensorflow as tf
import numpy as np

import keras
from keras.utils import multi_gpu_model
from keras import backend as K
from sklearn.utils import class_weight
from keras.preprocessing.image import array_to_img

MODEL_NAME = 'incep_tox21_ml_3_properties_t'
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


#print out the f1 score for each toxicological properties
def each_metric(input_data, y_test, model):
    #classes = np.array(['NR-AhR', 'NR-AR', 'NR-AR-LBD', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
                   #'SR-MMP', 'SR-p53', 'NR-Aromatase'])
    classes = np.array(['NR-AR', 'NR-ER-LBD', 'SR-ATAD5'])
    y_pred = model.predict(input_data)

    for i in range(y_pred.shape[1]):
        y_p = y_pred[:,i]
        y_t = y_test[:,i]
        y_p = tf.convert_to_tensor(y_p, np.float32)
        y_t = tf.convert_to_tensor(y_t, np.float32)
        f1 = masked_f1(y_t, y_p)
        acc = masked_accuracy(y_t, y_p)
        true_positives = K.sum(K.round(K.clip(y_t * y_p, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_p, 0, 1)))- K.sum(K.cast(K.equal(y_t * y_p, mask_value), dtype = 'float32'))
        false_positives = predicted_positives - true_positives
        y_p_rounded = K.round(K.clip(y_p, 0, 1))
        y_t_rounded = K.round(K.clip(y_t, 0, 1))
        y_p_0 = K.cast(K.equal(y_p_rounded, 0), dtype = 'float32')
        false_negatives =K.sum(K.cast(K.equal(y_p_0, y_t_rounded), dtype = 'float32'))
        predicted_negatives = K.sum(K.cast(K.equal(y_p_rounded, 0), dtype = 'float32'))
        true_negatives = predicted_negatives - false_negatives

        with tf.Session() as sess:
            sess.run(f1)
            sess.run(acc)
            print(classes[i])
            print("f1 score:", f1.eval())
            print("accuracy:", acc.eval())
            print ("true positives:", true_positives.eval())
            print ("false_positives", false_positives.eval())
            print ("true_negatives", true_negatives.eval())
            print ("false_negatives", false_negatives.eval())
    return

def incept_res_block_a(input, n_c):
    layer_1 = keras.layers.Conv2D(n_c, (3, 3), padding='same')(input)
    layer_2 = keras.layers.Conv2D(n_c, (5, 5), padding='same')(input)
    layer_3 = keras.layers.Conv2D(n_c, (7, 7), padding='same')(input)
    # Concatenate the different filters for the inception block
    concat = keras.layers.concatenate([layer_1, layer_2, layer_3], axis=-1)
    # Add shortcut for residual block
    shortcut = keras.layers.Conv2D(3 * n_c, (1,1))(input)
    z = keras.layers.add([shortcut, concat])
    output = keras.layers.Activation('relu')(z)
    return output


def incept_res_block_b(input, n_c):
    layer_1 = keras.layers.Conv2D(n_c, (1, 1), padding='same')(input)
    layer_2 = keras.layers.Conv2D(n_c, (3, 3), padding='same')(input)
    layer_3 = keras.layers.Conv2D(n_c, (5, 5), padding='same')(input)
    # Concatenate the different filters for the inception block
    concat = keras.layers.concatenate([layer_1, layer_2, layer_3], axis=-1)
    # Add shortcut for residual block
    shortcut = keras.layers.Conv2D(3 * n_c, (1,1))(input)
    z = keras.layers.add([shortcut, concat])
    output = keras.layers.Activation('relu')(z)
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
    x = incept_res_block_a(input_img, n_c=8)
    x = keras.layers.MaxPool2D(pool_size=2)(x)
    x = incept_res_block_a(x, n_c=16)
    x = keras.layers.MaxPool2D(pool_size=2)(x)

    x = incept_res_block_a(x, n_c=32)
    x = keras.layers.MaxPool2D(pool_size=2)(x)
    x = incept_res_block_a(x, n_c=64)
    x = incept_res_block_a(x, n_c=64)
    x = incept_res_block_a(x, n_c=64)
    x = keras.layers.MaxPool2D(pool_size=2)(x)
    x = incept_res_block_b(x, n_c=128)
    x = incept_res_block_b(x, n_c=128)
    x = incept_res_block_b(x, n_c=128)
    x = keras.layers.MaxPool2D(pool_size=2)(x)
    x = incept_res_block_b(x, n_c=256)
    x = incept_res_block_b(x, n_c=256)
    x = incept_res_block_b(x, n_c=256)
    x = keras.layers.MaxPool2D(pool_size=2)(x)
    x = keras.layers.Conv2D(128, (1, 1))(x)
    x = keras.layers.AveragePooling2D(pool_size=2)(x)
    x = keras.layers.Flatten()(x)
#    x = keras.layers.Dense(1024)(x)
#    x = keras.layers.LeakyReLU(alpha=0.03)(x)
#    x = keras.layers.Dropout(0.25)(x)
#    x = keras.layers.Dense(512)(x)
#    x = keras.layers.LeakyReLU(alpha=0.03)(x)
#    x = keras.layers.Dropout(0.5)(x)
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
    test_loss, test_f1_score, test_acc = incep_tox21.evaluate(x_test, y_test)

    print('\n#######################################')
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)
    print('Test F1 score:', test_f1_score)
    print('\n#######################################')
    each_metric(x_test, y_test, incep_tox21)


if __name__ == '__main__':
    main(train=True)