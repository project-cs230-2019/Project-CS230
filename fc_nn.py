""" Fully connected Neural Network fingerprints classifier for tox21 """
from utils.build_dataset import get_fingerprints_data
import tensorflow as tf
import keras
import matplotlib.pyplot as plt


def fcnn_classifier(n_x, n_y):
    """ This function returns a Fully Connected NN keras classifier
    :return:    keras untrained Fully Connected NN soft_max classifier
    """
    classifier = keras.Sequential([
        keras.layers.Dense(n_x, activation=tf.nn.relu),
        keras.layers.Dense(n_y, activation=tf.nn.sigmoid)
    ])

    classifier.compile(optimizer=tf.train.AdamOptimizer(),
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

    return classifier


def main(train=False):
    """ Main function """
    # Get train and test dataset
    (x_train, y_train), (x_test, y_test) = get_fingerprints_data()

    n_x = x_train[0].shape[0]
    n_y = y_train[0].shape[0]


    # Build classifier
    fcnn_clf = fcnn_classifier(n_x, n_y)

    epochs = 5

    if train:
        # Train classifier
        print('\ntrain the classifier')

        history = fcnn_clf.fit(x_train, y_train, epochs=epochs, validation_split=0.1)

        # Save weights
        fcnn_clf.save_weights('weights/fcnn_clf_%s.h5' % epochs)


        #Get data from history
        print(history.history.keys())
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title("model accuracy")
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("output/fully_connected_model_accuracy.png")
        plt.show()
        #Save the plot

        #Plot the loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("output/fully_connected_model_loss.png")
        plt.show()
    else:
        # Load the model weights
        import os
        weights_file_path = os.path.abspath(os.path.join(os.curdir, 'weights/fcnn_clf_%s.h5' % epochs))
        if not print(os.path.exists(weights_file_path)):
            print("The weights file path specified does not exists: %s" % os.path.exists(weights_file_path))
        fcnn_clf.load_weights(weights_file_path)

    print('\ntest the classifier')
    test_loss, test_acc = fcnn_clf.evaluate(x_test, y_test)

    print('\n#######################################')
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)


if __name__ == '__main__':
    main(train=True)

