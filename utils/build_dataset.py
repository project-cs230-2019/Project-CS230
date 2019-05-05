import numpy as np
from utils.misc import set_up_logging

# Set up logging
LOGGER = set_up_logging(__name__)


def split_train_test_dataset(dataset, split):
    """
    This function splits the dataset in train and test ration

    :param dataset:     Train dataset.
    :param split:        The ratio of test dataset compared to the train dataset.
    :return:             Tuple of train, test datasets
    """
    test_size = int(len(dataset)*split)
    train_set, test_set = dataset[test_size:], dataset[:test_size]

    return train_set, test_set


def get_data(data_file, test_data_file=None, split=0.1):
    # Read npz file
    with np.load(data_file) as data:
        X = data['x']
        Y = data['y']

    # Split train and test set
    if not test_data_file:
        x_train, x_test = split_train_test_dataset(X, split)
        y_train, y_test = split_train_test_dataset(Y, split)
    else:
        with np.load(test_data_file) as test_data:
            x_test = test_data['x']
            y_test = test_data['y']
        x_train = X
        y_train = Y

    LOGGER.debug(
        '\nx_train shape: %r\n'
        'y_train shape: %r\n'
        'x_test shape: %r\n'
        'y_test shape: %r\n',
        x_train.shape,
        y_train.shape,
        x_test.shape,
        y_test.shape
    )

    return (x_train, y_train), (x_test, y_test)


def main():
    (x_train, y_train), (x_test, y_test) = get_data('../data/ncidb_fingerprints.npz')
    print(list(x_test))
    print(list(y_test))


if __name__ == '__main__':
    main()