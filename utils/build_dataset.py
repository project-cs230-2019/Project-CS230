import numpy as np
import csv

from utils.data_preprocessing import convert_smiles_into_fingerprints


def split_train_test_dataset(dataset, split=0.1):
    """
    This function splits the dataset in train and test ration

    :param dataset:     Train dataset.
    :param split:        The ratio of test dataset compared to the train dataset.
    :return:             Tuple of train, test datasets
    """
    test_size = int(len(dataset)*split)
    train_set, test_set = dataset[test_size:], dataset[:test_size]

    return train_set, test_set


def build_fingerprints_dataset(dataset):
    """
    This function converts the smiles into fingerprints and the properties into
    vectors if multiples.

    :param dataset:         A list of dictionaries each one including
                            smile and property/ies
    :return:                A tuple of X and
    """
    X = []
    Y = []
    for mol in dataset:
        fpmol = convert_smiles_into_fingerprints(mol[0])
        X.append(np.array(list(fpmol)))
        Y.append(np.array(mol[1:]))

    return np.array(X), np.array(Y)


def get_fingerprints_data():
    # Read csv file
    with open('data/tox21_10k_data_all.csv', 'r') as csvfile:
        dataset = list(csv.reader(csvfile))[1:]

    # Split train and test set
    train_set, test_set = split_train_test_dataset(dataset)

    return build_fingerprints_dataset(train_set), build_fingerprints_dataset(test_set)