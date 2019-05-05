import logging
from rdkit import RDLogger
from keras import backend as K


def set_up_logging(logger_name):
    # Set up logging
    FORMAT = '%(asctime)s - %(levelname)s: %(message)s'
    logging.basicConfig(format=FORMAT)
    LOGGER = logging.getLogger(logger_name)
    LOGGER.setLevel(logging.DEBUG)

    # Set rdkit logger to critical
    rdlg = RDLogger.logger()
    rdlg.setLevel(RDLogger.CRITICAL)

    return LOGGER


def r_squared(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))