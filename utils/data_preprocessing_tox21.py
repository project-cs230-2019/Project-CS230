import random

import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdmolops import RDKFingerprint
from keras.preprocessing.image import load_img

from misc import set_up_logging

# Set random seed to make the result reproducible
random.seed(1)

# Set up logging
LOGGER = set_up_logging(__name__)


def convert_sdf_to_npz(
        sdf_file_path,
        struct_type='fingerprints',
        properties=None,
        npz_file_path=None,
        ):
    """
    Converts sdf files into numpy arrays files file containing the chemical structure
    as either fingerprint or 2D image and properties

    :param sdf_file_path:       (str) (str) Path to the sdf file.
    :param porperties:          (list) Molecular properties to include in the npz file.
    :return:                    None
    """
    print(
        'Processing sdf data file: "%s" '
        'converting it into dataset of structure: "%s" and properties: "%s"'
        % (sdf_file_path, struct_type, properties)
    )
    # Import list of molecular objects from sdf file
    suppl = Chem.SDMolSupplier(sdf_file_path)

    X = []
    Y = []
    tot_num_mols = len(suppl)
    counter = 0
    for mol in suppl:
        try:
            # Get molecular object properties dict
            properties_dict = mol.GetPropsAsDict()

            smile = Chem.MolToSmiles(mol)

            if struct_type == 'fingerprints':
                structure = convert_smiles_into_fingerprints(smile)
            elif struct_type == '2Dimg':
                # TODO implement the image importer
                raise NotImplementedError('The image importer has not been implemented yet')
            else:
                raise NameError(
                    'The structure type specified does not exist. '
                    'Valid values: ["fingerprints", "2Dimg"]'
                )

            # Check if the example contains all the searched properties
            # otherwise discard it
            if set(properties) & set(properties_dict.keys()):
                X.append(structure)
                Y1 = []
                for key in properties:
                    k = properties_dict.get(key)
                    if k is None:
                        Y1.append(-1)
                    else:
                        Y1.append(k)
                Y.append(Y1)
                print(Y1)
                # Y.append([properties_dict[key] for key in properties])

        except AttributeError as err:
            LOGGER.warning("Molecule discarded from the dataset because: %r", err)

        if counter%1000 == 0:
            print('Processed %d/%d molecules' % (counter, tot_num_mols))
        counter += 1

    # Zip X and Y for shuffle
    mols = list(zip(X, Y))

    LOGGER.debug(
        'len of X: "%s", len of Y: "%s", len of mols: "%s"', len(X), len(Y), len(mols)
    )
    assert len(X) == len(Y) == len(mols)

    # Shuffle the data for the subsequent train, validation and/or test split
    random.shuffle(mols)

    # Unzip X and Y after shuffle
    X, Y = zip(*mols)

    # Convert nested lists into numpy arrays
    X = np.array(X)
    Y = np.array(Y)

    LOGGER.info(
        "dataset is composed by %d molecules with the following properties: %s",
        len(mols),
        properties
    )

    # Assign a file path for npz file if None
    npz_file_path = npz_file_path or sdf_file_path.replace(".sdf", "_%s_multi.npz" % struct_type)

    # Save npz file
    np.savez(npz_file_path, x=X, y=Y)


def convert_smiles_into_2d_structure_images(smile):
    """
    Converts SMILES text strings into 2D structure images

    :param smile:       (str) SMILE chemical formula. E.g.: 'O=C(C)Oc1ccccc1C(=O)O'
    :return:            None
    """
    mol = Chem.MolFromSmiles(smile)
    Draw.MolToFile(mol, 'data/cdk2_mol1.o.png')


def convert_smiles_into_fingerprints(smile):
    """
    Converts SMILES text strings into RDKit standard fingerprints

    :param smile:       (str) SMILE chemical formula. E.g.: 'O=C(C)Oc1ccccc1C(=O)O'
    :return:            RDKit fingerprint object
    """
    mol = Chem.MolFromSmiles(smile)
    return list(RDKFingerprint(mol))


def convert_img_into_array(path_to_img):
    """

    :param path_to_img:     (str) Path to image file
    :return:                Keras array representing the image
    """
    # load the image
    img = load_img(path_to_img)
    # report details about the image
    print(type(img))
    print(img.format)
    print(img.mode)
    print(img.size)
    # show the image
    img.show()


def main():
    """ Main function """
    # Preprocess the two datasets
    # properties = ['NR-AR']
    properties = ['NR-AR', 'NR-ER-LBD', 'SR-ATAD5']
    # properties = ['NR-AhR', 'NR-AR', 'NR-AR-LBD', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53', 'NR-Aromatase']
    convert_sdf_to_npz('../data/tox21_10k_data_all.sdf', properties=properties)
    convert_sdf_to_npz('../data/ncidb.sdf', properties=['KOW logP'])


if __name__ == '__main__':
    main()
