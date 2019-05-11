from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdmolops import RDKFingerprint
from keras.preprocessing.image import load_img
import csv
import logging
import random

# Set random seed to make the result reproducible
random.seed(1)


# Set up logging
FORMAT = '%(asctime)s - %(levelname)s: %(message)s'
logging.basicConfig(format=FORMAT)
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)


def convert_sdf_to_csv(sdf_file_path, properties=None, csv_file_path=None):
    """
    Converts sdf files into csv file containing the chemical structure
    as smile and properties

    :param sdf_file_path:       (str) (str) Path to the sdf file.
    :param porperties:          (list) Molecular properties to include in the csv file.
    :return:                    None
    """
    suppl = Chem.SDMolSupplier(sdf_file_path)
    mols = []
    for mol in suppl:
        mol_dict = {}
        try:
            mol_dict["smile"] = Chem.MolToSmiles(mol)
            properties_dict = mol.GetPropsAsDict()
            # Check if the example contains all the searched properties
            # otherwise discard it
            if set(properties) <= set(properties_dict.keys()):
                mol_dict.update({property: properties_dict[property] for property in properties})
                mols.append(mol_dict)
        except Exception as err:
            LOGGER.error(err)

    # Shuffle the training data for the subsequent validation split
    random.shuffle(mols)

    LOGGER.info(
        "dataset is composed by %d molecules with the following properties: %s",
        len(mols),
        properties
    )

    csv_file_path = csv_file_path or sdf_file_path.replace(".sdf", ".csv")

    with open(csv_file_path, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=mols[0].keys())
        writer.writeheader()
        for mol_dict in mols:
            writer.writerow(mol_dict)


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
    return RDKFingerprint(mol)


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
    properties = ['NR-AR', 'NR-ER-LBD', 'SR-ATAD5']
    # properties = ['NR-AhR', 'NR-AR', 'NR-AR-LBD', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53', 'NR-Aromatase']
    convert_sdf_to_csv('data/tox21_10k_data_all.sdf', properties=properties)
    # convert_smiles_into_2d_structure_images('O=C(C)Oc1ccccc1C(=O)O')
    # convert_img_into_array('data/cdk2_mol1.o.png')
    # get_fingerprints_data('data/tox21_10k_data_all.csv')


if __name__ == '__main__':
    main()