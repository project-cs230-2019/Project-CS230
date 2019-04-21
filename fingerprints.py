from rdkit import Chem
from rdkit.Chem.rdmolops import RDKFingerprint


def convert_smiles_into_fingerprints(smile):
    """
    Converts SMILES text strings into RDKit standard fingerprints

    :param smile:       (str) SMILE chemical formula. E.g.: 'O=C(C)Oc1ccccc1C(=O)O'
    :return:            RDKit fingerprint object
    """
    mol = Chem.MolFromSmiles(smile)
    return RDKFingerprint(mol)


def main():
    """ Main function """
    # Example smiles
    aspirin = 'O=C(C)Oc1ccccc1C(=O)O'
    paracetamol = 'CC(=O)Nc1ccc(O)cc1'

    # fingerprints
    fpm1 = convert_smiles_into_fingerprints(aspirin)
    fpm2 = convert_smiles_into_fingerprints(paracetamol)

    print(
        'Aspirin fingerprint lenght: %d, bit vector: %s' %
        (len(list(fpm1)), str(list(fpm1)))
    )
    print(
        'Paracetamol fingerprint lenght: %d, bit vector: %s' %
        (len(list(fpm2)), str(list(fpm2)))
    )


if __name__ == '__main__':
    main()




