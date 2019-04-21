from rdkit import Chem
import csv


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
            if set(properties) <= set(properties_dict.keys()):
                mol_dict.update({property: properties_dict[property] for property in properties})
            mols.append(mol_dict)
        except Exception as err:
            print(err)

    csv_file_path = csv_file_path or sdf_file_path.replace(".sdf", ".csv")

    with open(csv_file_path, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=mols[0].keys())
        writer.writeheader()
        for mol_dict in mols:
            writer.writerow(mol_dict)


def main():
    """ Main function """
    convert_sdf_to_csv('data/tox21_10k_data_all.sdf', properties=['SR-HSE'])


if __name__ == '__main__':
    main()