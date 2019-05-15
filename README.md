# Project-CS230
**Title**: _Using Deep Learning to Predict Toxicity and Lipophilicity from Molecular Fingerprints and 2D Structures._

Please download the sdf datafiles from the following links and add them to `/data` folder:

* tox21db: https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_data_allsdf&sec=

* ncidb: https://cactus.nci.nih.gov/download/nci/ncidb.sdf.gz


Python version supported: `Python 3`.

Setup Steps:
1) Install miniconda (please see instructions in https://docs.conda.io/en/latest/miniconda.html )
2) Create a new environment:
    ``` conda create -n myenv python ```
3) Activate conda environment:
    ``` source activate myenv ```
4) Install RDKit (necessary for sdf files preprocessing):
    ``` conda install -c rdkit rdkit ```
5) Install requirements:
    ``` pip install -r requirements.txt ```