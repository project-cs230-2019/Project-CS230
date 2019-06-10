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


Using the models
----------------

The best performing models are the following:
* Lipophilicity predictors:
    * **fingerprints input**\
        fc_nn_6l_logp.py (LogP) \
        fc_nn_6l_exp_logp.py (experimental LogP)
    * **2D molecular images input** \
        incep_resnet_compact_v4_logp.py  (LogP) \
        incep_resnet_compact_v4_exp_logp.py  (experimental LogP)

* Toxicity predictors:
    * **fingerprints input**\
        fc_nn_tox21.py

    * **2D molecular images input** \
        incep_resnet_tox21_t.py

General instructions set ```Train=True``` in main function for training and ```Train=False```
for running in test mode specifying a previously saved weight file.
N.B: We include some of the best performing models weights files in ```weights/``` folder.

You can see come of the previous run results (metrics images and history files) in the ```output/``` folder.


