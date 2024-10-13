# Metanetwork experiments

## Dataset of trained networks

Our datasets of trained ResNets and Asymmetric ResNets can be found at [https://zenodo.org/records/13883507](https://zenodo.org/records/13883507).

### Files

Scripts for reproducing the experiments in our paper are in `paper_scripts`.

`data_utils.py` contains the dataset class for loading in neural network weights as data.

`models.py` contains metanetworks for processing input neural networks.

`train.py` contains the training and evaluation code.
