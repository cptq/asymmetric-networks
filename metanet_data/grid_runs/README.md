Making dataset of networks:

First, `make_hparams.py` makes a csv file of hyperparameters for the training runs (see `hparams.csv`).

Then `make_scripts.py` makes a `dataset_nets_train.sh` file where each line is a python command that trains a network with the corresponding set of hparams.

You can use `split` on the linux terminal to split the `dataset_nets_train.sh` into different chunks to run on different machines, e.g. `split -l 300 -d -a 3 dataset_nets_train.sh train_nets` will divide the scripts into files each of 300 lines (the `-a 3` part makes it so that the suffix is 3 characters long).

Finally, if you need a `.sh` file extension, you can do `for file in *; do mv -- "$file" "${file}.sh"; done` in the directory with all of the split scripts.
