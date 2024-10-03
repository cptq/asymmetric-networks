Based off of [Efficient Low Rank Gaussian Variational Inference for Neural Networks](https://proceedings.neurips.cc/paper/2020/file/310cc7ca5a76a446f85c1a0d641ba96d-Paper.pdf).

**fit.py** is taken directly from the above, and contains code for training the Bayesian Neural Networks.

**alg.py** contains the actual implementations of Asymmetric Bayesian Linear and Convolution2d layers. For a rank-4 covariance approximation, behind the scenes, the code has 6 different trainable versions of each network, one for each rank-1 variance component, one for the diagonal variance component, and one for the mean. In the forward pass, the output of each variance component is squared. By zeroing out the noise/variance in certain entries of the rank and diagonal variance components, and then fixing the corresponding entries of the mean Asymmetric network forward pass to (nonzero constants), we are fixing entries of the bayesian network.

**sparse.py** contains code for the underlying standard & asymmetric networks which then have a bayesian wrapper applied to them.

**train_sparse_MLP.py** contains code for training & testing Bayesian Asymmetric and Standard MLPs.

**train_sparse_ResNet.py** contains code for training & testing Bayesian Asymmetric and Standard ResNets.
