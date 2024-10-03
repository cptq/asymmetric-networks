This folder contains code for ResNet and MLP linear mode connectivity experiments. 

**train-mlp-mnist.py** contains code for running MLP linear mode connectivity tests on MNIST. The --symmetry argument determines which architecture is used according to 
whether it is 0 (Standard) 1 (W-Asym) or 2 (Sigma-Asym). If Standard is chosen, then the networks will be linearly interpolated between before and after Git Rebasin alignment.

**train-cifar-resnet.py** contains code for running ResNet linear mode connectivity tests on CIFAR10. The --symmetry argument determines which architecture is used according to 
whether it is 0 (Standard) 1 (W-Asym) or 2 (Sigma-Asym). If Standard is chosen, then the networks will be linearly interpolated between before and after alignment.

**models/models_mlp.py** and **models/models_resnet.py** contain source code for Asymmetric and Standard MLP and ResNet.

**rebasin** contains weight matching code from https://github.com/themrzmaster/git-re-basin-pytorch/tree/main

