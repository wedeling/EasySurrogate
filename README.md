# EasySurrogate

This is a first beta release of EasySurrogate, a toolkit designed to facilitate the creating of surrogate models for multiscale simulations. This software is part of the Verified Exascale Computing for Multiscale Applications ([VECMA](www.vecma.eu)) toolkit.

## Requirements

+ Numpy
+ Scipy
+ Matplotlib
+ h5py

## Installation

After cloning the repository, EasySurrogate, along with the requirements, can be installed via:

```
python3 setup.py install --user
```
## Current features

+ An overaching `Campaign' structure of creating surrogates, similar to [EasyVVUQ](https://github.com/UCL-CCS/EasyVVUQ), the VECMA toolkit component for forward uncertainty propagation.

+ Quantized Softmax Networks: a neural network surrogate based on conditional resampling of reference data. The corresponding article can be found [here](https://arxiv.org/abs/2004.01457).

+ Reduced surrogates: a data compression technique used to reduce the size of the training data down by several order of magnitude, while retaining accuracy for spatially integrated quantities of interest. The corresponding article can be found [here](https://www.sciencedirect.com/science/article/pii/S0045793020300438?casa_token=opUTwCki7QIAAAAA:GwBFszrT7xF-yV5LDSUzcVZK45pA3cDSCj-tDoHgKGNS8YtpREVNXRFpsJapA84-sSIlob61ZZue). This has only been tested on problems with 2 spatial dimensions. Will be generalized in a subsequent release. 

More surrogate methods will be added shortly, amongst others [Kernel Mixture Networks](https://arxiv.org/abs/1705.07111).

## Tutorials

The following tutorials can be found in the `tutorials` folder:

 + `/L96`: Quantized Softmax Network (QSN) surrogates for atmospheric model equations. In this tutorial, the subgrid-scale term of the Lorenz96 equations is replaced by a QSN surrogate.
 
 + `gray_scott`: reduced surrogates for a reaction diffusion equation. This applies the data compression technique to the two-dimensional gray-scott equations.
