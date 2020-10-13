# EasySurrogate

This is a first beta release of EasySurrogate, a toolkit designed to facilitate the creation of surrogate models for multiscale simulations. The development of this software is funded by the EU Horizon 2020 Verified Exascale Computing for Multiscale Applications ([VECMA](www.vecma.eu)) project.

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

+ [Kernel Mixture Networks](https://arxiv.org/abs/1705.07111): nonparametric estimation of conditional probability densities using neural networks. Unlike the Quantized Softmax Surrogate, no resampling of reference data is performed, and the probability density is continuous rather than discrete.

+ Reduced surrogates: a data compression technique used to reduce the size of the training data down by several order of magnitude, while retaining accuracy for spatially integrated quantities of interest. The corresponding article can be found [here](https://www.sciencedirect.com/science/article/pii/S0045793020300438?casa_token=opUTwCki7QIAAAAA:GwBFszrT7xF-yV5LDSUzcVZK45pA3cDSCj-tDoHgKGNS8YtpREVNXRFpsJapA84-sSIlob61ZZue). This has only been tested on problems with 2 spatial dimensions. Will be generalized in a subsequent release. 

+ Standard artificial neural networks, used for regression with (time-lagged) features.

+ It is possible to couple surrogates to the macroscopic model via [MUSCLE3](https://muscle3.readthedocs.io/en/latest/index.html), the third incarnation of the Multiscale Coupling Library and Environment. A tutorial is given below.

## Tutorials

The following tutorials can be found in the `tutorials` folder:

 + `/General`: a general introduction to EasySurrogate.

 + `/L96_QSN`: Quantized Softmax Network (QSN) surrogates for atmospheric model equations. In this tutorial, the subgrid-scale term of the Lorenz96 equations is replaced by a QSN surrogate.
 
  + `/L96_KMN`: Kernel Mixture Network (KMN) surrogates for atmospheric model equations. In this tutorial, the subgrid-scale term of the Lorenz96 equations is replaced by a KMN surrogate.

 + `/L96_ANN`: Artificial Neural Network (QSN) surrogates for atmospheric model equations. In this tutorial, the subgrid-scale term of the Lorenz96 equations is replaced by an ANN surrogate.

 + `/gray_scott`: reduced surrogates for a reaction diffusion equation. This applies the data compression technique to the two-dimensional gray-scott equations.
 
 + `/gray_scott_muscle`: this is basicaly the same tutorial as above, only the coupling with the reduced surrogates and the reaction diffusion equations is performed using MUSCLE3.
