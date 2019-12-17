# EasySurrogate

This branch of EasySurrogate contains all software required to reproduce the results from:

[ADD PAPER]

## Requirements

+ Numpy
+ Scipy
+ Matplotlib
+ h5py

## Installation

EasySurrogate, along with the requirements, can be installed via:

```
python3 setup.py install --user
```
## Tutorial

The `tests/` folder constains all numerical experiments. To execute these, perform the following steps

### Generate training data

For the Lorenz96 example, execute `python3 lorenz96.py`. This will generate training pairs that are used in `tests/lorenz96_quantized_softmax.py` and `tests/lorenz96_kernel_mixture_network.py`. You will be asked for a location to store the data (HDF5 format).
