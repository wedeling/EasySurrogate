# Tutorial Quantized Softmax Network for the Lorenz 96 system

## Files

The `tests/` folder constains following numerical experiments: 

+ `tests/lorenz_96/lorenz96_quantized_softmax.py`: applies a quantized softmax network to learn subgrid-scale term of the Lorenz96 system.

To execute these, perform the following steps

## Generate training data

Pre-generated training data for the Lorenz96 system is available in `test/samples/L96_training.hdf5`. If you wish to regenerate the data, execute `python3 tests/lorenz96/lorenz96.py`. This will generate training pairs that are used in `tests/lorenz_96/lorenz96_quantized_softmax.py`. You will be asked for a location to store the data (HDF5 format).

## Train the Quantized Softmax Network (QSN)




