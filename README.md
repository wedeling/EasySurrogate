# EasySurrogate

This branch of EasySurrogate contains all software required to reproduce the results from:

D. Crommelin, W. Edeling, "Resampling with neural networks for stochastic parameterization in multiscale systems
", (submitted), 2020. 

[Paper](https://arxiv.org/abs/2004.01457)

## Requirements

+ Numpy
+ Scipy
+ Matplotlib
+ h5py

## Installation

EasySurrogate, along with the requirements, can be installed via:

```lorenz96_quantized_softmax.py
python3 setup.py install --user
```

## Tutorials

The following tutorials are available:

+ `L96_Quantized_Softmax.md`: apply a quantized softmax network to learn subgrid-scale term of the Lorenz96 system.

## Files

There are 4 main files which were used to generate the results from the paper:

* `tests/lorenz96/lorenz96.py`: a script used to generate Lorenz96 training data.
* `tests/lorenz96/lorenz96_quantized_softmax.py`: the main script, where the QSN surrogate has features and an output which are non-local in space.
* `tests/lorenz96/lorenz96_quantized_softmax_local.py`: this is essentially the same script as the main script, except the surrogate is applied completely locally in space. This means the surrogate is applied pointwise, using only input features from the same spatial point.
* `tests/lorenz96/lorenz96_quantized_softmax_neighbourhood.py`: again, this is essentially the same script, except the surrogate is applied locally, while the input features are still non-local in space.

## Funding

This research is funded by the Netherlands Organization for Scientific Research (NWO) through the Vidi project "Stochastic models for unresolved scales in geophysical flows", and from the European Union Horizon 2020 research and innovation programme under grant agreement #800925 (VECMA project).
