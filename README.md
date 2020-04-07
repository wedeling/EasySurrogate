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

```
python3 setup.py install --user
```

## Tutorials

The following tutorials are available:

+ `L96_Quantized_Softmax.md`: apply a quantized softmax network to learn subgrid-scale term of the Lorenz96 system.

## Funding

This research is funded by the Netherlands Organization for Scientific Research (NWO) through the Vidi project "Stochastic models for unresolved scales in geophysical flows", and from the European Union Horizon 2020 research and innovation programme under grant agreement #800925 (VECMA project).
