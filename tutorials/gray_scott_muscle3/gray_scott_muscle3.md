# Tutorial: coupling surrogates to the macro model via MUSCLE3

MUSCLE3 is the third version of the Multiscale Coupling Library and Environment. Its purpose is to couple models operating at different spatial and/or temporal scales in order to form a coupled multiscale simulation. The full documentation can found [here](https://muscle3.readthedocs.io/en/latest/index.html). We will use MUSCLE3 to couple a surrogate model to the macroscopic partial differential equations of the problem. In particular, we will illustrate the approach using the two-dimensional Gray-Scott reaction diffusion equations. These equations model two chemical species `U` and `V`, which react according to `U + 2V --> 3V` and `V--> P`. The reactions are irreversible, so P is an inert product . The system is modelled by a set of partial differential equations for the local non-dimensional concentrations of `U` and `V`, denoted by `u(x, y, t)` and `v(x, y, t)`, both of which are in [0,1]. The **macroscopic** governing equations are

![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20%5Cbar%7Bu%7D%7D%7B%5Cpartial%20t%7D%20%3D%20D_u%5Cnabla%5E2%5Cbar%7Bu%7D%20-%20%5Cbar%7Bu%7D%5Cbar%7Bv%7D%5E2%20&plus;%20f%20%5Cleft%281%20-%20%5Cbar%7Bu%7D%5Cright%29%20&plus;%20%5Coverline%7BG_u%28u%2C%20v%29%7D%2C)

![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20%5Cbar%7Bv%7D%7D%7B%5Cpartial%20t%7D%20%3D%20D_v%5Cnabla%5E2%5Cbar%7Bv%7D%20&plus;%20%5Cbar%7Bu%7D%5Cbar%7Bv%7D%5E2%20-%20%5Cleft%28f%20&plus;%20k%5Cright%29%5Cbar%7Bv%7D%20&plus;%20%5Coverline%7BG_v%28u%2Cv%29%7D.)

The overline above `u` and `v` denotes the fact that this is a macroscopic variable, defined in this tutorial as the part of `u` and `v` which can be resolved on a spatial grid of `128x128` points. The `G_u` and `G_v` are subgrid-scale terms, meant to take the effect of spatially unresolved scales into account. 

In [this](https://github.com/wedeling/EasySurrogate/edit/master/tutorials/gray_scott/Gray_Scott.md) tutorial we created reduced surrogates for `G_u` and `G_v`, where we directly placed a call to the surrogate model inside the script of the macroscopic equations. We refer to the linked tutorial for more information on the reduced surrogates. Here, we will now place the reduced surrogate model in its own file, and use MUSCLE3 to perform the coupling.

## Files

+ `tests/gray_scott_reduced_muscle3/macro.py`: the macroscopic equations, modified with MUSCLE3 commands.
+ `tests/gray_scott_reduced_muscle3/micro.py`: the reduced surrogate.
+ `tests/gray_scott_reduced_muscle3/run.py`: the MUSCLE3 configuration file, which runs the simulation.

We will assume that MUSCLE3 has been properly installed.

## Modifications to the macroscopic model

![alt text](gray-scott_coupling.png)
