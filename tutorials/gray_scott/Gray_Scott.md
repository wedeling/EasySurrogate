# Tutorial: Reduced Surrogates for the 2D Gray-Scott reaction-diffusion equations

The two-dimensional Gray-Scott reaction-diffusion equations model two chemical species `U` and `V`, which react according to `U + 2V --> 3V` and `V--> P`. The reactions are irreversible, so P is an inert product . The system is modelled by a set of partial differential equations for the local non-dimensional concentrations of `U` and `V`, denoted by `u(x, y, t)` and `v(x, y, t)`, both of which are in [0,1]. The **macroscopic** governing equations are

![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20%5Cbar%7Bu%7D%7D%7B%5Cpartial%20t%7D%20%3D%20D_u%5Cnabla%5E2%5Cbar%7Bu%7D%20-%20%5Cbar%7Bu%7D%5Cbar%7Bv%7D%5E2%20&plus;%20f%20%5Cleft%281%20-%20%5Cbar%7Bu%7D%5Cright%29%20&plus;%20%5Coverline%7BG_u%28u%2C%20v%29%7D%2C)

![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20%5Cbar%7Bv%7D%7D%7B%5Cpartial%20t%7D%20%3D%20D_v%5Cnabla%5E2%5Cbar%7Bv%7D%20&plus;%20%5Cbar%7Bu%7D%5Cbar%7Bv%7D%5E2%20-%20%5Cleft%28f%20&plus;%20k%5Cright%29%5Cbar%7Bv%7D%20&plus;%20%5Coverline%7BG_v%28u%2Cv%29%7D.)

The overline above `u` and `v` denotes the fact that this is a macroscopic variable, defined in this tutorial as the part of `u` and `v` which can be resolved on a spatial grid of `128x128` points. The `G_u` and `G_v` are subgrid-scale terms, meant to take the effect of unresolved scale into account.

Our aim here is to close the system by replacing `G_u` and `G_v` with data-driven surrogates. We generate a database of training data by solving the Gray-Scott equations for u and v at a high spatial resolution of `512 x 512` nodes. Instead of creating a surrogate for the spatially dependent subgrid-scale terms, we will create so-called reduced surrogates using EasySurrogate. These are specifically geared towards predicting global (i.e. spatially integrated) quantities of interest of the form
