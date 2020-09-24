# Tutorial: Reduced Surrogates for the 2D Gray-Scott reaction-diffusion equations

The two-dimensional Gray-Scott reaction-diffusion equations model two chemical species `U` and `V`, which react according to `U + 2V --> 3V` and `V--> P`. The reactions are irreversible, so P is an inert product . The system is modelled by a set of partial differential equations for the local non-dimensional concentrations of `U` and `V`, denoted by `u(x, y, t)` and `v(x, y, t)`, both of which are in [0,1]. The **macroscopic** governing equations are

![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20%5Cbar%7Bu%7D%7D%7B%5Cpartial%20t%7D%20%3D%20D_u%5Cnabla%5E2%5Cbar%7Bu%7D%20-%20%5Cbar%7Bu%7D%5Cbar%7Bv%7D%5E2%20&plus;%20f%20%5Cleft%281%20-%20%5Cbar%7Bu%7D%5Cright%29%20&plus;%20%5Coverline%7BG_u%28u%2C%20v%29%7D%2C)

![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20%5Cbar%7Bv%7D%7D%7B%5Cpartial%20t%7D%20%3D%20D_v%5Cnabla%5E2%5Cbar%7Bv%7D%20&plus;%20%5Cbar%7Bu%7D%5Cbar%7Bv%7D%5E2%20-%20%5Cleft%28f%20&plus;%20k%5Cright%29%5Cbar%7Bv%7D%20&plus;%20%5Coverline%7BG_v%28u%2Cv%29%7D.)

The overline above `u` and `v` denotes the fact that this is a macroscopic variable, defined in this tutorial as the part of `u` and `v` which can be resolved on a spatial grid of `128x128` points. The `G_u` and `G_v` are subgrid-scale terms, meant to take the effect of unresolved scale into account. 

Chemical `U` is added to the system at a feed rate given by the model constant `f`, and `V` is removed at a 'kill' rate `f+k`, where `k` is another model constant. 
The system is very sensitive to `f` and `k`, and a wide variety of spatiotemporal patterns are possible (Pearson, 1993), see the figure below for a snapshot of `u`.
Conversion from `U` to `V` is governed by the nonlinear term `uv^2`, which is subtracted from the right-hand side of `u` and added to the right-hand side of `v`. 
Finally, two diffusion terms are present, configured such that `U` diffuses faster than `V`. We specify `D_u = 2*10^{-5}` and `D_v = 10^{-5}`, and use a `2.5 x 2.5` spatial domain with periodic boundary conditions. 

![alt text](snapshot_0.png)

Our aim here is to close the system by replacing the subgrid-scale terms `G_u` and `G_v` with data-driven surrogates. We generate a database of training data by solving the Gray-Scott equations for u and v at a higher spatial resolution of `256 x 256` nodes. Instead of creating a surrogate for the spatially dependent subgrid-scale terms, we will create so-called *reduced surrogates* using EasySurrogate. These are specifically geared towards predicting global (i.e. spatially integrated) quantities of interest of the form

![equation](https://latex.codecogs.com/gif.latex?Q_i%28t%29%20%3D%20%5Cfrac%7B1%7D%7BA%7D%5Cint%5Cint%20q_i%28%5Cbar%7Bu%7D%2C%20%5Cbar%7Bv%7D%3B%20x%2C%20y%2C%20t%29%5C%3Bdxdy.)

Here, `q_i` is some function of the primitive variables and `A` is the area of the spatial domain. For instance, let us define our QoIs by the following set of integrands; `{q_1 = u, q_2 = u^2, q_3 = v, q_4 = v^2}`, i.e. we are interested in the average concentration of `U` and `V`, as well as the average squared concentrations. During the *training phase*, the task of the reduced subgrid-scale surrogates is to 'track' the reference QoIs, i.e.\ to keep `Q_i^{ref}(t) - Q_i(t)` small for all times during training, where `Q_i^{ref}` is the reference QoI computed from the high-resolution training data. We skip details for the sake of brevity, and refer to [this paper](https://www.sciencedirect.com/science/article/pii/S0045793020300438?casa_token=opUTwCki7QIAAAAA:GwBFszrT7xF-yV5LDSUzcVZK45pA3cDSCj-tDoHgKGNS8YtpREVNXRFpsJapA84-sSIlob61ZZue) for more information. 
