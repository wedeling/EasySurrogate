# Tutorial: Reduced Surrogates for the 2D Gray-Scott reaction-diffusion equations

The two-dimensional Gray-Scott reaction-diffusion equations model two chemical species `U` and `V`, which react according to `U + 2V` --> `3V` and `V`--> `P`. The reactions are irreversible, so P is an inert product . The system is modelled by a set of partial differential equations for the local non-dimensional concentrations of `U` and `V`, denoted by `u(x, y, t)` and `v(x, y, t)`, both of which are in [0,1]. The *macroscopic* governing equations are

![equation](\frac{\partial \bar{u}}{\partial t} = D_u\nabla^2\bar{u} - \bar{u}\bar{v}^2 + f \left(1 - \bar{u}\right) + \overline{G_u(u, v)},)

![equation](\frac{\partial \bar{v}}{\partial t} = D_v\nabla^2\bar{v} + \bar{u}\bar{v}^2 - \left(f + k\right)\bar{v} + \overline{G_v(u,v)}.)

The overline above `u` and `v`
