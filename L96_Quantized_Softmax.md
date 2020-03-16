# Tutorial: Quantized Softmax Network for the Lorenz 96 system

The two layer Lorenz 96 system is given by:

![equation](https://latex.codecogs.com/svg.latex?%5Cdot%7BX%7D_k%20%3D%20X_%7Bk-1%7D%28X_%7Bk&plus;1%7D%20-X_%7Bk-2%7D%29%20-%20X_k%20&plus;%20F%20&plus;%20B_k)

![equation](https://latex.codecogs.com/svg.latex?%5Cdot%7BY%7D_%7Bj%2Ck%7D%20%3D%20%5Cfrac%7B1%7D%7B%5Cepsilon%7D%5Cleft%5BY_%7Bj&plus;1%2Ck%7D%28Y_%7Bj-1%2Ck%7D%20-%20Y_%7Bj&plus;2%2Ck%7D%29%29%20-%20Y_%7Bj%2Ck%7D%20&plus;%20h_yX_k%20%5Cright%5D)

![equation](https://latex.codecogs.com/svg.latex?B_k%20%3A%3D%20%5Cfrac%7Bh_x%7D%7BJ%7D%5Csum_%7Bj%3D1%7D%5E%7BJ%7D%20Y_%7Bj%2Ck%7D)

Here B_k is the subgrid-scale term for which we create a surrogate model in the form of a quantized softmax network, see the paper (referenced [here](./README.md)) for details.

## Files

The `tests/` folder constains following numerical experiments: 

+ `tests/lorenz96/lorenz96.py`: the solver for the Lorenz 96 system
+ `tests/lorenz96/lorenz96_quantized_softmax.py`: applies a quantized softmax network to learn subgrid-scale term of the Lorenz96 system.
+ `tests/lorenz96/figures`: contains the figures of the results
+ `tests/lorenz96/movies`: contains the movies of the results

To recreate the results, perform the following steps

## Generate training data

The first step is to generate the training data, by executing `python3 tests/lorenz96/lorenz96.py`. This will generate training pairs that are used in `tests/lorenz_96/lorenz96_quantized_softmax.py`. You will be asked for a location to store the data (HDF5 format).

## Training / predicting with the Quantized Softmax Network

As mentioned, the file `tests/lorenz_96/lorenz96_quantized_softmax.py` contains the QSN applied to L96 data. It contains the following flags:

```python
train = True            #train the network
make_movie = False      #make a movie (of the training)
predict = True          #predict using the learned SGS term
store = True            #store the prediction results
make_movie_pred = False #make a movie (of the prediction)
```

+ `train` (Boolean): train the QSN network on L96 data.
+ `make_movie` (Boolean): make a movie as displayed above. This is the QSN evaluated on the *training* data.

![alt text](https://github.com/wedeling/EasySurrogate/blob/phys_D/tests/movies/qsn.gif)

+ `predict` (Boolean): use the QSN to replace the subgrid-scale term of the L96 system.
+ `store` (Boolean): store the prediction results
+ `make_movie_pred`: make a movie of the *prediction* for one spatial point. Example shown below, which shows eventual divergence of the data and prediction trajectories due to the chaotic nature of L96.

![alt text](https://github.com/wedeling/EasySurrogate/blob/phys_D/tests/movies/qsn_pred.gif)

