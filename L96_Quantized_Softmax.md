# Tutorial: Quantized Softmax Network for the Lorenz 96 system

![alt text](https://github.com/wedeling/EasySurrogate/blob/phys_D/tests/movies/qsn.gif)

Left: The Quantized Softmax Network (QSN) prediction for one spatial location X_k. Right: the time series of the subgrid-scale data and the QSN prediction.

## Files

The `tests/` folder constains following numerical experiments: 

+ `tests/lorenz96/lorenz96.py`: the solver for the Lorenz 96 system
+ `tests/lorenz_96/lorenz96_quantized_softmax.py`: applies a quantized softmax network to learn subgrid-scale term of the Lorenz96 system.

To recreate the results, perform the following steps

## Generate training data

Pre-generated training data for the Lorenz96 system is available in `test/samples/L96_training.hdf5`. If you wish to regenerate the data, execute `python3 tests/lorenz96/lorenz96.py`. This will generate training pairs that are used in `tests/lorenz_96/lorenz96_quantized_softmax.py`. You will be asked for a location to store the data (HDF5 format).

## Train the Quantized Softmax Network (QSN)

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
+ `predict` (Boolean): use the QSN to replace the subgrid-scale term of the L96 system.
+ `store` (Boolean): store the prediction results
+ `make_movie_pred`: make a movie of the *prediction* for one spatial point. Example shown below, which shows eventual divergence of the data and prediction trajectories due to the chaotic nature of L96.

![alt text](https://github.com/wedeling/EasySurrogate/blob/phys_D/tests/movies/qsn_pred.gif)

