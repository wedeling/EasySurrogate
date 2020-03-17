# Tutorial: Quantized Softmax Network for the Lorenz 96 system

The two layer Lorenz 96 system is given by:

![equation](https://latex.codecogs.com/svg.latex?%5Cdot%7BX%7D_k%20%3D%20X_%7Bk-1%7D%28X_%7Bk&plus;1%7D%20-X_%7Bk-2%7D%29%20-%20X_k%20&plus;%20F%20&plus;%20B_k)

![equation](https://latex.codecogs.com/gif.latex?%5Cdot%7BY%7D_%7Bj%2Ck%7D%20%3D%20%5Cfrac%7B1%7D%7B%5Cepsilon%7D%5Cleft%5BY_%7Bj&plus;1%2Ck%7D%5Cleft%28Y_%7Bj-1%2Ck%7D%20-%20Y_%7Bj&plus;2%2Ck%7D%5Cright%20%29%20-%20Y_%7Bj%2Ck%7D%20&plus;%20h_yX_k%20%5Cright%5D)

![equation](https://latex.codecogs.com/svg.latex?B_k%20%3A%3D%20%5Cfrac%7Bh_x%7D%7BJ%7D%5Csum_%7Bj%3D1%7D%5E%7BJ%7D%20Y_%7Bj%2Ck%7D)

![equation](https://latex.codecogs.com/gif.latex?k%20%3D%201%2C%5Ccdots%2C%20K%20%5Cquad%5Cquad%20j%20%3D%201%2C%5Ccdots%2CJ)

Here B_k is the subgrid-scale (SGS) term for which we create a surrogate model in the form of a quantized softmax network, see the paper (referenced [here](./README.md)) for details.

## Files

The `tests/` folder constains following numerical experiments: 

+ `tests/lorenz96/lorenz96.py`: the solver for the Lorenz 96 system
+ `tests/lorenz96/lorenz96_quantized_softmax.py`: applies a quantized softmax network to learn subgrid-scale term of the Lorenz96 system.
+ `tests/lorenz96/figures`: contains the figures of the results
+ `tests/lorenz96/movies`: contains the movies of the results

To recreate the results, perform the following steps

## Generate training data

The first step is to generate the training data, by executing `python3 tests/lorenz96/lorenz96.py`. This will generate training pairs that are used in `tests/lorenz_96/lorenz96_quantized_softmax.py`. You will be asked for a location to store the data (HDF5 format).

## Setup the Quantized Softmax Network

The first step is to time lag the generated training data. This is done via:

```python
#Feature engineering object - loads data file
feat_eng = es.methods.Feature_Engineering(load_data = True)

#get training data
h5f = feat_eng.get_hdf5_file()

#Large-scale and SGS data - convert to numpy array via [()]
X_data = h5f['X_data'][()]
B_data = h5f['B_data'][()]

#Lag features as defined in 'lags'
lags = [[1, 10]]
X_train, y_train = feat_eng.lag_training_data([X_data], B_data, lags = lags)
```

+ `feat_eng` is a `Feature_Engineering` object that we (amongst others) use to lag the training data.
+ `lags` is a nested list of time lags. Every conditioning variable has one list of time lags in `lags`. Since we only condition on X here, there is only a single list. In this example we lag X by 1 and 10 time steps.
+ `X_train` are the time-lagged training features. In this example, every entry consists of 2 X *vectors* (each of size K), one lagged behind the corresponding SGS data B by one time step, and the other vector by 10 steps.
+ `y_train` are the SGS data vectors B (size K).

Next, we divide the SGS data B_k (at each spatial location index by k) into `n_bins` non-overlapping bins, and create one-hot encoded training data. This is done via:

```python
#number of bins per B_k
n_bins = 10
#one-hot encoded training data per B_k
feat_eng.bin_data(y_train, n_bins)
#simple sampler to draw random samples from the bins
sampler = es.methods.SimpleBin(feat_eng)
```

+ `feat_eng.bin_data(y_train, n_bins)` creates the one-hot encoded data from y_train. Here, we bin each B_k into 10 bins, meaning that the number of output neurons of the QSN is 10K (K=18 in our L96 setup). The one-hot encoded data is stored in `feat_eng.y_idx_binned`.
+ `sampler` is a `SimpleBin` object that is used to draw random samples from the bins identified by the QSN output.

Finally, a QSN object is created via:

```
#number of softmax layers (one per output location k)
n_softmax = K

#number of output neurons 
n_out = n_bins*n_softmax

#train the neural network
surrogate = es.methods.ANN(X=X_train, y=feat_eng.y_idx_binned, n_layers=4, n_neurons=256, 
                           n_softmax = K, n_out=K*n_bins, loss = 'cross_entropy',
                           activation='hard_tanh', batch_size=512,
                           lamb=0.0, decay_step=10**4, decay_rate=0.9, 
                           standardize_X=False, standardize_y=False, save=True)
```
This is all fairly self explanatory. We use an in-house Artificial Neural Network (ANN) code to create a QSN. This consists of a feed-forward network of a specified number of layers and neurons per layer, with K separate softmax layers at the output to predict the probability mass function (pmf) of the bins, for each B_k. We can sample the binnumbers from these pmfs, or just take the bin with the highest probablity. These binnumbers are then fed to the `sampler` to randomly resample reference data from the identified bins.

## Training / predicting with the Quantized Softmax Network

As mentioned, the file `tests/lorenz_96/lorenz96_quantized_softmax.py` contains the QSN applied in conjunction with the L96 system. It contains the following flags:

```python
train = True            #train the network
make_movie = False      #make a movie (of the training)
predict = True          #predict using the learned SGS term
store = True            #store the prediction results
make_movie_pred = False #make a movie (of the prediction)
```

+ `train` (Boolean): train the QSN network on L96 data.
+ `make_movie` (Boolean): make a movie as displayed below. This is the QSN evaluated on the *training* data.

![alt text](https://github.com/wedeling/EasySurrogate/blob/phys_D/tests/movies/qsn.gif)

+ `predict` (Boolean): use the QSN to replace the subgrid-scale term of the L96 system. Only the large scale X equation is solved.
+ `store` (Boolean): store the prediction results
+ `make_movie_pred`: make a movie of the *prediction* for one spatial point. Example shown below, which shows eventual divergence of the data and prediction trajectories due to the chaotic nature of L96.

![alt text](https://github.com/wedeling/EasySurrogate/blob/phys_D/tests/movies/qsn_pred.gif)

Pre-generated statistical results van be found in `tests/figures`, e.g. the probability density function of X_k computed from the full two-layer L96 system and the one-layer model forced by the QSN SGS model:

![alt text](https://github.com/wedeling/EasySurrogate/blob/phys_D/tests/figures/QSM_pdf.png)
