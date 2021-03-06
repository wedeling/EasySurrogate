# Tutorial: A vanilla Artificial Neural Network for the Lorenz 96 system

The two layer Lorenz 96 (L96) system is given by:

![equation](https://latex.codecogs.com/svg.latex?%5Cdot%7BX%7D_k%20%3D%20X_%7Bk-1%7D%28X_%7Bk&plus;1%7D%20-X_%7Bk-2%7D%29%20-%20X_k%20&plus;%20F%20&plus;%20B_k)

![equation](https://latex.codecogs.com/gif.latex?%5Cdot%7BY%7D_%7Bj%2Ck%7D%20%3D%20%5Cfrac%7B1%7D%7B%5Cepsilon%7D%5Cleft%5BY_%7Bj&plus;1%2Ck%7D%5Cleft%28Y_%7Bj-1%2Ck%7D%20-%20Y_%7Bj&plus;2%2Ck%7D%5Cright%20%29%20-%20Y_%7Bj%2Ck%7D%20&plus;%20h_yX_k%20%5Cright%5D)

![equation](https://latex.codecogs.com/svg.latex?B_k%20%3A%3D%20%5Cfrac%7Bh_x%7D%7BJ%7D%5Csum_%7Bj%3D1%7D%5E%7BJ%7D%20Y_%7Bj%2Ck%7D)

![equation](https://latex.codecogs.com/gif.latex?k%20%3D%201%2C%5Ccdots%2C%20K%20%5Cquad%5Cquad%20j%20%3D%201%2C%5Ccdots%2CJ)

It can be considered as a simplified atmospheric model on a circle of constant latitude. The `X_k` variables are the large-scale components of the system, whereas the `Y_{k,j}` are the small-scale counterparts. Each spatial location indexed by `k=1,...,K` has `J` small-scale components `Y_{k,j}`, with `j=1,...,J`. Thus the system consists of `JK` coupled ordinary differential equations (ODEs). In this tutorial we will use `K=18` and `J=20` such that we have 360 coupled ODEs. Finally, the `B_k` term is the subgrid-scale (SGS) term, through which the small-scale information enters the large-scale `X_k` ODEs. If we are able to create a surrogate for `B_k`, conditional on large-scale variables only, the dimension of the system drops from 360 down to 18. Here, we will create a surrogate model in the form of a Artificial Neural Network (ANN).

Our general aim is to create a surrogate such that the long-term statistics of the large-scale system match those generated from validation data. Thus we do not expect accuracy from the large-scale `X_k` system forced by the ANN surrogate at any given point in time.

**NOTE**: This tutorial is very similar to the Lorenz 96 Quantized Softmax Network tutorial, the only difference is that the stochastic, time-lagged QSN surrogate is replaced by a deterministic, time-lagged ANN surrogate.

## Files

The `tests/lorenz96_ann` folder constains all required scripts to execute this tutorial: 

+ `tests/lorenz96_ann/lorenz96.py`: the unmodified solver for the Lorenz 96 system, used to generate the training data.
+ `tests/lorenz96_ann/train_surrogate.py`: script to train a QSN surrogate on L96 data.
+ `tests/lorenz96_ann/lorenz96_ann.py`: this is the L96 solver again, where the call to the small-scale system is replaced by a call to the ANN surrogate.
+ `tests/lorenz96_ann/lorenz96_analysis.py`: the post-processing of the results of `lorenz96_ann.py`.

Execute these script in the specified order to run the tutorial. Details are given below.

## Generate training data

The first step is to generate the training data, by executing `python3 tests/lorenz96_ann/lorenz96.py`. This will generate training pairs that are used in `tests/lorenz96_ann/train_surrogate.py`. You will be asked for a location to store the data (HDF5 format).

## Train the Artificial Neural Network

As explained in the general EasySurrogate tutorial (`tutorials/General`), we begin by creating a EasySurrogate campaign, and loading the training data:

```python
# create EasySurrogate campaign
campaign = es.Campaign()

# load HDF5 data frame
data_frame = campaign.load_hdf5_data()

# supervised training data set
features = data_frame['X_data']
target = data_frame['B_data']
```

Here, our large-scale features will the the `K` time series of the `X_k` variables, and our target data are the `K` times series of the subgrid-scale term `B_k`. The next step is to create an ANN surrogate object via:

```
# create a (time-lagged) ANN surrogate
surrogate = es.methods.ANN_Surrogate()
```

One of our aims is to create a surrogate with 'memory', i.e. a surrogate that is non-Markovian. To add a memory dependence here, we create time-lagged feature vectors. Another means of doing so would be to use a recurrent neural network, which is an option planned for future releases. Say we wish to train an ANN surrogate using time-lagged input features at 1 and 10 time steps into the past. This is done via:

```
# create time-lagged features
lags = [[1, 10]]

# train the surrogate on the data
n_iter = 10000
surrogate.train([features], target, lags, n_iter, n_layers=4, n_neurons=256,
                batch_size=512)
```

The `train` method should be supplied with a list of (different) input features, and an array of target data points, in this case an array of `nx18` subgrid-scale data points. Here, `n` is the number of training points. If `test_frac > 0` as above, the specified fraction of the data is withheld as a test set, lowering the value of `n`. Various aspects of the feed-forward neural network are defined here as well, such as the number of layers, the number of neurons per layers, the type of activation function and the minibatch size used in stochastic gradient descent. Activation options are `tanh` (Default), `hard_tan`, `relu` and `leaky_relu`.

Once the surrogate is trained, it is saved and added to the campaign via

```
campaign.add_app(name='test_campaign', surrogate=surrogate)
campaign.save_state()
```

## Prediction with an ANN surrogate

To predict with an ANN surrogate, the original L96 code must be modified in 2 places, namely in the initial condition (IC) and the call to the micro model. Changing the IC is required due to the time-lagged nature. We use the data at the maximum lag specified as IC, such that we can build an initial time-lagged vector (also from the data) at the first time step. In `tests/lorenz96_ann/lorenz96_ann.py` we find:

```python
##############################
# Easysurrogate modification #
##############################

# load pre-trained campaign
campaign = es.Campaign(load_state=True)

# change IC
data_frame = campaign.load_hdf5_data()
X_n = data_frame['X_data'][campaign.surrogate.max_lag]
B_n = data_frame['B_data'][campaign.surrogate.max_lag]
# initial right-hand side
f_nm1 = rhs_X(X_n, B_n)

##################################
# End Easysurrogate modification #
##################################
```
Here, `campaign = es.Campaign(load_state=True)` loads the pre-trained ANN surrogate from `tests/lorenz96_ann/train_surrogate.py`, and we use the HDF5 training data from `tests/lorenz96_ann/lorenz96.py` to select the `X_k` and `B_k` snapshots at the timestep corresponding to the maximum lag that was specified. Upon finishing the training step, the corresponding first time-lagged feature vector is automatically stored in the surrogate.

The second modification involves replacing the call to the surrogate with a call to `surrogate.predict`:

```python
##############################
# Easysurrogate modification #
##############################

# Turn off call to small-scale model
# solve small-scale equation
# Y_np1, g_n, multistep_n = step_Y(Y_n, g_nm1, X_n)
# compute SGS term
# B_n = h_x*np.mean(Y_n, axis=0)

# replace SGS call with call to surrogate
B_n = campaign.surrogate.predict(X_n)

##################################
# End Easysurrogate modification #
##################################
```

Here, `B_n` is the current state of the subgrid-scale term, and `X_n` is the state of the large-scale variables. The subroutine `predict(X_n)` updates the time-lagged input features and returns a prediction for `B_n`. The rest of the code is unmodified. Moreover, this file is completely the same as its counterpart in the L96 QSN tutorial. We can therefore train different surrogates, and apply them to the same problem without further modification.

## Analysis

One of the statistics we might be interested are the probability density functions (pdfs) of `X_k` and `B_k`, for both the full (validation) data set and the data set obtained in the 'online' phase from the preceding step. This is done via a `Base_Analysis` object:

```python
# load the campaign
campaign = es.Campaign(load_state=True)
# load the training data (from lorenz96.py)
data_frame_ref = campaign.load_hdf5_data()
# load the data from lorenz96_ann.py here
data_frame_ann = campaign.load_hdf5_data()

# load reference data
X_ref = data_frame_ref['X_data']
B_ref = data_frame_ref['B_data']

# load data of ann surrogate
X_ann = data_frame_ann['X_data']
B_ann = data_frame_ann['B_data']

# create ann analysis object
analysis = es.analysis.BaseAnalysis()

#############
# Plot PDFs #
#############

start_idx = 0
fig = plt.figure(figsize=[8, 4])
ax = fig.add_subplot(121, xlabel=r'$X_k$')
X_dom_surr, X_pde_surr = analysis.get_pdf(X_ann[start_idx:-1:10].flatten())
X_dom, X_pde = analysis.get_pdf(X_ref[start_idx:-1:10].flatten())
ax.plot(X_dom, X_pde, 'k+', label='L96')
ax.plot(X_dom_surr, X_pde_surr, label='ann')
plt.yticks([])
plt.legend(loc=0)

ax = fig.add_subplot(122, xlabel=r'$B_k$')
B_dom_surr, B_pde_surr = analysis.get_pdf(B_ann[start_idx:-1:10].flatten())
B_dom, B_pde = analysis.get_pdf(B_ref[start_idx:-1:10].flatten())
ax.plot(B_dom, B_pde, 'k+', label='L96')
ax.plot(B_dom_surr, B_pde_surr, label='ann')
plt.yticks([])
plt.legend(loc=0)

plt.tight_layout()
```

Pre-generated statistical results are shown below:

![alt text](L96_pdf_ANN.png)

In `tests/lorenz96_ann/lorenz96_analysis.py` other statistical quantities are also computed.
