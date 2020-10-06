# Tutorial: Quantized Softmax Network for the Lorenz 96 system

The two layer Lorenz 96 (L96) system is given by:

![equation](https://latex.codecogs.com/svg.latex?%5Cdot%7BX%7D_k%20%3D%20X_%7Bk-1%7D%28X_%7Bk&plus;1%7D%20-X_%7Bk-2%7D%29%20-%20X_k%20&plus;%20F%20&plus;%20B_k)

![equation](https://latex.codecogs.com/gif.latex?%5Cdot%7BY%7D_%7Bj%2Ck%7D%20%3D%20%5Cfrac%7B1%7D%7B%5Cepsilon%7D%5Cleft%5BY_%7Bj&plus;1%2Ck%7D%5Cleft%28Y_%7Bj-1%2Ck%7D%20-%20Y_%7Bj&plus;2%2Ck%7D%5Cright%20%29%20-%20Y_%7Bj%2Ck%7D%20&plus;%20h_yX_k%20%5Cright%5D)

![equation](https://latex.codecogs.com/svg.latex?B_k%20%3A%3D%20%5Cfrac%7Bh_x%7D%7BJ%7D%5Csum_%7Bj%3D1%7D%5E%7BJ%7D%20Y_%7Bj%2Ck%7D)

![equation](https://latex.codecogs.com/gif.latex?k%20%3D%201%2C%5Ccdots%2C%20K%20%5Cquad%5Cquad%20j%20%3D%201%2C%5Ccdots%2CJ)

It can be considered as a simplified atmospheric model on a circle of constant latitude. The `X_k` variables are the large-scale components of the system, whereas the `Y_{k,j}` are the small-scale counterparts. Each spatial location indexed by `k=1,...,K` has `J` small-scale components `Y_{k,j}`, with `j=1,...,J`. Thus the system consists of `JK` coupled ordinary differential equations (ODEs). In this tutorial we will use `K=18` and `J=20` such that we have 360 coupled ODEs. Finally, the `B_k` term is the subgrid-scale (SGS) term, through which the small-scale information enters the large-scale `X_k` ODEs. If we are able to create a surrogate for `B_k`, conditional on large-scale variables only, the dimension of the system drops from 360 down to 18. Here, we will create a surrogate model in the form of a quantized softmax network (QSN), which is stochastic in nature. 

Our general aim is to create a surrogate such that the long-term statistics of the large-scale system match those generated from validation data. Thus we do not expect accuracy from the large-scale `X_k` system forced by the QSN surrogate at any given point in time.

The details of the QSN approach can be found in [this](https://arxiv.org/abs/2004.01457) preprint.

## Files

The `tests/lorenz96_qsn` folder constains all required scripts to execute this tutorial: 

+ `tests/lorenz96_qsn/lorenz96.py`: the unmodified solver for the Lorenz 96 system, used to generate the training data.
+ `tests/lorenz96_qsn/train_surrogate.py`: script to train a QSN surrogate on L96 data.
+ `tests/lorenz96_qsn/lorenz96_qsn.py`: this is the L96 solver again, where the call to the small-scale system is replaced by a call to the QSN surrogate.
+ `tests/lorenz96_qsn/lorenz96_analysis.py`: the post-processing of the results of `lorenz96_qsn.py`.

Execute these script in the specified order to run the tutorial. Details are given below.

## Generate training data

The first step is to generate the training data, by executing `python3 tests/lorenz96_qsn/lorenz96.py`. This will generate training pairs that are used in `tests/lorenz96_qsn/train_surrogate.py`. You will be asked for a location to store the data (HDF5 format).

## Train the Quantized Softmax Network

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

Here, our large-scale features will the the `K` time series of the `X_k` variables, and our target data are the `K` times series of the subgrid-scale term `B_k`. The next step is to create a QSN surrogate object via:

```
# create Quantized Softmax Network surrogate
surrogate = es.methods.QSN_Surrogate()
```

At this point the specifics of a QSN surrogate come into play. One of our aims is to create a surrogate with 'memory', i.e. a surrogate that is non-Markovian. At the heart of our QSN surrogate is a feed-forward neural network. To add a memory dependence here, we create time-lagged feature vectors. Another means of doing so would be to use a recurrent neural network, which is an option planned for future releases. Say we wish to train a QSN surrogate using time-lagged input features at 1 and 10 time steps into the past. This is done via:

```
# create time-lagged features
lags = [[1, 10]]

# train the surrogate on the data
n_iter = 2000
surrogate.train([features], target, lags, n_iter, 
		n_bins=10, test_frac=0.5,
		n_layers=4, n_neurons=256, activation='leaky_relu', batch_size=512)
```

The `train` method should be supplied with a list of (different) input features, and an array of target data points, in this case an array of `nx18` subgrid-scale data points. Here, `n` is the number of training points. If `test_frac > 0` as above, the specified fraction of the data is withheld as a test set, lowering the value of `n`. Various aspects of the feed-forward neural network are defined here as well, such as the number of layers, the number of neurons per layers, the type of activation function and the minibatch size used in stochastic gradient descent. Other activation options are `tanh`, `hard_tan` and `relu`.

For each of the 18 spatial locations, the QSN surrogate will predict a discrete probability mass function (pmf) over `n_bins=10` non-overlapping `B_k` intervals or 'bins', for `k=1,...,K=18`. These 18 pmfs can then be sampled to identify 18 intervals of `B_k` data, conditional on the time-lagged, large-scale input features. To obtain a stochastic surrogate, `B_k` values are randomly resampled from the identified intervals. This process is repeated every time step. A movie of this can be found below, where the QSN surrogate is evaluated off-line on the training dataset. Left shows the QSN prediction for a single spatial location, alongside the bin of the training data. Right show the corresponding `B_k` time series, for both the stochastic QSN surrogate and the actual time evolution of the training data.

![alt text](qsn.gif)

To evaluate the classification error of (a subset of) the training data, an analysis object must be created:

```
# QSN analysis object
analysis = es.analysis.QSN_analysis(surrogate)
analysis.get_classification_error(features[0:1000], target[0:1000])
```
This will print the classifiction error to the screen for each of the 18 spatial locations. Once the surrogate is trained, it is saved and added to the campaign via

```
campaign.add_app(name='test_campaign', surrogate=surrogate)
campaign.save_state()
```

## Prediction with a QSN surrogate

To predict with a QSN surrogate, the original L96 code must be modified in 2 places, namely in the initial condition (IC) and the call to the micro model. Changing the IC is required due to the time-lagged nature. We use the data at the maximum lag specified as IC, such that we can build a time-lagged vector (also from the data) at the first time step. In `tests/lorenz96_qsn/lorenz96_qsn.py` we find:

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
Here, `campaign = es.Campaign(load_state=True)` loads the pre-trained QSN surrogate from `tests/lorenz96_qsn/train_surrogate.py`, and we use the HDF5 training data from 
`tests/lorenz96_qsn/lorenz96.py` to select the `X_k` and `B_k` snapshots at the timestep corresponding to the maximum lag that was specified. Upon finishing the training step, the corresponding first time-lagged feature vector is automatically stored in the surrogate.

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

Here, `B_n` is the current state of the subgrid-scale term, and `X_n` is the state of the large-scale variables. The subroutine `predict(X_n)` updates the time-lagged input features and returns a stochastic prediction for `B_n` as described above. The rest of the code is unmodified.

Note that due to the stochastic nature of the surrogate, as well as the chaotic nature of L96, we cannot expect that the trajectories of `X_k` and `B_k` will follow those of the full system. Below we show a movie of the time evolution of the stochastic QSN `B_k` (in the 'online' phase, when the QSN surrogate is a source term in the `X_k` ODEs) and the `B_k` of the full system. Eventually the two trajectories start to diverge. That said, we reiterate that our quantities of interest are time-averaged statistics, dicussed next.

![alt text](qsn_pred.gif)

## Analysis

One of the statistics we might be interested are the probability density functions (pdfs) of `X_k` and `B_k`, for both the full (validation) data set and the data set obtained in the 'online' phase from the preceding step. This is done via a `QSN_Analysis` object:

```python
#load the campaign
campaign = es.Campaign(load_state=True)
#load the training data (from lorenz96.py)
data_frame_ref = campaign.load_hdf5_data()
#load the data from lorenz96_qsn.py here
data_frame_qsn = campaign.load_hdf5_data()

# load reference data
X_ref = data_frame_ref['X_data']
B_ref = data_frame_ref['B_data']

# load data of QSN surrogate
X_qsn = data_frame_qsn['X_data']
B_qsn = data_frame_qsn['B_data']

# create QSN analysis object
analysis = es.analysis.QSN_analysis(campaign.surrogate)

#############
# Plot PDEs #
#############

start_idx = 0
fig = plt.figure(figsize=[8, 4])
ax = fig.add_subplot(121, xlabel=r'$X_k$')
X_dom_surr, X_pde_surr = analysis.get_pdf(X_qsn[start_idx:-1:10].flatten())
X_dom, X_pde = analysis.get_pdf(X_ref[start_idx:-1:10].flatten())
ax.plot(X_dom, X_pde, 'k+', label='L96')
ax.plot(X_dom_surr, X_pde_surr, label='QSN')
plt.yticks([])
plt.legend(loc=0)

ax = fig.add_subplot(122, xlabel=r'$B_k$')
B_dom_surr, B_pde_surr = analysis.get_pdf(B_qsn[start_idx:-1:10].flatten())
B_dom, B_pde = analysis.get_pdf(B_ref[start_idx:-1:10].flatten())
ax.plot(B_dom, B_pde, 'k+', label='L96')
ax.plot(B_dom_surr, B_pde_surr, label='QSN')
plt.yticks([])
plt.legend(loc=0)

plt.tight_layout()
```

Pre-generated statistical results are shown below:

![alt text](L96_pdf_QSN.png)

In `tests/lorenz96_qsn/lorenz96_analysis.py` other statistical quantities are also computed.
