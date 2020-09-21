# Tutorial: Easysurrogate structure

This tutorial describes some general features about EasySurrogate, without delving into the details of any specific surrogate method. Our overall target are multiscale problems, in which case the small-scale component of the simulation tends to be the one which carries the most computational burden, and is therefore an obvious candidate for replacement by a surrogate.

##Campaign object

EasySurrogate follows a similar design as [EasyVVUQ](https://github.com/UCL-CCS/EasyVVUQ), a library for forward uncertainty propagation. The overarching object is called a 'Campaign', and is created via

```python
import easysurrogate as es

# create EasySurrogate campaign
campaign = es.Campaign()
```

The data used to train a surrogate is regulated at the level of the campaign. In the current beta release, only Hierarchical Data Format version 5 (HDF5) data is supported. This is a common  and flexible option to store very large amounts of (scientific) data. Note that pandas dataframe can also be [stored in hd5f format](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_hdf.html). We use the [h5py](https://www.h5py.org/) libray under the hood. To load training data the following command is used:

```python
# load HDF5 data frame
data_frame = campaign.load_hdf5_data()
```

This will open up a file dialog in order to select an HDF5 file, and it will return a python dictionary with the training data. If `file_path=<path_to_hdf5_file>` is specified, the corresponding file will be loaded directly without opening a file dialog window.

## Surrogate methods

The implemented surrogate methods are stored in `es.methods`. To create for instance a Qantized Softmax Network, one executes

```python
# create Quantized Softmax Network surrogate
surrogate = es.methods.QSN_Surrogate()
```

Overall, every surrogate method has at least following subroutines:

+ `train`: train the surrogate on the data.
+ `predict`: make a prediction using a trained (small-scale) surrogate, conditional on some (large-scale) input features.
+ `save_state`: saves the entire state of the surrogate object to a `pickle` file. Data is not stored at the level of the surrogate.
+ `load_state`: loads a saved state.

By having a common structure we aim be able to easily test different surrogate methods in the 'online phase', when the surrogate is part of the multiscale simulation.

## Surrogate App

A (trained) surrogate is added to the campaign in an 'app', e.g.

```python
campaign.add_app(name='test_campaign', surrogate=surrogate)
```
 This is again a design feature borrowed from EasyVVUQ, where a Sampler object (e.g. a Monte Carlo sampler) is added via an app. Note that the campaign object also has a `save_state` subroutine. Once the surrogate is added to the app, `campaign.save_state` will also store the surrogate, so this does not have to be done seperately. The same is true for the `load state` subroutine.
 
## Analysis objects

Finally, the post-processing is handled by an 'Analysis' object. All analysis objects inherit from `es.analysis.BaseAnalysis`, which contains several common subroutines useful for the different surrogate methods. Currently, it contains subroutines for computing an auto-correlation function, a cross-correlation function and a kernel-density estimator. Various other general verification/verification 'patterns' could be added here in future releases, e.g. subroutines for comparing probability density functions. Seperate analysis classes exist for the different surrogate methods, which contains more specific analysis subroutines. Use for instance

```python
analysis = es.analysis.QSN_analysis(surrogate)
analysis.get_classification_error(features[0:1000], data[0:1000])
```
to compute the classification error of a trained QSN surrogate, using the first 1000 supervised data points.


