
#Gaussian Process for Fusion transport tutorial example


#### Genrate training data 

First step to apply a surrogate is to generate training data.

In the first example we consider a surroagate trained on a one-shot design of numerical experiments

### Train the model

To train the model load the data from the run of numerical solver 


## Analysis of the surrogate performance

#### Accuracy of surrogate on testing set

In case when we have only information in the quasy-steady-state of the modeled system or do not have information
about any time evolution, we can estimate the accuracy of the surrogate as the estimator for the quantitites of 
interest we trained it on.

In context of uncertainty quantification we will be interested in Probability Density Functions of QOi and its moments.
Considering the given exemplary physical model, we might chose a scalar value that is interesting to analyse as
the "T_{e}(\rho=0)", which is the temperature in the core of the thermonuclear reactor in the point where it is
potentially highest, but also most uncertain.