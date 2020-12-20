"""
==============================================================================
CLASS FOR A ARTIFICIAL NEURAL NETWORK
------------------------------------------------------------------------------
Author: W. Edeling
==============================================================================
"""
from itertools import chain
import numpy as np
# import h5py
from ..campaign import Campaign
import easysurrogate as es


class ANN_Surrogate(Campaign):

    def __init__(self, **kwargs):
        print('Creating ANN_Surrogate Object')
        self.name = 'ANN Surrogate'

    ############################
    # START COMMON SUBROUTINES #
    ############################

    def train(self, feats, target, n_iter, lags=None, local=False,
              test_frac=0.0,
              n_layers=2, n_neurons=100,
              activation='tanh',
              batch_size=64, lamb=0.0, **kwargs):
        """
        Perform back propagation to train the ANN

        Parameters
        ----------
        feats : feature array, or list of different feature arrays
        target : the target data
        lags : list of time lags per feature
        n_iter : number of back propagation iterations
        test_frac : Fraction of the data to use for training.
                    The default is 0.0.
        n_layers : The number of layers in the neural network. Includes the
                   input layer and the hidden layers. The default is 2.
        n_neurons : The number of neurons per layer. The default is 100.
        activation : Type of activation function. The default is 'tanh'.
        batch_size : Mini batch size. The default is 64.
        lamb : L2 regularization parameter. The default is 0.0.

        Returns
        -------
        None.

        """

        if not isinstance(feats, list):
            feats = [feats]

        # the number of distinct feature arrays
        self.n_feat_arrays = len(feats)
        # time lags
        self.lags = lags
        # flag if the surrogate is to be applied locally or not
        self.local = local
        
        # initialize storage for online training
        self.online_feats = [[] for i in range(self.n_feat_arrays)]
        self.online_target = []

        # Feature engineering object
        self.feat_eng = es.methods.Feature_Engineering()

        # number of training samples
        self.n_samples = feats[0].shape[0]
        #number of points in the computational grid
        self.n_points = feats[0].shape[1]
        # compute the size of the training set based on value of test_frac
        self.n_train = np.int(self.n_samples * (1.0 - test_frac))
        print('Using first %d/%d samples to train ANN' % (self.n_train, self.n_samples))

        X = {}; y = {}
        # use the entire row as a feature
        if not local:
            # list of features
            X[0] = [X_i[0:self.n_train] for X_i in feats]
            # the data
            y[0] = target[0:self.n_train]
        # do not use entire row as feature, apply surrogate locally along second dimension
        else:
            #create a separate training set for every grid point
            for i in range(self.n_points):
                X[i] = [X_i[0:self.n_train, i] for X_i in feats]
                y[i] = target[0:self.n_train, i].reshape([-1, 1])

        X_train = []; y_train = []
        #No time-lagged training data
        if lags is not None:
            self.max_lag = np.max(list(chain(*self.lags)))
            print('Creating time-lagged training data...')
            # lag every training set in X and y
            for i in range(len(X)):
                X_train_i, y_train_i = self.feat_eng.lag_training_data(X[i], y[i], lags=lags)
                X_train.append(X_train_i); y_train.append(y_train_i)
            X_train = np.concatenate(X_train)
            y_train = np.concatenate(y_train)                      
            print('done')
        else:
            self.max_lag = 0
            #no time lag, just add every entry in X and y to an array
            for i in range(len(X)):
                X_train.append(np.array(X[i]).reshape([self.n_train, -1]))
                y_train.append(y[i])
            X_train = np.concatenate(X_train)
            y_train = np.concatenate(y_train)                      

        # number of output neurons
        n_out = y_train.shape[1]

        # create the feed-forward ANN
        self.neural_net = es.methods.ANN(X=X_train, y=y_train,
                                        n_layers=n_layers, n_neurons=n_neurons,
                                        n_out=n_out,
                                        loss='squared',
                                        activation=activation, batch_size=batch_size,
                                        lamb=lamb, decay_step=10**4, decay_rate=0.9,
                                        standardize_X=True, standardize_y=True,
                                        save=False)

        print('===============================')
        print('Training Artificial Neural Network...')

        # train network for N_iter mini batches
        self.neural_net.train(n_iter, store_loss=True)
        self.set_data_stats()
        if lags is not None:
            self.init_feature_history(feats)

    def train_online(self, n_iter=1, batch_size=1, verbose=False, sequential=False, **kwargs):
        """
        Perform online training, i.e. backpropagation while the surrogate is coupled
        to the macroscopic governing equations.

        Source:        
        Rasp, "Coupled online learning as a way to tackle instabilities and biases 
        in neural network parameterizations: general algorithms and Lorenz 96 
        case study", 2020.

        Parameters
        ----------
        feats : feature array, or list of different feature arrays
        target : the target data
        n_iter : number of back propagation iterations
        batch_size : Mini batch size. Default is 1. 
        verbose: print loss to screen during back propagation. Default is False.
        sequential: do not randomly sample the training data, Default is False.

        Returns
        -------
        None.

        """

        feats = self.online_feats
        target = np.array(self.online_target)

        #if lagged feature vectors are used
        if self.lags is not None:
            
            feats = [np.array(feat) for feat in feats]
            if not self.local:
                #create (time-lagged) training data from X
                X_train, y_train = self.feat_eng.lag_training_data(feats, target, lags=self.lags,
                                                                    init_feats=False)
            else:
                X_train = []; y_train = []
                #create a separate training set for every grid point
                for i in range(self.n_points):
                    X_i = [X_i[:, i] for X_i in feats]
                    y_i = target[:, i].reshape([-1, 1])
                    #create time-lagged data per gridpoint
                    X_train_i, y_train_i = self.feat_eng.lag_training_data(X_i, y_i, lags=self.lags,
                                                                           init_feats=False)
                    X_train.append(X_train_i); y_train.append(y_train_i)
                X_train = np.concatenate(X_train)
                y_train = np.concatenate(y_train)                

            # #the number of different features arrays that are used as input features
            # n_feat_arrays = len(feats[0])
            # X = [];
            # for i in range(n_feat_arrays):
            #     # make a single array containing all arrays of the i-th feature vector
            #     all_feat_i = np.array([feat[i] for feat in feats])
            #     # a list where each entry contains all data of a given feature vector
            #     if not self.local:
            #         X.append(all_feat_i)
            #         y = target
            #         #create (time-lagged) training data from X
            #         X_train, y_train = self.feat_eng.lag_training_data(X, y, lags=self.lags,
            #                                                            init_feats=False)
            #     else:
            #         # X.append(all_feat_i.T.flatten())
            #         # y = target.T.flatten().reshape([-1, 1])
            #         X = {}; y = {}
            #         for i in range(n_points):
            #             X[i] = [feat]

        #do not time lag data
        else:
            # make a single array where each row contains a concateated vector of all feature
            # vectors
            X_train = np.array([np.concatenate(feat) for feat in feats])
            # reshape in case the neural network is applied locally
            X_train = X_train.reshape([-1, self.neural_net.n_in])
            y_train = target.reshape([-1, self.neural_net.n_out])

        #set the training data, training size and bathc size for the online backprop step
        if not self.neural_net.batch_size == batch_size:
            self.neural_net.set_batch_size(batch_size)
        self.neural_net.n_train = X_train.shape[0]
        #standardize training data
        self.neural_net.X = (X_train - self.feat_mean) / self.feat_std
        self.neural_net.y = (y_train - self.output_mean) / self.output_std

        # train network for n_iter mini batches
        self.neural_net.train(n_iter, store_loss=True, sequential=sequential, verbose=verbose)

        #clear the old training data
        # self.online_feats = []; self.online_target = []
        
    def generate_online_training_data(self, feats,  LR_before, LR_after, HR_before, HR_after):
        """
        Compute the features and the target data for an online training step. Results are
        stored internally, and used within the 'train_online' subroutine.
        
        Source:        
        Rasp, "Coupled online learning as a way to tackle instabilities and biases 
        in neural network parameterizations: general algorithms and Lorenz 96 
        case study", 2020.

        Parameters
        ----------
        feats : array or list of arrays
            The input features
        LR_before : array
            Low resolution state at previous time step.
        LR_after : array
            Low resolution state at current time step.
        HR_before : array
            High resolution state at previous time step.
        HR_after : array
            High resolution state at current time step.

        Returns
        -------
        None.

        """

        #multiple faetures arrays are stored in a list. For consistency put a single
        #arrays also in a list.
        if type(feats) is np.ndarray:
            feats = [feats]

        # #make sure that all arrays in the faeture list have ndim equal to 1
        # for i in range(len(feats)):
        #     if feats[i].ndim != 1: feats[i] = feats[i].flatten()

        #store input features
        for i in range(self.n_feat_arrays):
            self.online_feats[i].append(feats[i])

        # difference of the low res model between time n and time n+1
        delta_LR = LR_after - LR_before
        # the difference between the low res and high res model at time n
        delta_nudge = LR_before - HR_before
        # difference of the high res model between time n and time n+1
        delta_HR = HR_after - HR_before

        # assume that over a small time interval [n, n+1] we can write: 
        # delta_HR = would_be_delta_without_nuding + delta_due_to_nudging
        delta_no_nudge_HR = delta_HR - delta_nudge / self.tau_nudge * self.dt_LR

        # compute correction from: delta_LR + correction = delta_no_nudge_HR
        correction = delta_no_nudge_HR - delta_LR
        
        # make the correction the target for the neural network. Divide by timestep
        # since update is LR += correction * dt
        self.online_target.append(correction / self.dt_LR)
        
        # remove oldest item from the online features is the window lendth is exceeded
        if len(self.online_feats) > self.window_length:
            self.online_feats.pop(0)
            self.online_target.pop(0)
        
    def set_online_training_parameters(self, tau_nudge, dt_LR, window_length):
        """
        Stores the nudiging time scale and time step of the low resolution model. Required
        for online training.

        Parameters
        ----------
        tau_nudge : float
            Nudging time scale.
        dt_LR : float
            Time step low resolution model.

        Returns
        -------
        None.

        """
        self.tau_nudge = tau_nudge
        self.dt_LR = dt_LR
        self.window_length = window_length

    def predict(self, X):
        """
        Make a stochastic prediction of the output y conditional on the
        input features X

        Parameters
        ----------
        X: the state at the current (time) step. If the ANN is conditioned on
        more than 1 (time-lagged) variable, X must be a list containing all
        variables at the current time step.

        Returns
        -------
        Prediction of the output y

        """

        if not isinstance(X, list):
            X = [X]
            
        # make sure all feature vectors have the same ndim.
        # This will raise an error when for instance X1.shape = (10,) and X2.shape = (10, 1)
        ndims = [X_i.ndim for X_i in X]
        assert all([ndim == ndims[0] for ndim in ndims]), "All features must have the same ndim"
        
        # make sure features are at most two dimensional arrays
        assert ndims[0] <= 2, "Only 1 or 2 dimensional arrays are allowed as features."

        # in the case of two dimensional arrays, make are the second dimension is the same 
        # for all features. This dimension must equal the grid size
        local = False
        if ndims[0] == 2:
            local = True
            shapes1 = [X_i.shape[1] for X_i in X]
            assert all([shape1 == shapes1[0] for shape1 in shapes1]), \
                   "The size of the second dimension must be the same for all features."
            # if a second dimension is specified, it is assumed that we must loop over this 
            # dimension in order to make a single prediction
            n_points = shapes1[0]

        if self.lags is not None:
            
            # append the current state X to the feature history
            self.feat_eng.append_feat(X)
            
            # get the feature history defined by the specified number of time lags.
            # Here, feat is an array with the same size as the neural network input layer
            if not local:
                feat = self.feat_eng.get_feat_history()
                return self._feed_forward(feat)
            else:
                y = []
                for p in range(n_points):
                    feat = self.feat_eng.get_feat_history(index=p)
                    y.append(self._feed_forward(feat))
                return np.array(y).flatten()
        else:
            if not local:
                feat = np.concatenate(X)
                return self._feed_forward(feat)
            else:
                X = np.array(X).reshape([n_points, len(X)])
                y = []
                for p  in range(n_points):
                    y.append(self._feed_forward(X[p]))
                return np.array(y).flatten()

    def _feed_forward(self, feat):
        # features were standardized during training, do so here as well
        feat = (feat - self.feat_mean) / self.feat_std
        # feed forward prediction step
        y = self.neural_net.feed_forward(feat.reshape([1, self.neural_net.n_in])).flatten()
        # transform y back to physical domain
        y = y * self.output_std + self.output_mean

        return y

    def save_state(self):
        """
        Save the state of the QSN surrogate to a pickle file
        """
        state = self.__dict__
        super().save_state(state=state, name=self.name)

    def load_state(self):
        """
        Load the state of the QSN surrogate from file
        """
        super().load_state(name=self.name)

    ##########################
    # END COMMON SUBROUTINES #
    ##########################

    def set_data_stats(self):
        """
        If the data were standardized, this stores the mean and
        standard deviation of the features in the ANN_Surrogate object.
        """
        if hasattr(self.neural_net, 'X_mean'):
            self.feat_mean = self.neural_net.X_mean
            self.feat_std = self.neural_net.X_std
        else:
            self.feat_mean = 0.0
            self.feat_std = 1.0

        if hasattr(self.neural_net, 'y_mean'):
            self.output_mean = self.neural_net.y_mean
            self.output_std = self.neural_net.y_std
        else:
            self.output_mean = 0.0
            self.output_std = 1.0

    def init_feature_history(self, feats, start=0):
        """
        The features are assumed to be lagged in time. Therefore, the initial
        time-lagged feature vector must be set up. The training data is used
        for this.

        Parameters
        ----------
        + feats : a list of the variables used to construct the time-lagged
        features: [var_1, var_2, ...]. Each var_i is an array such that
        var_i[0] gives the value of var_i at t_0, var_i[1] at t_1 etc.

        + start : the starting index of the training features. Default is 0.

        Returns
        -------
        None.

        """
        for i in range(self.max_lag):
            if not self.local:
                feat = [X_i[start + i] for X_i in feats]
            else:
                feat = [X_i[start + i].reshape([1, -1]) for X_i in feats]
                
            self.feat_eng.append_feat(feat)
