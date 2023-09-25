"""
CLASS FOR A GAUSSIAN PROCESS REGRESSION
------------------------------------------------------------------------------
Author: Y. Yudin
==============================================================================
"""

import numpy as np
import functools

from ..campaign import Campaign
import easysurrogate as es

from sklearn.preprocessing import StandardScaler


class GP_Surrogate(Campaign):

    def __init__(self, backend='scikit-learn', **kwargs):
        """
        GP_Surrogate class for Gaussian Process Regression

        """
        print('Creating Gaussian Process Object')

        self.name = 'GP Surrogate'
        self.feat_eng = es.methods.Feature_Engineering()
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.backend = backend

        #TODO: put all model-related parameters EITHER in constructor OR in .train() method
        """
        if 'noize' in kwargs: 
            self.noize = kwargs['noize']
        else:
            self.noize = False
        """
        
        if 'n_in' in kwargs:
            self.n_in = kwargs['n_in']
        else:
            self.n_in = 1
        
        if 'n_out' in kwargs:
            self.n_out = kwargs['n_out']
        else:
            self.n_out = 1

        """
        if 'process_type' in kwargs:
            self.process_type = kwargs['process_type']
        else:
            self.process_type = 'gaussian'
        """

    ############################
    # START COMMON SUBROUTINES #
    ############################

    def train(self, feats, target, n_iter=0,
              test_frac=0.0,
              **kwargs):
        """

        Args:
            feats: feature array, or list of different feature arrays
            target: the target data
            n_iter: number of hyperoptimisation restarts
            test_frac: Fraction of the data used for training

        Returns:
        -------
        None.
        """

        if 'kernel' not in kwargs:
            self.kernel = 'Matern'
        else:
            self.kernel = kwargs['kernel']

        if 'length_scale' not in kwargs:
            self.length_scale = 1.0
        else:
            self.length_scale = kwargs['length_scale']

        if 'nu_matern' not in kwargs:
            self.nu_matern = 1.0
        else:
            self.nu_matern = kwargs['nu_matern']

        if 'nu_stp' not in kwargs:
            self.nu_stp = 5
        else:
            self.nu_stp = kwargs['nu_stp']

        if 'noize' not in kwargs:
            self.noize = 'True'
        else:
            self.noize = kwargs['noize']

        if 'bias' not in kwargs:
            self.bias = False
        else:
            self.bias = kwargs['bias']

        if 'nonstationary' not in kwargs:
            self.nonstationary = False
        else: 
            self.nonstationary = kwargs['nonstationary']         

        if 'process_type' not in kwargs:
            self.process_type = 'gaussian'
        else:
            self.process_type = kwargs['process_type']
        
        X_train, y_train, X_test, y_test = self.feat_eng.get_training_data(
            feats, target, local=False, test_frac=test_frac, train_first=False)

        # scale the training data
        X_train = self.x_scaler.fit_transform(X_train)
        y_train = self.y_scaler.fit_transform(y_train)
        if len(X_test) > 0 and len(y_test) > 0:
            X_test = self.x_scaler.transform(X_test)
            y_test = self.y_scaler.transform(y_test)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # create a GP process
        print('===============================')
        print('Fitting Gaussian Process...')
        print(f">GP_Surrogate: self.n_out={self.n_out}") ###DEBUG
        self.model = es.methods.GP(
            kernel=self.kernel,
            n_in=self.n_in,
            n_out=self.n_out,
            bias=self.bias, # BIAS should not matter and yield factor parameter close to zero if data is whitened
            nonstationary=self.nonstationary,
            noize=self.noize,
            length_scale=self.length_scale,
            backend=self.backend,
            process_type=self.process_type,
            nu_matern=self.nu_matern,
            nu_stp=self.nu_stp
                                  ) 

        # get dimensionality of the output
        self.n_out = y_train.shape[1]
        print(f">GP_Surrogate: self.n_out={self.n_out}") ###DEBUG

        # get the dimesionality of te input
        self.n_in = X_train.shape[1]

        self.model.train(self.X_train, self.y_train)

    def predict(self, X):
        """
        Make a stochastic prediction of the output y conditional on the
        input features [X]

        Args:
            X: list of feature arrays, the state given at the point

        Returns:
        -------
        Stochastic prediction of the output y
        """
        # TODO slows down a lot, maybe FeatureEngineering should return training data still as a list
        x = np.array([x for x in X]).T # TODO: if no transformation needed, then list comprehension not needed
        x = self.x_scaler.transform(x)
        x = [np.array(i) for i in x.T.tolist()]

        # TODO unlike ANNs, GPs should provide API for vectorised .predict() and other methods
        y, std, _ = self.feat_eng._predict(x, feed_forward=lambda t: self.model.predict(t))

        print(f"> y in gp_surrogate.predict: {y}") ###DEBUG

        y = self.y_scaler.inverse_transform(y)

        self.y_scaler.with_mean = False
        std = self.y_scaler.inverse_transform(std * np.ones(y.shape))
        self.y_scaler.with_mean = True

        print(f">GP_Surrogate: y={y}") ###DEBUG
        return y, std

    def save_state(self, state=None, **kwargs):
        """
        Save the state of GP surrogate as a pickle file
        """
        state = self.__dict__
        save_state = super().save_state(state=state, name=self.name)

    def load_state(self):
        """
        Load the state of the GP surrogate from file
        """
        super.load_state(name=self.name)

    ##########################
    # END COMMON SUBROUTINES #
    ##########################

    def set_data_stats(self):
        """
        If the data were standardized, this stores the mean and
        standard derivation of the features in the GP_Surrogate object.
        """
        if hasattr(self.model, 'X_mean'):
            self.feat_mean = self.model.X_mean
            self.seat_Std = self.model.X_std
        else:
            self.feat_mean = 0.0 * np.ones((1, self.n_in))
            self.feat_std = 1.0 * np.ones((1, self.n_in))

        if hasattr(self.model, 'y_mean'):
            self.output_mean = self.model.y_mean
            self.output_std = self.model.y_std
        else:
            self.output_mean = 0.0 * np.ones((1, self.n_out))
            self.output_std = 1.0 * np.ones((1, self.n_out))
    
    def get_dimensions(self):
        """
        Get some useful dimensions of the GPR surrogate. Returns a dict with the number
        of training samples (n_train), the number of data samples (n_samples),
        the number of test samples (n_test), the size of input vector (n_in),
        and the size of output vector (n_out).

        Returns
        -------
        dims : dict
            The dimensions dictionary.

        """

        dims = {}
        dims['n_train'] = self.feat_eng.n_train
        dims['n_samples'] = self.feat_eng.n_samples
        dims['n_test'] = dims['n_samples'] - dims['n_train']
        dims['n_in'] = self.n_in
        dims['n_out'] = self.n_out

        return dims
   
    def train_sequentially(self, feats=None, target=None,
                           n_iter=0, **kwargs):
        """
        Update GP surrogate with a sequential design scheme
        if n_iter==1 finds a suggestion for new candidates for which to evaluate function 

        Parameters
        ----------
        feats: list of feature arrays
        target: array of target data
        n_iter: integer, number of iterations of sequential optimisation
        """

        self.set_data_stats()

        #TODO: in every function specs write in which scaling the passed data are
        target = self.y_scaler.transform(np.array(target).reshape(1,-1)) 
        #X_test_unscaled = [self.x_scaler.inverse_transform(X_i.reshape(1,-1))[0].tolist() for X_i in self.X_test]
        #TODO: this is horrible and should not exist

        #print('Comparing scaled and unscaled candidate lists: {0} and {1} \n'.format(self.X_test, X_test_unscaled)) ###DEBUG

        if 'acquisition_function' in kwargs:
            acq_func_arg = kwargs['acquisition_function']
            if acq_func_arg == 'poi':
                acq_func_obj = self.poi_acquisition_function
            elif acq_func_arg == 'mu':
                acq_func_obj = self.maxunc_acquisition_function
            elif acq_func_arg == 'poi_sq_dist_to_val' and target is not None:
                #TODO currently simplified to less general version of function, not a metric anymore
                #acq_func_obj = functools.partial(self.poi_function_acquisition_function, func=(lambda y1,y2: np.power(y1-y2, 2)), target=target)
                acq_func_obj = functools.partial(self.poi_function_acquisition_function, func=(lambda y: np.power(y-target, 2)), target=target)
            else:
                raise NotImplementedError(
                    'This rule for sequential optimisation is not implemented, using default.')
        else:
            acq_func_obj = self.maxunc_acquisition_function

        if 'save_history' in kwargs:
            save_history = kwargs['save_history']
        else:
            save_history = False

        if save_history:
            self.design_history = []

        if 'savefile_postfix' in kwargs:
            file_path_postfix = '_' + kwargs['savefile_postfix']
        else:
            file_path_postfix = ''

        if self.backend == 'scikit-learn' or self.backend == 'local':

            """
            0) iterate for n_iter
                1) state on step n: object has X_train, X_test, their indices, model instance
                2) find set of candidates at minima of acq function X_cand; now object has
                    X_train:=X_train U X_cand, X_test = X_test U_ X_cand, global inidces and set sizes updated
                3) model instance is updated : first approach to train new model for new X_train
            """

            if n_iter == 1:

                X_new, x_new_ind_test, x_new_ind_glob = self.feat_eng.\
                        choose_feature_from_acquisition(acq_func_obj, 
                                                       self.X_test, 
                                                       candidate_search=False)

                X_new = X_new.reshape(1, -1)
                
                X_new = self.x_scaler.inverse_transform(X_new)

                #TODO optimisation works well for scikitlearn+gaussian+rbf --> test for local/student-t/matern
                
                #TODO: local Matern-3/2 implementation gives spurious zeros for some (x,y) pairs

                cand_file_path = 'surrogate_al_cands' + file_path_postfix + '.csv'
                np.savetxt(cand_file_path, X_new, header=''.join([f+',' for f in feats]), comments='', delimiter=',')
                print('>Performed a single optimisation iteration, the suggested candidates are in {0}'.format(cand_file_path))
                
                return X_new

            else:

                for i in range(n_iter):

                    X_new, x_new_ind_test, x_new_ind_glob = self.feat_eng.\
                        choose_feature_from_acquisition(acq_func_obj, self.X_test)
                    X_new = X_new.reshape(1, -1)

                    # x_new_inds = feats.index(X_new)  # feats is list of features, for this
                    # has to be list of samples
                    y_new = self.y_test[x_new_ind_test].reshape(1, -1)

                    self.feat_eng.train_indices = np.concatenate([self.feat_eng.train_indices,
                                                                np.array(x_new_ind_glob).reshape(-1)])
                    self.feat_eng.test_indices = np.delete(
                        self.feat_eng.test_indices, x_new_ind_test, 0)
                    self.feat_eng.n_train += 1
                    self.feat_eng.n_test -= 1

                    X_train = np.concatenate([self.X_train, X_new])
                    y_train = np.concatenate([self.y_train, y_new])
                    X_test = np.delete(self.X_test, x_new_ind_test, 0)
                    y_test = np.delete(self.y_test, x_new_ind_test, 0)

                    if save_history:
                        self.design_history.append(x_new_ind_test)

                    # TODO update the scaler - has to transform back to original values, then refit
                    #X_train = self.x_scaler.fit_transform(X_train)
                    #y_train = self.y_scaler.fit_transform(y_train)
                    #X_test = self.x_scaler.transform(X_test)
                    #y_test = self.y_scaler.transform(y_test)

                    self.X_train = X_train
                    self.y_train = y_train
                    self.X_test = X_test
                    self.y_test = y_test

                    # self.model = es.methods.GP(X_train, y_train,
                    # kernel=self.base_kernel, bias=False, noize=self.noize,
                    # backend=self.backend)

                    self.model.train(X_train, y_train)

        elif self.backend == 'mogp':
            pass
        elif self.backend == 'local':
            pass
        else:
            raise NotImplementedError('Currently supporting only scikit-learn, mogp, and custom backend')

    def derivative_x(self, X):
        """
        Make a prediction of the derivative of output y by input x

        Args:
            X: list of feature arrays

        Returns:
        -------
        dy/dx
        """
        # TODO ideally should be a jacobian (dy_i/dx_j)_ij
        if self.backend == 'mogp':

            x = np.array([x for x in X]).T
            x = self.x_scaler.transform(x)
            x = [np.array(i) for i in x.T.tolist()]

            _, _, der = self.feat_eng._predict(x, feed_forward=lambda t: self.model.predict(t))
        else:
            raise NotImplementedError(
                "Gaussian Process derivatives w.r.t. inputs are implemented only for MOGP")

        self.y_scaler.with_mean = False
        der = self.y_scaler.inverse_transform(der)
        self.y_scaler.with_mean = True

        self.x_scaler.with_mean = False
        der = self.x_scaler.transform(der)
        self.x_scaler.with_mean = True

        return der

    def maxunc_acquisition_function(self, sample, candidates=None):
        """
        Returns the uncertainty of the model as (a posterior variance on Y) for a given sample
        Args:
            sample: a single sample from a feature array
            candidates: list of input parameter files to choose optimum from
        Returns:
            the value of uncertainty (variance) of the model
        """

        if sample.ndim == 1:
            sample = sample[None, :]

        _, uncertatinty, _ = self.model.predict(sample)

        poi = -1. * uncertatinty

        #print('acq-n f-n value of type {0} = {1}'.format(type(poi), poi)) ###DEBUG

        return poi

    def poi_acquisition_function(self, sample, candidates=None):
        """
        Returns the probability of improvement for a given sample
        Args:
            sample: a single sample from a feature array
            candidates: list of input parameter files to choose optimum from
        Returns:
            the probability of improvement if a given sample will be added to the model
        """

        jitter = 1e-9
        f_star = self.output_mean

        if sample.ndim == 1:
            sample = sample[None, :]

        mu, std, d = self.model.predict(sample)
        poi = np.linalg.norm(np.divide(abs(mu - f_star), std + jitter), ord=2)

        return poi

    def poi_function_acquisition_function(self, sample, func=(lambda x: x), target=None, candidates=None):
        """
        Returns the probability of a given sample to maximize given function
        Args:
            sample: a single sample from a feature array
            func: a function object to be optimise
            target: an external value e.g. to find samples closer to a certain outcome value (in scaled space)
            candidates: list of input parameter files to choose optimum from
        Returns:
            the probability of improvement if a given sample will be added to the model
        """

        jitter = 1e-4 # depends on the scaling, for whitened data 1e-9 to 1e-1

        if sample.ndim == 1:
            sample = sample[None, :]

        mu, std, d = self.model.predict(sample)
        #print('mean predicted for input {1} during optimisation substeps: {0}, std={2}'.format(mu, sample, std)) ###DEBUG

        #func_val = func(mu, target)        
        func_val = func(mu)
        
        #TODO: !Double check with scripts at MFW repo and in papers! 
        #  -> the former include additive constant that doesn't influence the an optimisation step
        
        #poi = np.linalg.norm(np.divide(np.pow(mu - f_star, 2), std + jitter), ord=2)
        
        poi = np.divide(-func_val + jitter, std + jitter)[0] # Workaround for dimensionality
        #poi = -func_val[0] #ATTENTION: ###DEBUG checking suspiciously low QoI value from surrogate for optimisation result

        #print('acq-n f-n value of type {0} = {1}'.format(type(poi), poi)) ###DEBUG

        return -poi # ATTENTION: this is what being minimized
