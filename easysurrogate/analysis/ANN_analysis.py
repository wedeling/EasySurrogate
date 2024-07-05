"""
CLASS TO PERFORM ANALYSIS ON RESULTS FROM AN ARTIFICIAL NEURAL NETWORK.
"""
import numpy as np
import pandas as pd
from scipy.stats import mode

from itertools import combinations

from .base import BaseAnalysis

from matplotlib import pyplot as plt

class ANN_analysis(BaseAnalysis):
    """
    ANN analysis class
    """

    def __init__(self, ann_surrogate):
        print('Creating ANN_analysis object')
        self.ann_surrogate = ann_surrogate

    def sensitivity_measures(self, feats, norm=True):
        """
        Compute global derivative-based sensitivity measures using the
        derivative of squared L2 norm of the output, computing usoing back propagation.
        Integration of the derivatives over the input space is done via MC on the provided
        input features in feats.

        Parameters
        ----------
        feats : array
            An array of input parameter values.

        norm : Boolean, optional, default is True.
            Compute the gradient of ||y||^2_2. If False it computes the gradient of
            y, if y is a scalar. If False and y is a vector, the resulting gradient is the
            column sum of the full Jacobian matrix.

        Returns
        -------
        idx : array
            Indices corresponding to input variables, ordered from most to least
            influential.

        """

        # standardize the features
        feats = (feats - self.ann_surrogate.feat_mean) / self.ann_surrogate.feat_std
        N = feats.shape[0]
        # Set the batch size to 1
        self.ann_surrogate.neural_net.set_batch_size(1)
        # initialize the derivatives
        self.ann_surrogate.neural_net.d_norm_y_dX(feats[0].reshape([1, -1]),
                                                  norm=norm)
        # compute the squared gradient
        norm_y_grad_x2 = self.ann_surrogate.neural_net.layers[0].delta_hy**2
        # compute the mean gradient
        mean = norm_y_grad_x2 / N
        # loop over all samples
        for i in range(1, N):
            # compute the next (squared) gradient
            self.ann_surrogate.neural_net.d_norm_y_dX(feats[i].reshape([1, -1]))
            norm_y_grad_x2 = self.ann_surrogate.neural_net.layers[0].delta_hy**2
            mean += norm_y_grad_x2 / N
        # order parameters from most to least influential based on the mean
        # squared gradient
        idx = np.fliplr(np.argsort(np.abs(mean).T))
        print('Parameters ordered from most to least important:')
        print(idx)
        return idx, mean

    def get_errors(self, feats, data, relative=True, return_predictions=False):
        """
        Get the training and test error of the ANN surrogate to screen. This method
        uses the ANN_Surrogate.get_dimensions() dictionary to determine where the split
        between training and test data is:

            [0,1,...,n_train,n_train+1,...,n_samples]

        Hence the last entries are used as test data, and feats and data should structured as
        such.

        Parameters
        ----------
        feats : array, size = [n_samples, n_feats]
            The features.
        data : array, size = [n_samples, n_out]
            The data.
        relative : boolean, default is True
            Compute relative instead of absolute errors.
        return_predictions : boolean, default is False
            Also return the train and test predictions.

        Returns
        -------
        err_train, err_test : float
            The training and test errors

        """
        dims = self.ann_surrogate.get_dimensions()
        # run the trained model forward at training locations
        n_mc = dims['n_train']
        train_pred = np.zeros([n_mc, dims['n_out']])
        for i in range(n_mc):
            train_pred[i, :] = self.ann_surrogate.predict(feats[i])

        train_data = data[0:dims['n_train']]
        if relative:
            err_train = np.mean(np.linalg.norm(train_data - train_pred, axis=0) /
                                np.linalg.norm(train_data, axis=0), axis=0)
            print("Relative training error = %.4f %%" % (err_train * 100))
        else:
            err_train = np.mean(np.linalg.norm(train_data - train_pred, axis=0), axis=0)
            print("Training error = %.4f " % (err_train))

        # run the trained model forward at test locations
        test_pred = np.zeros([dims['n_test'], dims['n_out']])
        for idx, i in enumerate(range(dims['n_train'], dims['n_samples'])):
            test_pred[idx] = self.ann_surrogate.predict(feats[i])
        test_data = data[dims['n_train']:]
        if relative:
            err_test = np.mean(np.linalg.norm(test_data - test_pred, axis=0) /
                               np.linalg.norm(test_data, axis=0), axis=0)
            print("Relative test error = %.4f %%" % (err_test * 100))
        else:
            err_test = np.mean(np.linalg.norm(test_data - test_pred, axis=0), axis=0)
            print("Test error = %.4f " % (err_test))

        if not return_predictions:
            return err_train, err_test
        else:
            return err_train, err_test, train_pred, test_pred

    def plot_scan(self, X_train, input_number=0, output_number=0, file_name_suf='0', **kwargs):
        
        """
        Saves a .pdf 1D plot of a QoI value predicted by a GP surrogate for a single varied input component
        """
        #TODO ideally, this can be inherited from a supercalss, but it is not clear how to do it :)

        xlabels = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho']
        ylabels = ['te_transp_flux', 'ti_transp_flux']

        lookup_names_short = {
            "ti_value": "$T_{{i}}$",
            "ti.value": "$T_{{i}}$",
            "te_value": "$T_{{e}}$",
            "te.value": "$T_{{e}}$",
            "ti_ddrho": "$\\nabla T_{{i}}$",
            "ti.ddrho": "$\\nabla T_{{i}}$",
            "te_ddrho": "$\\nabla T_{{e}}$",
            "te.ddrho": "$\\nabla T_{{e}}$",
            "te_transp_flux": "$Q_{{e}}$",
            "te_transp.flux": "$Q_{{e}}$",
            "ti_transp_flux": "$Q_{{i}}$",
            "ti_transp.flux": "$Q_{{i}}$",
            "rho": "$\\rho_{{tor}}^{{norm}}$",
            "profiles_1d_q": "$q$",
            "profiles_1d_gm3" : "$gm^{{3}}$",
            }
     
        lookup_units = {
            "ti_value": "$eV$",
            "ti.value": "$eV$",
            "te_value": "$eV$",
            "te.value": "$eV$",
            "ti_ddrho": "$eV/m$",
            "te_ddrho": "$eV/m$",
            "te.ddrho": "$eV/m$",
            "ti.drho": "$eV/m$",
            "te_transp_flux": "$W/m^{{2}}$",
            "ti_transp_flux": "$W/m^{{2}}$",
            "te_transp.flux": "$W/m^{{2}}$",
            "ti_transp.flux": "$W/m^{{2}}$",
            "rho": "",
            "profiles_1d_q" : "",
            "profiles_1d_gm3" : "",
            }

        nft = kwargs['nft'] if 'nft' in kwargs else 0

        n_in_comps = len(xlabels)

        alpha_t = 0.7

        extend_factor = 0.2
        n_points_new = 1024

        if 'fig' not in kwargs and 'ax' not in kwargs:
            fig,ax = plt.subplots(figsize=[7, 7])
        else:
            fig = kwargs['fig']
            ax  = kwargs['ax']
            col = kwargs['col']

        # Take the input component according to input_name
        # Select a range of values for this component
        # Select values for fixed components
        # Make an fine resolved array of values for this component
        # Make a new array with all components
        # Predict the QoI for this array
        # Plot the QoI vs the input component
        
        i_num = input_number

        x_values = X_train[:, i_num] # check the order of axis
        #print(x_values) ###DEBUG

        x_values_new = np.linspace(x_values.min() - extend_factor * abs(x_values.min()), 
                                   x_values.max() + extend_factor * abs(x_values.max()), n_points_new)
        
        data_remainder = {}
        # Choose the place of the cut
        cut_option = kwargs['cut_option'] if 'cut_option' in kwargs else 'median'
        x_remainder = np.delete(X_train, i_num, axis=1)

        # Option 1: Mean of every other dimension / center of existing sample
        if cut_option == 'mean':
            x_remainder_value = x_remainder.mean(axis=0) # for a data that is not on a full tensor product and/or with each vector not having odd number of components, mean would not coincide with exsisting points
        # Option 2: Mode of values among existing sample closest to the median for every other dimension
        elif cut_option == 'median':
            x_remainder_value = np.median(x_remainder, axis=0) # Should work for partial data on a fully tensor product grid
        # Option 3: read from a file
        elif cut_option == 'file':
            if 'remainder_values' in kwargs:
                file_remainder_values = kwargs['remainder_values']
                df_remainder_values = pd.read_csv(file_remainder_values, header=[0, 1], index_col=0,) # tupleize_cols=True)
                x_remainder_value = df_remainder_values[(f"ft{nft}", xlabels[i_num])]
                x_remainder_value = np.array(x_remainder_value)
        # Option 4: if on a full tensor product grid, use the mode (center) of the values for every other dimension
        elif cut_option == 'center': # should work for 100% data on a fully tensor product grid
            X_train_unique_vals = []
            indices = []
            X_train_mid_vals = np.zeros((X_train.shape[1]))
            for i in range(X_train.shape[1]):
                X_train_unique_vals.append(np.unique(X_train[:,i]))
                indices.append(int(X_train_unique_vals[-1].shape[0]/2))
                X_train_mid_vals[i] = X_train_unique_vals[-1][indices[-1]]
            x_remainder_value = np.delete(X_train_mid_vals, i_num)
            mid_indices = [ 
                            np.isclose(X_train[:,1], X_train_mid_vals[1]) & np.isclose(X_train[:,2], X_train_mid_vals[2]) & np.isclose(X_train[:,3], X_train_mid_vals[3]),
                            np.isclose(X_train[:,0], X_train_mid_vals[0]) & np.isclose(X_train[:,2], X_train_mid_vals[2]) & np.isclose(X_train[:,3], X_train_mid_vals[3]),
                            np.isclose(X_train[:,0], X_train_mid_vals[0]) & np.isclose(X_train[:,1], X_train_mid_vals[1]) & np.isclose(X_train[:,3], X_train_mid_vals[3]),
                            np.isclose(X_train[:,0], X_train_mid_vals[0]) & np.isclose(X_train[:,1], X_train_mid_vals[1]) & np.isclose(X_train[:,2], X_train_mid_vals[2]) ]
            
            #mid_indices = [all([np.isclose(X_train[:,z], X_train_mid_vals[z]) for z in y]) for y in combinations([x for x in range(n_in_comps)], n_in_comps-1)].reverse()
            mid_indices_loc = mid_indices[i_num]

        # Fall back option - error
        else:
            raise ValueError(f"Unknown cut_option: {cut_option}")
        
        #print(f"for {xlabels[input_number]} remainder values are: {x_remainder_value}") ###DEBUG
        data_remainder[(f"ft{nft}", xlabels[i_num])] = x_remainder_value

        # Training points to be displayed
        if 'y_train' in kwargs:
            y_train = kwargs['y_train']
            y_train_plot = y_train[mid_indices_loc, output_number]
            X_train_plot = x_values[mid_indices_loc]

        X_new = np.zeros((x_values_new.shape[0], X_train.shape[1]))
        X_new[:, i_num] = x_values_new
        for j in range(X_new.shape[0]):
            X_new[j, np.arange(X_train.shape[1]) != i_num] = x_remainder_value
        
        #print(f"X_new = {X_new}") ###DEBUG
        #print(f"y_new = {self.ann_surrogate.predict(X_new[0,:])}") ###DEBUG

        y = np.array([self.ann_surrogate.predict(
            X_new[i, :])[output_number] for i in range(X_new.shape[0])])

        if 'scale_function' in kwargs:
            scale_function = kwargs['scale_function']
            y_avg = scale_function(y_avg)
            y_train_plot = scale_function(y_train_plot)

        ax.plot(x_values_new, 
                y, 
                color=col,
                alpha=alpha_t,
                #label=f"{input_number}->{output_number}",
                label=f"{lookup_names_short[ylabels[output_number]]}, prediction mean",
                )
           
        # Plot training points
        if 'y_train' in kwargs:
            ax.scatter(X_train_plot,
                       y_train_plot,
                       marker='o',
                       s=9,
                       color=col,
                       alpha=alpha_t, 
                       label=f"{lookup_names_short[ylabels[output_number]]}, training sample"
                       )

        ax.grid(which='major', linestyle='-')
        ax.legend(loc='best')

        #ax.set_xlabel(xlabels[input_number])
        #ax.set_ylabel(ylabels[output_number])

        ax.set_xlabel(f"{lookup_names_short[xlabels[input_number]]}, {lookup_units[xlabels[output_number]]}")
        ax.set_ylabel(f"$Q_{{e,i}}$, $W/m^{{2}}$")

        #ax.set_title(f"{xlabels[input_number]}->{ylabels[output_number]}(@ft#{nft})")
        fig.savefig('scan_'+'i'+str(input_number)+'o'+str(output_number)+'f'+str(nft)+'.pdf')

        # Store and save the remainder input values i.e. the coordinates of the cut
        data_remainder = pd.DataFrame(data_remainder)
        data_remainder.to_csv(f"scan_gem0ann_remainder_{xlabels[input_number]}_{file_name_suf}_ft{nft}.csv")

        data = pd.DataFrame({'x': x_values_new, 'y': y})
        
        return data

    def plot_loss(self, name_sufix='', **kwargs):

        if "fig" not in kwargs:
            fig,ax = plt.subplots(figsize=[7, 7])
        else:
            fig = kwargs["fig"]
            ax = kwargs["ax"]

        y = self.ann_surrogate.neural_net.loss_vals
        x = np.arange(len(y))

        ax.semilogy(x, 
                    y, 
                    alpha=0.5,
                    label=f"f.t. {name_sufix}")

        ax.set_xlabel('optimisation step')
        ax.set_ylabel(f"{self.ann_surrogate.loss} loss of the model")        

        fig.savefig(f"loss_{name_sufix}.pdf")

        return 0
    