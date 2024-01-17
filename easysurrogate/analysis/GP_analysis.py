"""
CLASS TO PERFORM ANALYSIS ON RESULTS FROM A GAUSSIAN PROCESS SURROGATE
"""

from shutil import which
from .base import BaseAnalysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error as mse
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches

#latexplotlib
import latexplotlib as lpl
plt.style.use("default")
plt.rcParams.update({
    "font.family": "serif", 
    "text.usetex": False,
            })

class GP_analysis(BaseAnalysis):
    """
    GP analysis class
    """

    def __init__(self, gp_surrogate, **kwargs):

        print('Creating GP_analysis object')

        self.gp_surrogate = gp_surrogate

        if 'target_name_selected' in kwargs:
            self.target_name_selected = kwargs['target_name_selected']
        
        if 'features_names_selected' in kwargs:
            self.features_names_selected = kwargs['features_names_selected']

        if 'nft' in kwargs:
            self.nft = kwargs['nft']
        
    def plot_err(self, error, name, original=None, addit_name=''):
        plt.ioff()
        plt.title(name)
        plt.plot(range(len(error)), error, '.', label='GP metamodel error', color='b')
        if original is not None:
            plt.plot(original, '.', label='Simulation model', color='green')
        plt.xlabel('Run number')
        plt.ylabel('Results of {}'.format(name))
        plt.legend()
        plt.grid('both')
        #plt.yscale("symlog") #if comment then TEMP
        plt.savefig(f"gp_abs_err_{addit_name}.pdf")
        plt.close()

    def plot_predictions_vs_groundtruth(self,
                                        y_test_orig, 
                                        y_test_pred, 
                                        y_test_pred_var=None,
                                        y_test_orig_var=None,
                                        name=f"Predictions against ground truth values",
                                        addit_label='',
                                        addit_name='',
                                       ):
        """
        Plots original values of QoI on X axis and predicted values on Y axis
        """

        #plt.ioff()

        plt.style.use("latex10pt")
        plt.rcParams.update({"axes.grid": True, "font.family": "serif", "text.usetex": False})
        #lpl_context = [550, 350]
        lpl_context = [347, 549] # LNCS paper size
        lpl_context = [347, int(549*1.25)] # LNCS paper size - WIDENED
        lpl_context = [600, 1000]
        lpl.size.set(*lpl_context)

        #fig = plt.figure()
        #fig, ax = plt.subplots()
        fig, ax = lpl.subplots(1, 1, scale=0.5) #,aspect='equal',

        val_err = 0.15
        beta_expplot = 0.05

        #ax.set_title(name)

        if y_test_pred_var is not None and y_test_orig_var is not None:

            ax.errorbar(
                x=y_test_orig,
                y=y_test_pred,
                yerr=1.96 * y_test_pred_var,
                xerr=1.96 * y_test_orig_var,
                label=f"variance of GPR model on test and train data",
                fmt='b.')

        elif y_test_pred_var is not None and y_test_orig_var is None:

            ax.errorbar(
                x=y_test_orig,
                y=y_test_pred,
                yerr=1.96 * y_test_pred_var,
                label=f"mean and 95% CI of GPR \nmodel on test data; {addit_label}",
                fmt='b.')

        elif y_test_pred_var is None and y_test_orig_var is not None:

            ax.errorbar(
                x=y_test_orig,
                y=y_test_pred,
                xerr=1.96 * y_test_orig_var,
                label=f"variance of GPR model on train data",
                fmt='b.')

        else:

            ax.plot(y_test_orig, y_test_pred, label=f"pred-s vs g.t.", fmt='.')

        # Adding a mean of X=Y
        ax.plot(y_test_orig, y_test_orig, 'k-')

        # Adding error bands around original mean
        ax.plot(y_test_orig, (1.+val_err)*y_test_orig, 'k-', alpha=0.25, label=f"+/-{val_err*100}% error")
        ax.plot(y_test_orig, (1.-val_err)*y_test_orig, 'k-', alpha=0.25,) # label="-{0}%".format(val_err*100))

        # Limitting plot to make it square
        maxlim = max((1.+val_err)*(1.+beta_expplot)*y_test_orig)
        minlim = min((1.-val_err)*y_test_orig)
        maxlim_less = max(y_test_orig)*(1.+beta_expplot)
        minlim_less = min(y_test_orig)
        ax.set_xlim(minlim_less, maxlim_less)
        ax.set_ylim(minlim, maxlim)

        #Adding a highlight square for the results
        #TODO: undo hardcode - get values from a result of a workflow run, for example
        ref_low = 1.35E+6
        ref_high = 3.9E+6
        ref_high = maxlim_less = max(y_test_orig)*(1.+beta_expplot/2)
        ref_delta = ref_high - ref_low

        box_color = 'green'
        box_alpha = 0.9
        box_style = '-'
        ax.hlines(y=ref_low, xmin=ref_low, xmax=ref_high, color=box_color, linestyle=box_style, alpha=box_alpha, label=f"region of interest")
        ax.hlines(y=ref_high, xmin=ref_low, xmax=ref_high, color=box_color, linestyle=box_style, alpha=box_alpha)
        ax.vlines(x=ref_low, ymin=ref_low, ymax=ref_high, color=box_color, linestyle=box_style, alpha=box_alpha)
        ax.vlines(x=ref_high, ymin=ref_low, ymax=ref_high, color=box_color, linestyle=box_style, alpha=box_alpha)
        # TODO: consider shading out outer part with hatches

        #rect = patches.Rectangle((ref_low, ref_high), ref_delta, ref_delta, linewidth=1, edgecolor='gray', facecolor='none', alpha=0.25)
        #ax.add_patch(rect)

        # Scaling of the plots
        #ax.yscale('symlog') #if comment then TEMP
        #ax.xscale('symlog') #if comment then TEMP

        ax.ticklabel_format(useMathText=True)

        ax.legend(fancybox=True, framealpha=0.5, loc='best')
        #ax.grid(True, which='both', axis='both')
        ax.set_xlabel(f"Original values")
        ax.set_ylabel(f"Predicted values")
        #fig.tight_layout()
        fig.savefig(f"pred_vs_orig_{addit_name}.pdf")
        plt.close()

    def plot_res(self,
                 x_train, y_train_pred, y_train_orig,
                 x_test=None, y_test_pred=None, y_test_orig=None,
                 y_std_pred_test=None, y_std_pred_train=None,
                 name='', num='1', type_train='rand', train_n=10, out_color='b', 
                 output_folder='', addit_name=''):

        plt.ioff()
        plt.figure(figsize=(12, 15))

        n_train = y_train_pred.shape[0]
        n_test = y_test_pred.shape[0] if y_test_pred is not None else 0
        
        #x_test = [] if x_test is None else x_test

        y_pred = np.zeros(n_test + n_train)
        y_pred[x_train] = y_train_pred
        if x_test is not None:
            y_pred[x_test] = y_test_pred

        #y_orig = np.zeros(y_test_orig.shape[0] + y_train_orig.shape[0])
        y_orig = np.zeros(n_test + n_train)
        y_orig[x_train] = y_train_orig
        if y_test_orig is not None:
            y_orig[x_test] = y_test_orig

        # --- 1) Plotting prediction vs original test data
        plt.subplot(311)

        plt.title('{} training sample of size: {}'.format(type_train, str(train_n)))
        plt.xlabel('Run number')
        plt.ylabel('Prediction of {}'.format(name))

        #plt.yscale("symlog") #if comment then TEMP

        plt.plot(x_train, y_train_orig, '*', label='Simulation, train', color='green')
        if y_test_orig is not None:
            plt.plot(x_test, y_test_orig, '.', label='Simulation, test', color='red')

        if y_std_pred_train is not None:
            
            if y_std_pred_test is not None:
                y_std_pred = np.concatenate([y_std_pred_test, y_std_pred_train])
            else:
                y_std_pred = y_std_pred_train

            plot_ticks = np.logspace(np.log10((y_pred-2*y_std_pred).min()), np.log10((y_pred+2*y_std_pred).max()), num=8)
            
            #print("ticks with errors are : {0}".format(plot_ticks)) ###DEBUG

            if (y_pred-2*y_std_pred).min() <= 0:
                plot_ticks = np.hstack([
                    -np.logspace(1., np.log10(-(y_pred-2*y_std_pred).min()), num=4)[::-1], 
                     np.logspace(1., np.log10( (y_pred+2*y_std_pred).max()), num=4)
                ])

            #print("ticks with errors now are : {0}".format(plot_ticks)) ###DEBUG
                
            plt.errorbar(
                x=x_train,
                y=y_train_pred,
                yerr=1.96 *
                y_std_pred_train,
                label='GP metamodel, train',
                fmt='+')

            if y_test_pred is not None:
                plt.errorbar(
                    x=x_test,
                    y=y_test_pred,
                    yerr=1.96 *
                    y_std_pred_test,
                    label='GP metamodel, test',
                    fmt='+')

        else:
            
            plt.plot(x_train, y_train_pred, '*', label='GP metamodel', color=out_color)
            if y_test_pred is not None:
                plt.plot(x_test, y_test_pred, '.', label='GP metamodel', color=out_color)

            plot_ticks = np.logspace(y_pred.min(), y_pred.max(), num=8)   

        plt.legend()
        plt.grid(True, which='both', axis='both')
        plt.yticks(ticks=plot_ticks)
        #plt.tight_layout()

        # --- 2) Plotting absolute errors
        plt.subplot(312)

        #plt.yscale("symlog") #if comment then TEMP
        plt.xlabel('Run number')
        plt.ylabel('Error')

        err_abs = y_pred - y_orig

        plt.plot(err_abs, '.', color=out_color, label='y_pred - y_orig')
        plt.grid()
        plt.legend()

        # print('Indices of test data where absolute error is larger than {} : {} '
        #       .format(2e5, np.where(abs(err_abs) > 2e5)[0]))

        # --- 3) Plotting relative errors
        plt.subplot(313)
        plt.plot(np.fabs((y_pred - y_orig) / y_orig) * 100, '.', color=out_color, label='|(y_p - y_o) / y_o|*100%')
        plt.grid()
        plt.xlabel('Run number')
        plt.ylabel('Relative error (%)')
        #plt.yscale("log") #if comment then TEMP
        plt.legend()
        #plt.tight_layout()

        # Finishing
        plt.savefig(
            output_folder +
            'GP_prediction_' +
            num +
            '_' +
            type_train +
            '_' +
            str(train_n) +
            '_' +
            addit_name +
            '.pdf',
            bbox_inches='tight',
            dpi=100)
        plt.clf()
        plt.close()

    def plot_scan(self, X_train, input_number=0, output_number=0, file_name_suf='0', **kwargs):
        """
        Saves a .pdf 1D plot of a QoI value predicted by a GP surrogate for a single varied input component
        """

        xlabels = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho']
        ylabels = ['te_transp_flux', 'ti_transp_flux']

        nft = kwargs['nft'] if 'nft' in kwargs else 0

        extend_factor = 0.5
        fig,ax = plt.subplots(figsize=[7, 7])

        # Take the input component according to input_name
        # Select a range of values for this component
        # Make an fine resolved array of values for this component
        # Make a new array with all components
        # Predict the QoI for this array
        # Plot the QoI vs the input component
        
        name_dict = {} #TODO?
        #i_num = name_dict[input_name]
        i_num = input_number

        x_values = X_train[:, i_num] # check the order of axis
        #print(x_values) ###DEBUG

        x_values_new = np.linspace(x_values.min() - extend_factor * abs(x_values.min()) , 
                                   x_values.max() + extend_factor * abs(x_values.max()), 1000)
        
        data_remainder = {}
        # Choose place to cut
        cut_option = kwargs['cut_option'] if 'cut_option' in kwargs else 'median'
        x_remainder = np.delete(X_train, i_num, axis=1)

        # Option 1: Mean of every other dimension / center of existing sample
        if cut_option == 'mean':
            x_remainder_value = x_remainder.mean(axis=0)
        # Option 2: Mode of values among existing sample closest to the median for every other dimension
        elif cut_option == 'median':
            x_remainder_value = np.median(x_remainder, axis=0) # Should work for partial data on a fully tensor product grid
        # Option 3: read from a file
        elif cut_option == 'file':
            if 'remainder_values' in kwargs:
                file_remainder_values = kwargs['remainder_values']
                df_remainder_values = pd.read_csv(f"{file_remainder_values}", header=[0, 1], index_col=0,) # tupleize_cols=True)
                x_remainder_value = df_remainder_values[(f"ft{nft}", xlabels[i_num])]
                x_remainder_value = np.array(x_remainder_value)
        # Option 4: if data is on a tensor product grid, then use the center of the grid
        elif cut_option == 'center': # should work for 100% data on a fully tensor product grid
            X_train_unique_vals = []
            indices = []
            X_train_mid_vals = np.zeros((X_train.shape[1]))
            for i in range(X_train.shape[1]):
                #print(f"X_train col={X_train[:,i]}") ###DEBUG
                X_train_unique_vals.append(np.unique(X_train[:,i]))
                #print(f"X_train_unique_vals last={X_train_unique_vals[-1]}") ###DEBUG
                indices.append(int(X_train_unique_vals[-1].shape[0]/2))
                #print(f"indices last={indices[-1]}")
                X_train_mid_vals[i] = X_train_unique_vals[-1][indices[-1]]
            #print(f"X_train_unique_vals={X_train_unique_vals}") ###DEBUG
            #print(f"X_train_mid_vals={X_train_mid_vals}") ###DEBUG
            x_remainder_value = np.delete(X_train_mid_vals, i_num)
            mid_indices = [ np.isclose(X_train[:,1], X_train_mid_vals[1]) & np.isclose(X_train[:,2], X_train_mid_vals[2]) & np.isclose(X_train[:,3], X_train_mid_vals[3]),
                            np.isclose(X_train[:,0], X_train_mid_vals[0]) & np.isclose(X_train[:,2], X_train_mid_vals[2]) & np.isclose(X_train[:,3], X_train_mid_vals[3]),
                            np.isclose(X_train[:,0], X_train_mid_vals[0]) & np.isclose(X_train[:,1], X_train_mid_vals[1]) & np.isclose(X_train[:,3], X_train_mid_vals[3]),
                            np.isclose(X_train[:,0], X_train_mid_vals[0]) & np.isclose(X_train[:,1], X_train_mid_vals[1]) & np.isclose(X_train[:,2], X_train_mid_vals[2]) ]
            mid_indices_loc = mid_indices[i_num]

        # Fall back option - error
        else:
            raise ValueError(f"Unknown cut_option: {cut_option}")
        
        # Write (and display) remiander values of the cut location
        print(f"for {xlabels[i_num]} @ft#{nft} remainder values are: {x_remainder_value}") ###DEBUG
        data_remainder[(f"ft{nft}", xlabels[i_num])] = x_remainder_value
        
        # Training points to be displayed
        if 'y_train' in kwargs:
            #print(f"mid_indices_loc={mid_indices_loc}") ###DEBUG
            y_train = kwargs['y_train']
            y_train_plot = y_train[mid_indices_loc]
            X_train_plot = x_values[mid_indices_loc]
        
        X_new = np.zeros((x_values_new.shape[0], X_train.shape[1]))
        X_new[:, i_num] = x_values_new
        for j in range(X_new.shape[0]):
            X_new[j, np.arange(X_train.shape[1]) != i_num] = x_remainder_value
        
        #print(f"y_pred={self.gp_surrogate.predict(X_new[0,:].reshape(-1,1))}") ###DEBUG
        #print(f"{self.gp_surrogate.predict(X_new[0,:].reshape(-1,1))[1][0][0]}") ###DEBUG
        #print(f"{self.gp_surrogate.predict(X_new[0,:].reshape(-1,1))[0][0][0]}") ###DEBUG

        # resulting y is [(array(n_outputs), array(n_outputs))] : element of list dimensionality is (2 x 1 x n_outputs)
        y_avg = np.array([self.gp_surrogate.predict(
            X_new[i, :].reshape(-1, 1))[0][0][output_number] for i in range(X_new.shape[0])])
        
        y_std = np.array([self.gp_surrogate.predict(
            X_new[i, :].reshape(-1, 1))[1][0][output_number] for i in range(X_new.shape[0])])
        
        #print(y) ###DEBUG

        #Sax.plot(x_values_new, y, label=f"{input_number}->{output_number}")
        ax.errorbar(x_values_new, 
                    y_avg, 
                    yerr=1.96 * y_std, 
                    label=f"{input_number}->{output_number}",
                    alpha=0.2,
                    )
        
        # Plot training points
        if 'y_train' in kwargs:
            #print(f"Plotting training points for {input_number}->{output_number}") ###DEBUG
            ax.plot(X_train_plot, y_train_plot, 'ko', label='training points')

        ax.set_xlabel(xlabels[input_number])
        ax.set_ylabel(ylabels[output_number])
        ax.set_title(f"{xlabels[input_number]}->{ylabels[output_number]}(@ft#{file_name_suf})")
        fig.savefig('scan_'+'i'+str(input_number)+'o'+str(output_number)+'f'+file_name_suf+'.pdf')

        # Store and save the remainder input values i.e. the coordinates of the cut
        data_remainder = pd.DataFrame(data_remainder)
        data_remainder.to_csv(f"scan_gem0gpr_remainder_{xlabels[input_number]}_ft{file_name_suf}.csv")

        data = pd.DataFrame({'x': x_values_new, 'y': y_avg})

        return data
    
    def plot_pdfs(self, y_dom, y_pdf, y_dom_sur, y_pdf_sur,
                  y_dom_tr=None, y_pdf_tr=None, y_dom_tot=None, y_pdf_tot=None,
                  names=['GEM testing', 'GP', 'GEM training', 'GEM', ],
                  qoi_names=['TiFl'], filename='pdfs'):

        fig, ax = plt.subplots(figsize=[7, 7])

        ax.plot(y_dom, y_pdf, 'r-', label=names[0])
        ax.plot(y_dom_sur, y_pdf_sur, 'b', label=names[1])

        if y_dom_tr is not None and y_pdf_tr is not None:
            ax.plot(y_dom_tr, y_pdf_tr, 'g-', label=names[2])

        if y_dom_tot is not None and y_pdf_tot is not None:
            ax.plot(y_dom_tot, y_pdf_tot, 'k-', label=names[3])

        ax.set_xlabel(qoi_names[0])
        ax.set_ylabel('pdf')

        ax.xaxis.grid(True, which='major')
        ax.yaxis.grid(True, which='major')

        plt.legend(loc='best')
        plt.title('PDF of simulated and predicted target values')
        plt.savefig(filename + '.pdf')
        plt.close()

    def plot_2d_design_history(self, x_test=None, y_test=None):

        if not hasattr(self.gp_surrogate, 'design_history'):
            raise RuntimeWarning('This surrogate has no recorded history of sequential design')
            return

        #ytr = self.gp_surrogate.y_scaler.inverse_transform(self.gp_surrogate.y_train)
        ytr = self.gp_surrogate.y_scaler.inverse_transform(self.gp_surrogate.X_train, self.gp_surrogate.y_train) # for custom scaler
        ytr = ytr.reshape(-1)
        if y_test is not None:
            yts = y_test
            yts = yts.reshape(-1)
        else:
            yts = ytr

        x_train = self.gp_surrogate.x_scaler.inverse_transform(self.gp_surrogate.X_train)
        x1tr = x_train[:, 0]
        x2tr = x_train[:, 1]
        if x_test is not None:
            x1ts = x_test[:, 0]
            x2ts = x_test[:, 1]
        else:
            x1ts = x1tr
            x2ts = x1tr

        fig, ax = plt.subplots()

        cntr1 = ax.tricontourf(x1ts, x2ts, yts, levels=12, cmap="RdBu_r")
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="8%", pad=0.1)
        cbar1 = fig.colorbar(cntr1, cax1)
        ax.set_title('Ground simulation results', pad=10.0)
        ax.set_xlabel(r'$x_{1}$')
        ax.set_ylabel(r'$ x_{2}$')
        # cbar1.set_label(r'$y$')
        ax.set_aspect('equal')

        ax.plot(x1tr, x2tr, 'ko', ms=3)
        n_iter = len(self.gp_surrogate.design_history)
        for num, ind in enumerate(self.gp_surrogate.design_history):
            #p_i = (x1[ind], x2[ind])
            p_i = (x1tr[-n_iter + num], x1tr[-n_iter + num])
            ax.annotate(str(num), p_i, textcoords="offset points", xytext=(0, 10), ha='center')

        # plt.tight_layout()
        # plt.subplots_adjust()

        plt.show()
        plt.savefig('surorrogate_seq_des.png')
        plt.close()

    def get_r2_score(self, X, y):
        """
        Compute R^2 score of the regression for the test data
          Returns
          -------
            r2: float
            value of R^2 score
        """

        # f = gpr.predict(X)
        # y_mean = y.mean()
        # r2 = 1 - np.multiply(y - f, y - f).sum() / (np.multiply(y - y_mean, y - y_mean).sum())
        
        """
        if self.gp_surrogate.backend == 'scikit-learn':
            r2 = self.gp_surrogate.model.instance.score(X, y)
        elif self.gp_surrogate.backend == 'local:
            r2 = self.gp_surrogate.model.instance.r2_score(X, y)
        else:
            r2 = 0.
        """

        r2 = self.gp_surrogate.model.instance.score(X, y)
        return r2

    def get_regression_error(
            self,
            X_test,
            y_test,
            X_train=None,
            y_train=None,
            index=None,
            flag_plot=True,
            **kwargs):
        """
        Compute the regression RMSE error of GP surrogate.
        Prints RMSE regression error and plots the errors of the prediction for different simulation runs/samples
        # TODO add correlation analysis

        Args:
            index: array, indices of the dataset to form a test set

        Returns:
            None
        """

        if 'addit_name' in kwargs:
            addit_name = kwargs['addit_name']
        else:
            addit_name = ''

        if index is not None:
            X_test = self.gp_surrogate.model.X[index]
            y_test = self.gp_surrogate.model.y[index]

        x_test_inds = self.gp_surrogate.feat_eng.test_indices
        x_train_inds = self.gp_surrogate.feat_eng.train_indices

        only_train_set = False
        if len(x_test_inds) == 0:
            only_train_set = True 
            y_t_len = 0

        ### ---
        # For every component of input and output plot the scan of the GP model predictions for the orthogonal middle cut of training domain
        print("Scanning the GP model predictions")
        remainder_file_date = kwargs["remainder_file_date"] if "remainder_file_date" in kwargs else "20240110"
        remainder_file_path = kwargs["remainder_file_path"] if "remainder_file_path" in kwargs else "scan_gem0_remainder_"
        scan_dict = {}
        for output_num in range(y_train.shape[1]):
            for input_num in range(X_train.shape[1]):
                # Option 1: pick the scan file with the remainder values for the cut
                # scan_data = self.plot_scan(X_train, input_number=input_num, output_number=output_num, file_name_suf=addit_name,
                #                            nft=self.nft, remainder_values=f"{remainder_file_path}{self.features_names_selected[input_num]}_{remainder_file_date}.csv",
                #                            )
                # Option 2: scan for the middle values of the fulll grid
                scan_data = self.plot_scan(X_train, input_number=input_num, output_number=output_num, file_name_suf=addit_name,
                                           nft=self.nft, cut_option='center', y_train=y_train[:,output_num],
                                           )

                scan_dict[f"{self.features_names_selected[input_num]}_{self.target_name_selected[output_num]}"] = scan_data
        scan_dataframe = pd.DataFrame.from_dict({(i,j): scan_dict[i][j] for i in scan_dict.keys() for j in scan_dict[i].keys()})
        scan_dataframe.to_csv(f"scan_{self.nft}.csv")
        ### ---

        print("Prediction of new QoI")
        # TODO make predict call work on a (n_samples, n_features) np array
        # TODO: IMPORTANT - it's needed for vector outputs e.g. (Q_e, Q_i) and currently reshaping is a bloody mess

        n_output = y_train.shape[1]
        err_abs_list = []
        err_rel_list = []
        r2_test_list = []

        for n_out in range(n_output):

            addit_name_new = addit_name + f"_o{n_out}"

            y_pred = []
            y_std_pred = []
            if not only_train_set:
                y_pred = [self.gp_surrogate.predict(X_test[i, :].reshape(-1, 1))[0]
                        for i in range(X_test.shape[0])]
                y_std_pred = [self.gp_surrogate.predict(X_test[i, :].reshape(-1, 1))[1]
                            for i in range(X_test.shape[0])]
                y_t_len = len(y_test)

            y_pred_train = [self.gp_surrogate.predict(
                X_train[i, :].reshape(-1, 1))[0] for i in range(X_train.shape[0])]
            y_std_pred_train = [self.gp_surrogate.predict(
                X_train[i, :].reshape(-1, 1))[1] for i in range(X_train.shape[0])]

            # Reshape the resulting arrays
            # TODO: here squeeze should not mix the output components
            if not only_train_set:
                y_pred = np.squeeze(np.array(y_pred), axis=1)
                y_std_pred = np.squeeze(np.array(y_std_pred), axis=1)
            else:
                y_pred = np.ones((0, len(y_pred_train[n_out])))
                y_std_pred = np.ones((0, len(y_pred_train[n_out])))
                
                #print('y_pred shape {}'.format(y_pred.shape)) ###DEBUG
                
            y_pred_train = np.squeeze(np.array(y_pred_train), axis=1)
            y_std_pred_train = np.squeeze(np.array(y_std_pred_train), axis=1)

            #print(f"y_test: \n{y_test}") ###DEBUG

            # Check if we a working with a vector or scalar QoI:
            #  If it is vector then consider only the first component
            y_test_plot = y_test
            y_train_plot = y_train

            if y_pred.shape[1] != 1:

                y_train_plot = y_train[:, [n_out]]
                y_pred_train = y_pred_train[:, n_out]
                y_std_pred_train = y_std_pred_train[:, n_out]

                if not only_train_set:
                    y_test_plot = y_test[:, [n_out]]            
                    y_pred = y_pred[:, n_out]
                    y_std_pred = y_std_pred[:, n_out]

            if len(y_pred.shape) == 1:

                y_pred_train = y_pred_train.reshape(-1, 1)

                if not only_train_set:
                    y_pred = y_pred.reshape(-1, 1)

            # Calculate the errors
            # test data appears smaller in length (by 1)
            err_abs = np.subtract(y_pred[:y_t_len, :], y_test)
            err_rel = np.divide(err_abs, y_test)
            
            err_abs_list.append(err_abs)
            err_rel_list.append(err_rel)

            # Scale back the features and targets
            if not only_train_set:
                #y_test_scale = self.gp_surrogate.y_scaler.transform(y_test)
                y_test_scale = self.gp_surrogate.y_scaler.transform(X_test, y_test) # for custom scaler
                X_test_scale = self.gp_surrogate.x_scaler.transform(X_test)

                r2_test = self.get_r2_score(X_test_scale, y_test_scale) # commenting out *_*_scale to check if original scale data work
            else:
                r2_test = 1.0
            
            print('R2 score for the test data is : {:.3}'.format(r2_test))

            mse_test   = 0.
            mse_train  = 0.
            rmse_test  = 0.
            rmse_train = 0.
            if not only_train_set:
                mse_test  = mse(y_pred[:y_t_len, 0], y_test_plot[:, 0])
                rmse_test = mse(y_pred[:y_t_len, 0], y_test_plot[:, 0], squared=False)
                print('MSE of the GPR prediction is: MSE={:.3} RMSE={:.3}'.format(mse_test, rmse_test))
            else:
                print('MSE of the GPR prediction for training set is: {:.3}'.format(
                    mse(y_pred_train[:, 0], y_train_plot[:, 0])))

            if not only_train_set:
                print('Mean relative test error is {:.3}'.format(np.abs(err_rel).mean()))

            self.y_pred = y_pred
            self.err_abs = err_abs
            self.err_rel = err_rel
            self.r2_test = r2_test

            r2_test_list.append(r2_test)

            # Save the prediction results
            if not only_train_set:
                csv_array = np.concatenate(
                        (y_test_plot[:, 0].reshape(-1,1), 
                        y_pred[:y_t_len, 0].reshape(-1,1), 
                        y_std_pred.reshape(y_pred[:y_t_len, 0].shape).reshape(-1,1)), #TODO ugly
                        axis=1)
                np.savetxt(f"res_{addit_name_new}.csv", csv_array, delimiter=",")

            print("Printing and plotting the evaluation results")
            #TODO: resolve case whe y_groundtruth==0.0
            #TODO: add color for test case in Err_abs and Err_rel plots
    
            if flag_plot and not only_train_set:
    
                self.plot_err(error=err_rel[:, n_out],
                            name=f"rel. err. of prediction mean for test dataset in fluxes nu {n_out}",
                            #original=y_test[:, 0],
                            )
                
                self.plot_predictions_vs_groundtruth(
                    y_test_plot[:, 0], 
                    y_pred[:y_t_len, 0], 
                    y_std_pred.reshape(y_pred[:y_t_len, 0].shape),
                    #y_std_pred_train.reshape(y_pred_train[:,0].shape),
                    addit_label=f"$R^{{2}}={{{r2_test:.3}}}$",
                    addit_name=addit_name_new,
                                                    )

            train_n = self.gp_surrogate.feat_eng.n_samples - y_t_len

            #print(x_train_inds, y_pred_train, y_train_plot) ###DEBUG

            #TODO: y_test_plot now have 2 columns, so simple reshaping would not work

            if flag_plot:
                
                if not only_train_set:

                    self.plot_res(
                            x_train_inds, y_pred_train[:, 0], y_train_plot[:, 0],
                            x_test_inds, y_pred[:y_t_len, 0], y_test_plot[:, 0],
                            y_std_pred.reshape(y_pred[:y_t_len, 0].shape), y_std_pred_train.reshape(y_pred_train[:, 0].shape),
                            name=r'$Y_i$', num=str(n_out), type_train='rand',
                            train_n=train_n, out_color='b',
                            addit_name=addit_name_new,
                                )
                
                else:
                    
                    self.plot_res(
                            x_train_inds, 
                            y_pred_train[:, 0],
                            y_train_plot[:, 0],
                            y_std_pred_train=y_std_pred_train.reshape(y_pred_train[:, 0].shape),
                            name=r'$Y_i$', num=str(n_out), type_train='rand',
                            train_n=train_n, out_color='b',
                            addit_name=addit_name_new,
                                )

        return err_abs_list, err_rel_list, r2_test_list
