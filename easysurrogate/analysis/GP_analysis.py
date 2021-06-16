"""
CLASS TO PERFORM ANALYSIS ON RESULTS FROM A GAUSSIAN PROCESS SURROGATE
"""

from .base import BaseAnalysis
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error as mse
from mpl_toolkits.axes_grid1 import make_axes_locatable


class GP_analysis(BaseAnalysis):
    """
    GP analysis class
    """

    def __init__(self, gp_surrogate, **kwargs):

        print('Creating GP_analysis object')

        self.gp_surrogate = gp_surrogate

    def plot_err(self, error, name, original=None):
        plt.ioff()
        plt.title(name)
        plt.plot(range(len(error)), error, '.', label='GP metamodel error', color='b')
        if original is not None:
            plt.plot(original, '.', label='Simulation model', color='green')
        plt.xlabel('Run number')
        plt.ylabel('Results of {}'.format(name))
        plt.legend()
        plt.grid('major')
        plt.yscale("symlog")
        plt.savefig('gp_abs_err.png')
        plt.close()

    def plot_res(self,
                 x_test, y_test_pred, y_test_orig,
                 x_train, y_train_pred, y_train_orig,
                 y_var_pred_test=None, y_var_pred_train=None,
                 name='', num='1', type_train='rand', train_n=10, out_color='b', output_folder=''):

        plt.ioff()
        plt.figure(figsize=(8, 11))

        y_pred = np.zeros(y_test_pred.shape[0] + y_train_pred.shape[0])
        y_pred[x_train] = y_train_pred
        y_pred[x_test] = y_test_pred

        y_orig = np.zeros(y_test_orig.shape[0] + y_train_orig.shape[0])
        y_orig[x_train] = y_train_orig
        y_orig[x_test] = y_test_orig

        # --- Plotting prediction vs original test data
        plt.subplot(311)
        plt.title('{} training sample of size: {}'.format(type_train, str(train_n)))

        plt.plot(x_test, y_test_orig, '.', label='Simulation, test', color='red')
        plt.plot(x_train, y_train_orig, '*', label='Simulation, train', color='green')

        if y_var_pred_test is not None and y_var_pred_train is not None:
            plt.errorbar(
                x=x_test,
                y=y_test_pred,
                yerr=1.96 *
                y_var_pred_test,
                label='GP metamodel, test',
                fmt='+')
            plt.errorbar(
                x=x_train,
                y=y_train_pred,
                yerr=1.96 *
                y_var_pred_train,
                label='GP metamodel, train',
                fmt='+')
        else:
            plt.plot(x_test, y_test_pred, '.', label='GP metamodel', color=out_color)
            plt.plot(x_train, y_train_pred, '*', label='GP metamodel', color=out_color)

        plt.xlabel('Run number')
        plt.ylabel('Prediction of {}'.format(name))
        plt.legend()
        plt.grid()
        plt.yscale("symlog")

        # --- Plotting absolute errors
        plt.subplot(312)
        err_abs = y_pred - y_orig

        plt.plot(err_abs, '.', color=out_color)

        plt.grid()
        plt.yscale("symlog")
        plt.xlabel('Run number')
        plt.ylabel('Error')

        # print('Indices of test data where absolute error is larger than {} : {} '
        #       .format(2e5, np.where(abs(err_abs) > 2e5)[0]))

        # --- Plotting relative errors
        plt.subplot(313)
        plt.plot(np.fabs(y_pred - y_orig) / y_orig * 100, '.', color=out_color)
        plt.grid()
        plt.xlabel('Run number')
        plt.ylabel('Relative error (%)')
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(
            output_folder +
            'GP_prediction_' +
            num +
            '_' +
            type_train +
            '_' +
            str(train_n) +
            '.png',
            bbox_inches='tight',
            dpi=100)
        plt.clf()
        plt.close()

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
        plt.savefig(filename + '.png')
        plt.close()

    def plot_2d_design_history(self, x_test=None, y_test=None):

        if not hasattr(self.gp_surrogate, 'design_history'):
            raise RuntimeWarning('This surrogate has no recorded history of sequential design')
            return

        x_train = self.gp_surrogate.x_scaler.inverse_transform(self.gp_surrogate.X_train)
        x1tr = x_train[:, 0]
        x2tr = x_train[:, 1]
        if x_test is not None:
            x1ts = x_test[:, 0]
            x2ts = x_test[:, 1]
        else:
            x1ts = x1tr
            x2ts = x1tr

        ytr = self.gp_surrogate.y_scaler.inverse_transform(self.gp_surrogate.y_train)
        ytr = ytr.reshape(-1)
        if y_test is not None:
            yts = y_test
            yts = yts.reshape(-1)
        else:
            yts = ytr

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
        Returns: vale of R^2 score
        """

        # f = gpr.predict(X)
        # y_mean = y.mean()
        # r2 = 1 - np.multiply(y - f, y - f).sum() / (np.multiply(y - y_mean, y - y_mean).sum())

        return self.gp_surrogate.model.instance.score(X, y)

    def get_regression_error(
            self,
            X_test,
            y_test,
            X_train=None,
            y_train=None,
            index=None,
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

        if index is not None:
            X_test = self.gp_surrogate.model.X[index]
            y_test = self.gp_surrogate.model.y[index]

        x_test_inds = self.gp_surrogate.feat_eng.test_indices
        x_train_inds = self.gp_surrogate.feat_eng.train_indices

        print("Prediction of new QoI")
        # TODO make predict call work on a (n_samples, n_features) np array
        y_pred = [self.gp_surrogate.predict(X_test[i, :].reshape(-1, 1))[0]
                  for i in range(X_test.shape[0])]
        y_var_pred = [self.gp_surrogate.predict(X_test[i, :].reshape(-1, 1))[1]
                      for i in range(X_test.shape[0])]
        y_t_len = len(y_test)

        y_pred_train = [self.gp_surrogate.predict(
            X_train[i, :].reshape(-1, 1))[0] for i in range(X_train.shape[0])]
        y_var_pred_train = [self.gp_surrogate.predict(
            X_train[i, :].reshape(-1, 1))[1] for i in range(X_train.shape[0])]

        # reshape the resulting arrays
        y_pred = np.squeeze(np.array(y_pred), axis=1)
        y_pred_train = np.squeeze(np.array(y_pred_train), axis=1)

        y_var_pred = np.squeeze(np.array(y_var_pred), axis=1)
        y_var_pred_train = np.squeeze(np.array(y_var_pred_train), axis=1)

        # check if we a working with a vector or scalar QoI, if vector -> consider
        # only the first component
        y_test_plot = y_test
        y_train_plot = y_train
        if y_pred.shape[1] is not 1:
            y_test_plot = y_test[:, [0]]
            y_train_plot = y_train[:, [0]]
            y_pred = y_pred[:, 0]
            y_pred_train = y_pred_train[:, 0]
            y_var_pred = y_var_pred[:, 0]
            y_var_pred_train = y_var_pred_train[:, 0]

        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
            y_pred_train = y_pred_train.reshape(-1, 1)

        # calculate the errors
        # test data appears smaller in length (by 1)
        err_abs = np.subtract(y_pred[:y_t_len, :], y_test)
        err_rel = np.divide(err_abs, y_test)

        print("Printing and plotting the evaluation results")
        self.plot_err(err_rel[:, 0], y_test[:, 0],
                      'rel. err. of prediction mean for test dataset in Ti fluxes')

        train_n = self.gp_surrogate.feat_eng.n_samples - y_t_len
        self.plot_res(x_test_inds, y_pred[:y_t_len, 0], y_test_plot[:, 0],
                      x_train_inds, y_pred_train[:, 0], y_train_plot[:, 0],
                      y_var_pred, y_var_pred_train,
                      r'$Y_i$', num='1', type_train='rand',
                      train_n=train_n, out_color='b')

        r2_test = self.get_r2_score(X_test, y_test)
        print('R2 score for the test data is : {:.3}'.format(r2_test))

        print('MSE of the GPR prediction is: {:.3}'.format(
            mse(y_pred[:y_t_len, 0], y_test_plot[:, 0])))

        self.y_pred = y_pred
        self.err_abs = err_abs
        self.err_rel = err_rel
        self.r2_test = r2_test
