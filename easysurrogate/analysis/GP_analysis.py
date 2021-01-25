"""
CLASS TO PERFORM ANALYSIS ON RESULTS FROM A GAUSSIAN PROCESS SURROGATE
"""

from .base import BaseAnalysis
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error as mse

class GP_analysis(BaseAnalysis):
    """
    GP analysis class
    """

    def __init__(self, gp_surrogate, **kwargs):
        print('Creating GP_analysis object')
        self.gp_surrogate = gp_surrogate

    def plot_err(self, metric, original, name): #TODO: clean up and add subplots, use for abs error
        plt.ioff()
        #plt.figure(figsize=(2,1))
        #plt.subplot(311)
        #plt.title('training sample of size: {}'.format(str(len(train_n))))
        plt.title(name)
        plt.plot(range(len(metric)), metric, '.', label='GP metamodel error', color='b')
        #plt.plot(original, '.', label='Turbulence model', color='green')
        plt.xlabel('Run number')
        plt.ylabel('Prediction of {}'.format(name))
        plt.legend()
        plt.grid()
        plt.yscale("log")
        plt.savefig('gp_abs_err.png')

    def plot_res(prediction, original, name, num, type_train, train_n, out_color, output_folder):
        plt.ioff()
        plt.figure(figsize=(5, 11))
        plt.subplot(311)
        plt.title('{} training sample of size: {}'.format(type_train, str(len(train_n))))
        plt.plot(prediction, '.', label='GP metamodel', color=out_color)
        plt.plot(original, '.', label='Turbulence model', color='green')
        plt.xlabel('Run number')
        plt.ylabel('Prediction of {}'.format(name))
        plt.legend()
        plt.grid()
        plt.yscale("log")
        plt.subplot(312)
        err_abs = prediction - original
        print(err_abs.where(abs(err_abs) > 2e5))  # no where nparray
        plt.plot(err_abs, '.', color=out_color)
        plt.grid()
        plt.yscale("symlog")
        plt.xlabel('Run number')
        plt.ylabel('Error')
        plt.subplot(313)
        plt.plot(np.fabs(prediction - original) / original * 100, '.', color=out_color)
        plt.grid()
        plt.xlabel('Run number')
        plt.ylabel('Relative error (%)')
        # plt.yscale("log")
        plt.tight_layout()
        plt.savefig(output_folder + '/GP_prediction_' + num + '_' + type_train + '_' + str(len(train_n)) + '.png',
                    bbox_inches='tight', dpi=100)
        plt.clf()

    def get_regression_error(self, index=None, **kwargs):  # TODO add correlation analysis
        """
        Compute the regression RMSE error of GP surrogate
        Args:
            index: array
            indices of the dataset to form a test set

        Returns:
        prints RMSE regression error and plots the errors of the prediciton for different simulation runs/samples
        """
        if index is not None:
            X_test = self.gp_surrogate.model.X[index] # TODO store indexes at a model state
            y_test = self.gp_surrogate.model.y[index]
        else:
            X_test = self.gp_surrogate.X_test
            y_test = self.gp_surrogate.y_test

        #X_single = X_test[0, :] ### we use this a single sample to pass
        #y_pred_single = self.gp_surrogate.predict(X_single)

        print("Prediction of new fluxes")
        y_pred = [self.gp_surrogate.predict(X_test[i,:]) for i in range(X_test.shape[0])]

        y_pred = np.squeeze(np.array(y_pred), axis=1)
        err_abs = np.subtract(y_pred[:-1,:], y_test)  # TODO check why test appears smaller in length
        err_rel = np.divide(err_abs, y_test)

        self.plot_err(err_rel[:,0], y_test[:,0], 'relative error in Te fluxes')

        print('MSE of the GPR predistion is: {0}'.format(mse(y_pred[:-1,0], y_test[:,0])))
