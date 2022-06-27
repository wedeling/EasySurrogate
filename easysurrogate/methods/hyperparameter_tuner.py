"""
EasySurrogate Hyperparameter_tuner class
"""

#from msilib.schema import Error
from sklearn.model_selection import GridSearchCV

class Hyperparameter_tuner:
    """
    Object to train and analyse multiple surrogates within a single Campaign
    """

    def __init__(self, **kwargs):
        """
        Initialize a hyper-parameter tuner object
        """

        self.models = []

        if 'model' in kwargs:
            self.models.append(kwargs['model'])

    def set_optimization_parameters(self, params):
        """
        Save the value of optimizer parameters 
        """
        self.params=params

    def optimize(self, X, y, **kwargs):
        """
        Optimize hyperparameters of models aggregated by the tuner
        """
        
        for model in self.models:
            if model.backend == 'scikit-learn':
                regressors = GridSearchCV(model, self.params)
                #TODO: model has to be a fresh scikit-learn object, params has to be set in a way suitable for sklearn
                regressors.fit(X, y) 
                #TODO here the training and testing input/outputs have to known for .fit()
