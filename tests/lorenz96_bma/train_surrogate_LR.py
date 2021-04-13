import numpy as np
import sys
import easysurrogate as es
import matplotlib.pyplot as plt

"""
Note that I create a surrogate that predicts a scalar quantity, i.e. the value of B_k for k in {0, ..., K-1}, 
as done by Rasp 2020, and not the full vector of B_k values for each k as done in Crommelin and Edeling 2021.
In this way the ML model for the surrogate should be simpler and there is no physical reason in the model 
to have a surrogate that predicts the full state (as if there were differences between k and \tilde{k})
"""

# create EasySurrogate campaign
campaign = es.Campaign()

# load HDF5 data frame
data_frame = campaign.load_hdf5_data()

# supervised training data set
X_data = data_frame['X_data']
B_data = data_frame['B_data']

lags = [] #[np.arange(start=1,stop=2,step=1)] # remember that the ending point of the interval is not included!

if len(lags) == 0: 
    # no lagged features
    print('No lagged features.')
    features = X_data.flatten() # don't care of the specific X_n; just want a surrogate that makes a "good" single estimate for all k
    target = B_data.flatten()
else:
    # check for negative lags
    if any(lag<0 for lag in lags):
        sys.exit("Error: defined negative lag.")
    else:
        # consider also lagged features
        max_lag = np.max(lags)
        if max_lag == 0:
            # no lagged features
            print('No lagged features.')
            features = X_data.flatten()
            target = B_data.flatten()
        else:
            print('Considering also lagged features.')
            target = B_data[max_lag:].flatten()
        
            L = len(X_data.flatten()) - max_lag*len(X_data[0])
            features = np.zeros((L,len(lags)+1))
            idx = 0
            features[:,idx] = X_data[max_lag:,:].flatten()
            for lag in lags:
                lag = np.int(lag)
                idx += 1
                features[:,idx] = X_data[max_lag-lag:-lag,:].flatten()
        

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.2)

# Standardize the data
from sklearn.preprocessing import StandardScaler
sc_f = StandardScaler()
sc_t = StandardScaler()

if (len(lags) == 0 or max_lag == 0):
    features_train = features_train.reshape(-1,1)
    features_test = features_test.reshape(-1,1)

features_train = sc_f.fit_transform(features_train)
features_test = sc_f.transform(features_test)

target_train = sc_t.fit_transform(target_train.reshape(-1,1))
target_test = sc_t.transform(target_test.reshape(-1,1))

# create a linear regression surrogate
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features_train, target_train)

# Getting the linear regression coefficients
print(regressor.coef_)
print(regressor.intercept_)

# Predicting the Test set results
target_pred = regressor.predict(features_test)

# Check the R squares and the mean squared error
from sklearn.metrics import mean_squared_error, r2_score
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(target_test, target_pred))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(target_test, target_pred))


if (len(lags) == 0 or max_lag == 0):
    # Plot data vs linear regression
    plt.figure() 
    plt.scatter(sc_f.inverse_transform(features_train), sc_t.inverse_transform(target_train), color = 'red', label='Training samples')
    plt.scatter(sc_f.inverse_transform(features_test), sc_t.inverse_transform(target_test), color = 'purple', label='Test samples')
    plt.plot(sc_f.inverse_transform(features_test), sc_t.inverse_transform(target_pred), color = 'blue', label='Predictions')
    plt.title('How good is my regressor?')
    plt.xlabel('Features')
    plt.ylabel('Target')
    plt.legend(loc='best')
    plt.show()

campaign.add_app(name='test_campaign_LR', surrogate=regressor)
campaign.add_scalers(name='test_campaign_LR', scaler_features=sc_f, scaler_target=sc_t)

campaign.save_state()

