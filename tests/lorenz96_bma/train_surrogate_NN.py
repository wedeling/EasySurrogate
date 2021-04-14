import numpy as np
import sys
import easysurrogate as es

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

lags = [1, 10] #[np.arange(start=1,stop=2,step=1)] 

if len(lags) == 0: 
    # no lagged features
    print('No lagged features.')
    features = X_data.flatten() 
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

# create a surrogate based on ANN (deterministic)
import tensorflow as tf
# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the first hidden layer; the input layer is added automatically 
ann.add(tf.keras.layers.Dense(units=32, activation='elu')) 

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=32, activation='elu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1))

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Training the ANN on the training set
ann.fit(features_train, target_train, batch_size = 64, epochs = 50)

# Predicting the Test set results
target_pred = ann.predict(features_test)

# Check the R squares and the mean squared error
from sklearn.metrics import mean_squared_error, r2_score
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(target_test, target_pred))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(target_test, target_pred))

#campaign.add_app(name='test_campaign_NN', surrogate=ann) 
ann.save('/home/federica/EasySurrogate/tests/lorenz96_bma/')
campaign.add_scalers(name='test_campaign_NN', scaler_features=sc_f, scaler_target=sc_t)

if len(lags) == 0:
    campaign.add_max_lag(name='test_campaign_NN', max_lag=0)
else:
    campaign.add_max_lag(name='test_campaign_NN', max_lag=max_lag)

campaign.save_state()

