from itertools import chain
import numpy as np
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
features = data_frame['X_data']
target = data_frame['B_data']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, target_train, target_test = train_test_split(features[::1,0], target[::1,0], test_size = 0.2)

# Standardize the data
from sklearn.preprocessing import StandardScaler
sc_f = StandardScaler()
sc_t = StandardScaler()

features_train = sc_f.fit_transform(features_train.reshape(-1,1))
features_test = sc_f.transform(features_test.reshape(-1,1))

target_train = sc_t.fit_transform(target_train.reshape(-1,1))
target_test = sc_t.transform(target_test.reshape(-1,1))

# create a linear regression surrogate
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#from sklearn.ensemble import RandomForestRegressor 
#regressor = RandomForestRegressor(n_estimators = 10) 

#from sklearn.svm import SVR
#regressor = SVR(kernel='rbf')

regressor.fit(features_train, target_train)

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


"""
Plot true data vs regressor
"""
plt.figure()
plt.scatter(sc_f.inverse_transform(features_train), sc_t.inverse_transform(target_train), color = 'red', label='Training samples')
plt.scatter(sc_f.inverse_transform(features_test), sc_t.inverse_transform(target_test), color = 'purple', label='Test samples')
plt.plot(sc_f.inverse_transform(features_test), sc_t.inverse_transform(target_pred), color = 'blue', label='Predictions')
plt.title('How good is my regressor?')
plt.xlabel('Features')
plt.ylabel('Target')
plt.legend(loc='best')
plt.show()

# create Quantized Softmax Network surrogate
#surrogate = es.methods.QSN_Surrogate()
#
## create time-lagged features
#lags = [[1, 10]]
#
## train the surrogate on the data
#n_iter = 20000
#surrogate.train([features], target, n_iter, lags=lags, n_layers=4, n_neurons=256,
#                batch_size=512)
#
#campaign.add_app(name='test_campaign', surrogate=surrogate)
#campaign.save_state()
#
## QSN analysis object
#analysis = es.analysis.QSN_analysis(surrogate)
#analysis.get_classification_error(index=np.arange(0, 10000))
