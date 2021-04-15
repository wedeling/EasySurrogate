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
import torch.nn as nn
import torch.nn.functional as Fn

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(features.shape[1], 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = Fn.elu(self.fc1(x))
        x = Fn.elu(self.fc2(x))
        x = self.fc3(x)
        return x

ann = ANN()    
print(ann)

# Create a stochastic gradient descent optimizer
from torch import optim
learning_rate = 0.01
#optimizer = optim.SGD(ann.parameters(), lr=learning_rate, momentum=0.9)
optimizer = optim.Adam(ann.parameters(), lr=learning_rate)
# Create a loss function
loss_fn = nn.MSELoss() # Mean Squared Error

# Training the ANN on the training set
import torch
features_train = torch.from_numpy(features_train).float()
target_train = torch.from_numpy(target_train).float()

batch_size = 64
trainloader = torch.utils.data.DataLoader(np.concatenate((features_train,target_train),axis=1), \
                                          batch_size=batch_size, shuffle=True)
#testloader = torch.utils.data.DataLoader(np.concatenate((features_test,target_test),axis=1), \
#                                         batch_size=batch_size, shuffle=True)

def round_tensor(t, decimal_places=3):
  return round(t.item(), decimal_places)

for epoch in range(50):
    for i, data in enumerate(trainloader, 0):
        # separate features and target
        inputs = data[:,:-1]
        target = data[:,-1]
        # clear accumulated gradients
        optimizer.zero_grad()
        # forward propagation
        target_pred = ann(inputs)
        target_pred = torch.squeeze(target_pred)
        # loss calculation
        loss = loss_fn(target_pred, target)
        # backward propagation
        loss.backward()
        # weight optimization
        optimizer.step()
            
    print(f'''epoch {epoch}
Train set - loss: {round_tensor(loss)}''')
    

# Predicting the Test set results
features_test = torch.from_numpy(features_test).float()
target_test = torch.from_numpy(target_test).float()

target_pred = ann(features_test)

# Check the R squares and the mean squared error
from sklearn.metrics import mean_squared_error, r2_score
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(target_test.detach().numpy()[:,:], target_pred.detach().numpy()[:,:]))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(target_test.detach().numpy()[:,:], target_pred.detach().numpy()[:,:]))

#MODEL_PATH = 'model.pth'
#torch.save(ann, MODEL_PATH) # ann = torch.load(MODEL_PATH)

campaign.add_app(name='test_campaign_NN_torch', surrogate=ann)
campaign.add_scalers(name='test_campaign_NN_torch', scaler_features=sc_f, scaler_target=sc_t)

if len(lags) == 0:
    lags = [0] # avoid adding empty string to the campaign object
campaign.add_lags(name='test_campaign_NN_torch', lags=lags)

campaign.save_state()

