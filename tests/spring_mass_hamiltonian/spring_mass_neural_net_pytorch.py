import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es

class Net(nn.Module):
    
    def __init__(self, n_inputs, n_neurons):
        super(Net, self).__init__()
        
        self.n_inputs = n_inputs
        
        self.l1 = nn.Linear(n_inputs, n_neurons)
        self.l2 = nn.Linear(n_neurons, n_neurons)
        self.l3 = nn.Linear(n_neurons, 2)
        
    def forward(self, x):
        
        x = x.view(-1, self.n_inputs)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)        
        
        return x  

def step(p_n, q_n):
    
    p_np1 = p_n - dt*q_n
    q_np1 = q_n + dt*p_np1
    
    #Hamiltonian
    H_n = 0.5*p_n**2 + 0.5*q_n**2
    
    return p_np1, q_np1, H_n

plt.close('all')

##################
# Initialisation #
##################

# distinguished particle:
q_n = 1  # position
p_n = 0  # momentum

#####################################
# Integration with symplectic Euler #
#####################################

dt = 0.01  # integration time step
M = 10**4

#####################
# Network parameters
#####################

#Feature engineering object - loads data file
feat_eng = es.methods.Feature_Engineering()
#get training data
h5f = feat_eng.get_hdf5_file()

p_n = h5f['p_n'][()].flatten()
q_n = h5f['q_n'][()].flatten()
dpdt = h5f['dpdt'][()].flatten()
dqdt = h5f['dqdt'][()].flatten()

X_train = np.zeros([M, 2])
X_train[:, 0] = p_n
X_train[:, 1] = q_n

y_train = np.zeros([M, 2])
y_train[:, 0] = dpdt
y_train[:, 1] = dqdt

###########
# Pytorch #
###########

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

#create a neural net
n_inputs = X_train.size(1)
net = Net(n_inputs = 2, n_neurons = 200)

#cross entropy loss function
criterion = nn.MSELoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

n_train = X_train.size()[0]

batch_size = 64

def train(epoch):
    net.train()
    for i in range(int(n_train)):
        # data, target = data.to(device), target.to(device)
        
        idx = np.random.randint(0, n_train, batch_size)
        feats = X_train[idx]
        target = y_train[idx]
        # target = target.view(batch_size)
        
        optimizer.zero_grad()
        output = net.forward(feats)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if np.mod(i, 100) == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, i , n_train,
                100. * i / n_train, loss.item()))
            
train(1)

fig = plt.figure()
ax = fig.add_subplot(111)

y = np.zeros([M, 2])

p_n = X_train[0][0]
q_n = X_train[0][1]

for i in range(M):
    
    feat = torch.Tensor([p_n, q_n])
    
    diff = net.forward(feat)[0]
    
    p_n = p_n + diff[0]*dt
    q_n = q_n + diff[1]*dt

    y[i, 0] = p_n; y[i, 1] = q_n
    
ax.plot(y[:, 0], y[:, 1], 'b+')

plt.axis('equal')
plt.tight_layout()

plt.show()