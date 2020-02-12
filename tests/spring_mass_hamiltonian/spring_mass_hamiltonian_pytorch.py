import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es

class HNN(torch.nn.Module):
    '''Learn arbitrary vector fields that are sums of conservative and solenoidal fields'''
    def __init__(self, input_dim, differentiable_model, field_type='solenoidal',
                    baseline=False, assume_canonical_coords=True):
        super(HNN, self).__init__()
        self.baseline = baseline
        self.differentiable_model = differentiable_model
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim) # Levi-Civita permutation tensor
        self.field_type = field_type

    def forward(self, x):
        # traditional forward pass
        if self.baseline:
            return self.differentiable_model(x)

        y = self.differentiable_model(x)
        assert y.dim() == 2 and y.shape[1] == 2, "Output tensor should have shape [batch_size, 2]"
        return y.split(1,1)

    def time_derivative(self, x, t=None, separate_fields=False):
        '''NEURAL ODE-STLE VECTOR FIELD'''
        if self.baseline:
            return self.differentiable_model(x)

        '''NEURAL HAMILTONIAN-STLE VECTOR FIELD'''
        F1, F2 = self.forward(x) # traditional forward pass

        conservative_field = torch.zeros_like(x) # start out with both components set to 0
        solenoidal_field = torch.zeros_like(x)

        if self.field_type != 'solenoidal':
            dF1 = torch.autograd.grad(F1.sum(), x, create_graph=True)[0] # gradients for conservative field
            conservative_field = dF1 @ torch.eye(*self.M.shape)

        if self.field_type != 'conservative':
            dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True)[0] # gradients for solenoidal field
            solenoidal_field = dF2 @ self.M.t()
         
        if separate_fields:
            return [conservative_field, solenoidal_field]

        return conservative_field + solenoidal_field

    def permutation_tensor(self,n):
        M = None
        if self.assume_canonical_coords:
            M = torch.eye(n)
            M = torch.cat([M[n//2:], -M[:n//2]])
        else:
            '''Constructs the Levi-Civita permutation tensor'''
            M = torch.ones(n,n) # matrix of ones
            M *= 1 - torch.eye(n) # clear diagonals
            M[::2] *= -1 # pattern of signs
            M[:,::2] *= -1
    
            for i in range(n): # make asymmetric
                for j in range(i+1, n):
                    M[i,j] *= -1
        return M

class Net(nn.Module):
    
    def __init__(self, n_inputs, n_neurons, n_out):
        super(Net, self).__init__()
        
        self.n_inputs = n_inputs
        
        self.l1 = nn.Linear(n_inputs, n_neurons)
        self.l2 = nn.Linear(n_neurons, n_neurons)
        self.l3 = nn.Linear(n_neurons, n_out)
        
    def forward(self, x):
        
        x = x.view(-1, self.n_inputs)
        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(x))
        x = self.l3(x)        
        
        return x  

def step(p_n, q_n):
    
    p_np1 = p_n - dt*q_n
    q_np1 = q_n + dt*p_np1
    
    #Hamiltonian
    H_n = 0.5*p_n**2 + 0.5*q_n**2
    
    return p_np1, q_np1, H_n

def L2_loss(u, v):
  return (u-v).pow(2).mean()

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

# from data import get_dataset
# data = get_dataset()
# X_train = data['x']
# y_train = data['dx']

###########
# Pytorch #
###########

#create a neural net
net = Net(n_inputs = 2, n_neurons = 200, n_out = 2)

hnn = HNN(2, net)

#cross entropy loss function
criterion = nn.MSELoss()

optim = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

n_train = X_train.shape[0]

batch_size = 64

for epoch in range(1):
    net.train()
    for i in range(int(n_train)):
        # data, target = data.to(device), target.to(device)
        
        idx = np.random.randint(0, n_train, batch_size)
        feats = torch.tensor(X_train[idx], requires_grad=True, dtype=torch.float32)
        target = torch.tensor(y_train[idx], dtype=torch.float32)
        # target = target.view(batch_size)
        
        # optimizer.zero_grad()
        # output = net.forward(feats)
        
        # jac = torch.autograd.grad(output.sum(), feats, create_graph=True)[0]
               
        # test = jac @ torch.eye(2)
        
        test = hnn.time_derivative(feats)
      
        loss = criterion(test, target)
        loss.backward() ; optim.step() ; optim.zero_grad()
        
        if np.mod(i, 100) == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, i , n_train,
                100. * i / n_train, loss.item()))

# fig = plt.figure()
# ax = fig.add_subplot(111)

# y = torch.zeros([M, 2])

# feat = torch.tensor(X_train[0], requires_grad=True, dtype=torch.float32)
# p_n = feat[0]
# q_n = feat[1]

# for i in range(750):
#     # diff = net.forward(feat)[0]
    
#     diff = hnn.time_derivative(feat)
    
#     p_n = p_n + diff[0]*dt
#     q_n = q_n + diff[1]*dt

#     y[i, 0] = p_n; y[i, 1] = q_n

#     feat = y[i, :]
    
#     print(i)
    
# ax.plot(y[:, 0].detach().numpy(), y[:, 1].detach().numpy(), 'b+')

# plt.axis('equal')
# plt.tight_layout()

plt.show()