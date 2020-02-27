import numpy as np
from scipy.stats import norm

class RNN_Layer:
    
    
    def __init__(self, W_in, W_h, 
                 n_neurons, r, n_layers, activation, loss, sequence_size,
                 bias = False, lamb = 0.0, on_gpu = False,
                 n_softmax = 0, **kwargs):

        #input weight matrix
        self.W_in = W_in
        self.W_h = W_h
        self.n_neurons = n_neurons
        self.r = r
        self.n_layers = n_layers
        self.activation = activation
        self.loss = loss
        self.sequence_size = sequence_size
        self.bias = bias
        self.lamb = lamb
        self.n_softmax = n_softmax
        
        self.hidden_history = []
        
        #use either numpy or cupy via xp based on the on_gpu flag
        global xp
        if on_gpu == False:
            import numpy as xp
        else:
            import cupy as xp

        if self.bias == True:
            self.n_bias = 1
        else:
            self.n_bias = 0
            
        #the number of classes in each softmax layer
        if self.n_softmax > 0:
            self.n_bins = int(self.n_neurons/self.n_softmax)
            
        self.init_h_tm1()
       
        #init momentum, squared grad and regularization matrices
        self.V_in = xp.zeros([W_in.shape[0], W_in.shape[1]])
        self.A_in = xp.zeros([W_in.shape[0], W_in.shape[1]])

        if r < n_layers:
            self.V_h = xp.zeros([W_h.shape[0], W_h.shape[1]])
            self.A_h = xp.zeros([W_h.shape[0], W_h.shape[1]])

        # self.Lamb = xp.ones([self.layer_rm1.n_neurons + self.layer_rm1.n_bias, self.n_neurons])*self.lamb
        
        # #do not apply regularization to the bias terms
        # if self.bias == True:
        #     self.Lamb[-1, :] = 0.0

        
    #connect this layer to its neighbors
    def meet_the_neighbors(self, layer_rm1, layer_rp1):
        #if this layer is an input layer
        if self.r == 0:
            self.layer_rm1 = None
            self.layer_rp1 = layer_rp1
        #if this layer is an output layer
        elif self.r == self.n_layers:
            self.layer_rm1 = layer_rm1
            self.layer_rp1 = None
        #if this layer is hidden
        else:
            self.layer_rm1 = layer_rm1
            self.layer_rp1 = layer_rp1
            
            
    def init_h_tm1(self):
        #initialize the hidden state of the previous time level to zero
        self.h_tm1 = np.zeros([self.n_neurons + self.n_bias, 1])
        if self.bias:
            self.h_tm1[-1] = 1.0
     

    #compute the output of the current layer in one shot using matrix - vector/matrix multiplication    
    def compute_output(self, x_t):

        #input to hidden layers        
        if self.r < self.n_layers:
            a_t = xp.dot(self.W_in.T, x_t) + xp.dot(self.W_h.T, self.h_tm1)
        #output layer has no W_h
        else:
            a_t = xp.dot(self.W_in.T, x_t)
       
        #apply activation to a_t
        if self.activation == 'linear':
            self.h_t = a_t
        elif self.activation == 'tanh':
            self.h_t = xp.tanh(a_t)
        else:
            print('Unknown activation type')
            import sys; sys.exit()
        
        #add bias neuron output
        if self.bias == True:
            self.h_t = xp.vstack([self.h_t, xp.ones(1)])
        self.a_t = a_t
        
        #overwrite value of h_tm1
        self.h_tm1 = np.copy(self.h_t)

        self.hidden_history.append(self.h_t)
        if len(self.hidden_history) > self.sequence_size:
            self.hidden_history.pop(0)

        #compute the gradient of the activation function, 
        self.compute_grad_Phi()
        
        return self.h_t

                
    #compute the gradient in the activation function Phi wrt its input
    def compute_grad_Phi(self):
        
        if self.activation == 'linear':
            self.grad_Phi = xp.ones([self.n_neurons, 1])
        elif self.activation == 'tanh':
            self.grad_Phi = 1.0 - self.h_t[0:self.n_neurons]**2


    #compute the value of the loss function
    def compute_loss(self, y_t):
        
        #only compute if in an output layer
        if self.layer_rp1 == None:
            
            h_t = self.h_t
            
            if self.loss == 'squared':
                self.L_t = self.L_t + (y_t - h_t)**2
            elif self.loss == 'cross_entropy':
                #compute values of the softmax layer
                #more than 1 (independent) softmax layer can be placed at the output
                o_i = []
                [o_i.append(xp.exp(h_i)/xp.sum(np.exp(h_i), axis=0)) for h_i in xp.split(h_t, self.n_softmax)]
                self.o_i = xp.concatenate(o_i)
                
                self.L_t = self.L_t - xp.sum(y_t*xp.log(self.o_i))
            else:
                print('Cannot compute loss: unknown loss and/or activation function')
                import sys; sys.exit()
               
  
    #initialize the value of delta_ho at the output layer
    def compute_delta_oo(self, y_t):
        
        #if the neuron is in the output layer, initialze delta_oo
        if self.layer_rp1 == None:
            h_t = self.h_t
            
            #compute the loss function
            self.compute_loss(y_t)
            
            #for binary classification
            if self.loss == 'squared':# and self.activation == 'linear':
                
                self.delta_ho = -2.0*(y_t - h_t)
                
            #for multinomial classification
            elif self.loss == 'cross_entropy':
         
                #(see eq. 3.22 of Aggarwal book)
                self.delta_ho = self.o_i - y_t
        else:
            print('Can only initialize delta_oo in output layer')
            import sys; sys.exit()
            
    #compute the gradient of the loss function wrt the activation functions of this layer
    def compute_delta_ho(self):
        #get the delta_ho values of the next layer (layer r+1)
        delta_ho_rp1 = self.layer_rp1.delta_ho
        
        #get the grad_Phi values of the next layer
        grad_Phi_rp1 = self.layer_rp1.grad_Phi
        
        #the weight matrix of the next layer
        W_rp1 = self.layer_rp1.W_in
        
        self.delta_ho = xp.dot(W_rp1, delta_ho_rp1*grad_Phi_rp1)[0:self.n_neurons, :]

    #compute the gradient of the loss function wrt the weights of this layer
    def compute_L_grad_W(self):
        
        delta_ho_grad_Phi = self.delta_ho*self.grad_Phi

        #output previous hidden layer r-1, current time level t
        h_rm1_t = self.layer_rm1.h_t
        
        #output current hidden layer r, previous time level t-1
        h_r_tm1 = self.h_tm1

        #cumulative add over all time levels of the training sequence
        self.L_grad_W_in = self.L_grad_W_in + xp.dot(h_rm1_t, delta_ho_grad_Phi.T)
        self.L_grad_W_h = self.L_grad_W_h + xp.dot(h_r_tm1, delta_ho_grad_Phi.T)
    
    #perform the backpropogation operations of the current layer
    def back_prop(self, y_i):
        
        if self.r == self.n_layers:
            self.compute_delta_oo(y_i)
            self.compute_L_grad_W()
        else:
            self.compute_delta_ho()
            self.compute_L_grad_W()
                    